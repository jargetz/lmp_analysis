#!/usr/bin/env python3
"""
Process 2026 raw CAISO ZIP files from S3 into Parquet, then load into MotherDuck.

Usage:
    python3 process_2026_raw.py

Expects:
    - ZIPs already uploaded to s3://[bucket]/2026_raw/
    - AWS and MOTHERDUCK env vars set

Steps:
    1. Convert each ZIP in 2026_raw/ -> Parquet in lmp_parquet/year=2026/month=MM/
    2. Migrate new Parquet files into MotherDuck node_hourly_lmp
    3. Rebuild node_bx_monthly_summary for 2026
"""

import os
import sys
import logging
import subprocess

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def step1_convert_zips_to_parquet():
    logger.info("=" * 60)
    logger.info("STEP 1: Converting 2026_raw/ ZIPs → Parquet in S3")
    logger.info("=" * 60)

    from s3_data_loader import S3DataLoader
    loader = S3DataLoader()

    files = loader.list_caiso_files(prefix='2026_july5_raw/')
    if not files:
        logger.error("No DAM_LMP ZIP files found in s3://[bucket]/2026_raw/")
        logger.error("Make sure you uploaded the bulk ZIPs to that folder.")
        return False

    logger.info(f"Found {len(files)} ZIP file(s) in 2026_raw/")

    already_done = [f for f in files if loader.check_file_already_processed(f)]
    to_process   = [f for f in files if not loader.check_file_already_processed(f)]

    logger.info(f"  Already converted to Parquet: {len(already_done)}")
    logger.info(f"  To process now:               {len(to_process)}")

    if not to_process:
        logger.info("All ZIPs already converted — skipping to Step 2.")
        return True

    errors = []
    for i, key in enumerate(to_process, 1):
        logger.info(f"  [{i}/{len(to_process)}] Processing {key} ...")
        result = loader.download_and_process_file(key, calculate_bx=False)
        if result.get('success'):
            rows = result.get('records_saved', 0)
            parquet = result.get('parquet_key', '')
            logger.info(f"    ✓ {rows:,} rows → {parquet}")
        else:
            err = result.get('error', 'unknown error')
            logger.error(f"    ✗ FAILED: {err}")
            errors.append((key, err))

    if errors:
        logger.warning(f"{len(errors)} file(s) failed:")
        for key, err in errors:
            logger.warning(f"  {key}: {err}")
        return False

    logger.info("Step 1 complete.")
    return True


def step2_migrate_to_motherduck():
    logger.info("=" * 60)
    logger.info("STEP 2: Migrating Parquet → MotherDuck node_hourly_lmp")
    logger.info("=" * 60)

    import duckdb

    token = os.environ.get('MOTHERDUCK_TOKEN')
    bucket = os.environ.get('AWS_S3_BUCKET')
    aws_key = os.environ.get('AWS_ACCESS_KEY_ID')
    aws_secret = os.environ.get('AWS_SECRET_ACCESS_KEY')

    conn = duckdb.connect(f'md:?motherduck_token={token}')
    conn.execute("SET enable_progress_bar = false")
    conn.execute("USE caiso_lmp")

    if aws_key and aws_secret:
        conn.execute(f"""
            CREATE OR REPLACE SECRET s3_secret (
                TYPE S3,
                KEY_ID '{aws_key}',
                SECRET '{aws_secret}',
                REGION 'us-west-2'
            )
        """)

    existing = conn.execute(
        "SELECT COUNT(DISTINCT opr_dt) FROM node_hourly_lmp WHERE opr_dt >= '2026-01-01'"
    ).fetchone()[0]
    logger.info(f"2026 dates already in MotherDuck: {existing}")

    already_loaded = set(
        r[0].isoformat()
        for r in conn.execute(
            "SELECT DISTINCT opr_dt FROM node_hourly_lmp WHERE opr_dt >= '2026-01-01'"
        ).fetchall()
    )

    import boto3
    s3 = boto3.client('s3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret)

    paginator = s3.get_paginator('list_objects_v2')
    parquet_keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix='lmp_parquet/year=2026/'):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('.parquet'):
                date_str = key.split('/')[-1].replace('.parquet', '')
                if date_str not in already_loaded:
                    parquet_keys.append((date_str, key))

    parquet_keys.sort()
    logger.info(f"Parquet files to load into MotherDuck: {len(parquet_keys)}")

    if not parquet_keys:
        logger.info("Nothing new to migrate.")
        conn.close()
        return True

    loaded = 0
    for date_str, key in parquet_keys:
        s3_url = f"s3://{bucket}/{key}"
        try:
            conn.execute(f"""
                INSERT INTO node_hourly_lmp (node, opr_dt, opr_hr, mw)
                SELECT node,
                       DATE '{date_str}' AS opr_dt,
                       opr_hr,
                       mw
                FROM read_parquet('{s3_url}')
            """)
            rows = conn.execute(
                f"SELECT COUNT(*) FROM node_hourly_lmp WHERE opr_dt = DATE '{date_str}'"
            ).fetchone()[0]
            logger.info(f"  ✓ {date_str}: {rows:,} rows")
            loaded += 1
        except Exception as e:
            logger.error(f"  ✗ {date_str}: {e}")

    conn.close()
    logger.info(f"Step 2 complete — {loaded} date(s) loaded.")
    return True


def step3_rebuild_bx_summary():
    logger.info("=" * 60)
    logger.info("STEP 3: Rebuilding node_bx_monthly_summary for 2026")
    logger.info("=" * 60)

    result = subprocess.run(
        [sys.executable, 'rebuild_node_bx_summary.py', '2026', '--force'],
        capture_output=False
    )
    if result.returncode != 0:
        logger.error("rebuild_node_bx_summary.py failed")
        return False

    logger.info("Step 3 complete.")
    return True


if __name__ == '__main__':
    logger.info("Starting 2026 data load pipeline")

    if not step1_convert_zips_to_parquet():
        logger.error("Step 1 failed — check errors above.")
        sys.exit(1)

    if not step2_migrate_to_motherduck():
        logger.error("Step 2 failed — check errors above.")
        sys.exit(1)

    if not step3_rebuild_bx_summary():
        logger.error("Step 3 failed — check errors above.")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("All done! 2026 data fully loaded.")
    logger.info("=" * 60)
