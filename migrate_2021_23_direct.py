"""
Direct migration of 2021-2023 node LMP data from S3 ZIP files to MotherDuck.

Bypasses the intermediate Parquet step:
  S3 ZIP (2021.22.23/) → CSV parse → INSERT INTO node_hourly_lmp

Processes one month at a time, skips months already loaded.
"""
import os
import sys
import io
import re
import zipfile
import time
import logging
import boto3
import duckdb
import pandas as pd
from datetime import date

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

FOLDER = '2021.22.23/'
YEARS = [2021, 2022, 2023]


def connect_motherduck():
    token = os.environ.get('MOTHERDUCK_TOKEN')
    if not token:
        raise RuntimeError('MOTHERDUCK_TOKEN not set')
    conn = duckdb.connect(f'md:caiso_lmp?motherduck_token={token}')
    conn.execute("SET enable_progress_bar = false")
    return conn


def list_dam_lmp_files(s3, bucket, year):
    """List all DAM_LMP ZIP files for a given year from the 2021.22.23 folder."""
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    prefix = FOLDER
    pattern = re.compile(rf'^{FOLDER}{year}\d{{4}}_{year}\d{{4}}_DAM_LMP_GRP_N_N.*\.zip$')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if pattern.match(key):
                files.append(key)
    return sorted(files)


def parse_zip_to_records(zip_content, s3_key):
    """Extract and parse the PRC_LMP_DAM_LMP CSV from a ZIP."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zf:
            for name in zf.namelist():
                if 'PRC_LMP_DAM_LMP' in name and name.endswith('.csv'):
                    with zf.open(name) as f:
                        content = f.read().decode('utf-8')
                    return parse_caiso_csv(content, s3_key)
        logger.warning(f"No PRC_LMP_DAM_LMP CSV found in {s3_key}")
        return None, []
    except Exception as e:
        logger.error(f"Failed to parse {s3_key}: {e}")
        return None, []


def parse_caiso_csv(content, s3_key):
    """Parse CAISO DAM LMP CSV. Returns (opr_date, records list of dicts)."""
    try:
        lines = content.split('\n')
        header_idx = None
        for i, line in enumerate(lines):
            if 'INTERVALSTARTTIME_GMT' in line or 'OPR_DT' in line:
                header_idx = i
                break
        if header_idx is None:
            logger.warning(f"No header found in {s3_key}")
            return None, []

        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])))
        df.columns = [c.strip() for c in df.columns]

        price_col = 'VALUE' if 'VALUE' in df.columns else 'MW'
        required = {'OPR_DT', 'OPR_HR', 'NODE_ID', 'LMP_TYPE', price_col}
        if not required.issubset(set(df.columns)):
            logger.warning(f"Missing columns in {s3_key}: have {list(df.columns)}")
            return None, []

        df = df[df['LMP_TYPE'] == 'LMP'].copy()
        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df['OPR_HR'] = pd.to_numeric(df['OPR_HR'], errors='coerce')
        df = df.dropna(subset=[price_col, 'OPR_HR', 'NODE_ID'])
        df = df[df['OPR_HR'].between(1, 24)]

        if df.empty:
            return None, []

        opr_dt_str = str(df['OPR_DT'].iloc[0]).strip()
        try:
            opr_date = date.fromisoformat(opr_dt_str[:10])
        except Exception:
            m = re.search(r'(\d{4}-\d{2}-\d{2})', opr_dt_str)
            if m:
                opr_date = date.fromisoformat(m.group(1))
            else:
                logger.warning(f"Cannot parse OPR_DT '{opr_dt_str}' in {s3_key}")
                return None, []

        records = [
            {'node': str(r['NODE_ID']), 'opr_dt': opr_date,
             'opr_hr': int(r['OPR_HR']), 'mw': float(r[price_col])}
            for _, r in df.iterrows()
        ]
        return opr_date, records

    except Exception as e:
        logger.error(f"CSV parse error for {s3_key}: {e}")
        return None, []


def insert_records(conn, records):
    """Bulk insert records into node_hourly_lmp."""
    if not records:
        return 0
    df = pd.DataFrame(records)
    conn.register('_batch', df)
    conn.execute("""
        INSERT INTO node_hourly_lmp (node, opr_dt, opr_hr, mw)
        SELECT node, opr_dt, opr_hr, mw FROM _batch
    """)
    conn.unregister('_batch')
    return len(records)


def main():
    years_to_load = [int(y) for y in sys.argv[1:]] if len(sys.argv) > 1 else YEARS

    bucket = os.environ.get('AWS_S3_BUCKET', 'oasis-data-for-replit-2025')
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    )

    conn = connect_motherduck()

    logger.info("Fetching already-loaded dates from node_hourly_lmp...")
    existing_dates = set(
        str(r[0]) for r in conn.execute("""
            SELECT DISTINCT opr_dt FROM node_hourly_lmp
            WHERE EXTRACT(YEAR FROM opr_dt) IN (2021, 2022, 2023)
        """).fetchall()
    )
    logger.info(f"Already loaded: {len(existing_dates)} days for 2021-2023")

    total_inserted = 0
    total_files = 0
    total_errors = 0

    for year in years_to_load:
        logger.info(f"\n=== Year {year} ===")
        files = list_dam_lmp_files(s3, bucket, year)
        logger.info(f"Found {len(files)} DAM_LMP ZIP files for {year}")

        for s3_key in files:
            m = re.search(r'(\d{4})(\d{2})(\d{2})_', s3_key)
            if not m:
                continue
            file_date = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))

            if file_date.isoformat() in existing_dates:
                continue

            t0 = time.time()
            try:
                resp = s3.get_object(Bucket=bucket, Key=s3_key)
                zip_content = resp['Body'].read()
            except Exception as e:
                logger.error(f"Download failed {s3_key}: {e}")
                total_errors += 1
                continue

            opr_date, records = parse_zip_to_records(zip_content, s3_key)
            if not records:
                logger.warning(f"No records from {s3_key}")
                total_errors += 1
                continue

            n = insert_records(conn, records)
            elapsed = time.time() - t0
            logger.info(f"  {opr_date}: {n:,} rows ({elapsed:.1f}s) [{s3_key}]")
            total_inserted += n
            total_files += 1
            existing_dates.add(file_date.isoformat())

        year_count = conn.execute(f"""
            SELECT COUNT(*) FROM node_hourly_lmp
            WHERE EXTRACT(YEAR FROM opr_dt) = {year}
        """).fetchone()[0]
        logger.info(f"Year {year} total in DB: {year_count:,} rows")

    logger.info(f"\n=== Migration Complete ===")
    logger.info(f"Files processed: {total_files}")
    logger.info(f"Rows inserted: {total_inserted:,}")
    logger.info(f"Errors: {total_errors}")

    final = conn.execute("SELECT COUNT(*) FROM node_hourly_lmp").fetchone()[0]
    logger.info(f"Total rows in node_hourly_lmp: {final:,}")
    conn.close()


if __name__ == '__main__':
    main()
