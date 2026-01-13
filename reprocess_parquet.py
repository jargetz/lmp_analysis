#!/usr/bin/env python3
"""
Reprocess all parquet files with corrected OPR_HR values.

This script:
1. Lists all existing parquet files for a year
2. Deletes them from S3
3. Re-runs the data loader which uses OPR_HR column directly (not derived from GMT timestamp)

CRITICAL: opr_hr must come from CAISO's OPR_HR column (Pacific time 1-24),
NOT derived from INTERVALSTARTTIME_GMT (which causes an 8-hour offset).
"""

import boto3
import os
import logging
from s3_data_loader import S3DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def delete_parquet_files_for_year(year: int) -> int:
    """Delete all parquet files for a given year from S3."""
    s3 = boto3.client(
        's3',
        aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY')
    )
    bucket = os.environ.get('AWS_S3_BUCKET')
    prefix = f"lmp_parquet/year={year}/"
    
    deleted_count = 0
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            logger.info(f"Deleting {key}")
            s3.delete_object(Bucket=bucket, Key=key)
            deleted_count += 1
    
    return deleted_count


def reprocess_year(year: int, batch_size: int = 50):
    """Delete parquet files and reprocess from S3 zip files."""
    logger.info(f"Starting reprocess for year {year}")
    
    logger.info(f"Step 1: Deleting existing parquet files for {year}...")
    deleted = delete_parquet_files_for_year(year)
    logger.info(f"Deleted {deleted} parquet files")
    
    logger.info(f"Step 2: Reloading data from S3 zip files...")
    loader = S3DataLoader()
    
    def progress_callback(current, total, file_name, result):
        status = 'OK' if result.get('success') else 'SKIP' if result.get('skipped') else 'ERR'
        logger.info(f"[{current}/{total}] {status}: {file_name} - {result.get('records_saved', 0)} records")
    
    result = loader.load_all_data(
        progress_callback=progress_callback,
        batch_size=batch_size,
        calculate_bx=False
    )
    
    logger.info(f"Reprocessing complete: {result}")
    return result


if __name__ == '__main__':
    import sys
    
    year = 2024
    batch_size = 50
    
    if len(sys.argv) > 1:
        year = int(sys.argv[1])
    if len(sys.argv) > 2:
        batch_size = int(sys.argv[2])
    
    print(f"Reprocessing parquet files for year {year} with batch_size={batch_size}")
    print("This will DELETE existing parquet files and reprocess from original ZIP files.")
    print("The corrected files will use OPR_HR column directly (not derived from GMT timestamp).")
    print()
    
    result = reprocess_year(year, batch_size)
    print(f"\nFinal result: {result}")
