"""Backfill BX data from existing parquet files in S3."""
import os
import logging
import psycopg2
import pandas as pd
from datetime import date, timedelta

logging.basicConfig(level=logging.INFO)

from parquet_storage import ParquetStorage
from bx_calculator import BXCalculator

def get_zone_mapping() -> dict:
    """Get node to zone mapping from database."""
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    cur.execute("SELECT pnode_id, zone FROM caiso.node_zone_mapping WHERE zone IS NOT NULL")
    mapping = {row[0]: row[1] for row in cur.fetchall()}
    cur.close()
    conn.close()
    return mapping

def compute_bx_from_df(df: pd.DataFrame, opr_date: date, bx_calc: BXCalculator, zone_mapping: dict):
    """Compute BX averages from DataFrame and store in PostgreSQL."""
    df['zone'] = df['node'].map(zone_mapping).fillna('UNMAPPED')
    
    for bx in range(4, 11):
        zone_bx = {}
        
        for zone in ['NP15', 'SP15', 'ZP26', 'Overall']:
            if zone == 'Overall':
                zone_df = df
            else:
                zone_df = df[df['zone'] == zone]
            
            if len(zone_df) == 0:
                continue
            
            zone_sorted = zone_df.sort_values(['node', 'mw'])
            zone_sorted['rank'] = zone_sorted.groupby('node').cumcount() + 1
            bx_df = zone_sorted[zone_sorted['rank'] <= bx]
            
            zone_bx[zone] = bx_df['mw'].mean()
        
        bx_calc.store_daily_bx_batch(opr_date, zone_bx, bx)

def backfill_bx_from_parquet():
    """Read existing parquet files and compute BX for any missing dates."""
    parquet = ParquetStorage()
    bx_calc = BXCalculator()
    zone_mapping = get_zone_mapping()
    
    # Get dates that have BX data
    conn = psycopg2.connect(os.environ['DATABASE_URL'])
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT opr_dt FROM caiso.bx_daily_summary")
    bx_dates = {row[0] for row in cur.fetchall()}
    cur.close()
    conn.close()
    print(f"Found {len(bx_dates)} dates with BX data")
    
    # Check all dates in 2024
    missing_dates = []
    for day_offset in range(366):
        check_date = date(2024, 1, 1) + timedelta(days=day_offset)
        if check_date not in bx_dates:
            # Check if parquet exists
            if parquet.check_date_exists(check_date):
                missing_dates.append(check_date)
    
    print(f"Found {len(missing_dates)} dates with parquet but no BX")
    
    if not missing_dates:
        print("No missing BX data to backfill")
        return
    
    # Process each missing date
    for i, opr_date in enumerate(missing_dates):
        print(f"Processing {i+1}/{len(missing_dates)}: {opr_date}")
        
        try:
            # Read parquet data
            table = parquet.read_day_from_parquet(opr_date)
            if table is None:
                print(f"  No data in parquet for {opr_date}")
                continue
            
            # Convert to pandas
            df = table.to_pandas()
            if df.empty:
                print(f"  Empty data for {opr_date}")
                continue
            
            # Compute and store BX
            compute_bx_from_df(df, opr_date, bx_calc, zone_mapping)
            print(f"  Stored BX for {opr_date}")
            
        except Exception as e:
            import traceback
            print(f"  Error processing {opr_date}: {e}")
            traceback.print_exc()

if __name__ == "__main__":
    backfill_bx_from_parquet()
