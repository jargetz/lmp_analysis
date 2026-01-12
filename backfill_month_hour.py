#!/usr/bin/env python3
"""
Backfill month_hour_summary table from S3 parquet files.

Optimized version using vectorized operations.
"""

import os
import logging
import pandas as pd
from datetime import date
from parquet_storage import ParquetStorage
from database import DatabaseManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def backfill_month_hour_summary():
    """Backfill month_hour_summary from parquet files using vectorized operations."""
    storage = ParquetStorage()
    db = DatabaseManager()
    
    zone_map = {}
    result = db.execute_query("SELECT pnode_id, zone FROM caiso.node_zone_mapping WHERE zone IS NOT NULL")
    if result:
        zone_map = {r['pnode_id']: r['zone'] for r in result}
    logger.info(f"Loaded {len(zone_map)} node-zone mappings")
    
    available_dates = storage.list_available_dates(year=2024)
    logger.info(f"Found {len(available_dates)} dates in parquet storage")
    
    if not available_dates:
        logger.error("No parquet files found")
        return
    
    all_chunks = []
    
    for i, d in enumerate(available_dates):
        if i % 50 == 0:
            logger.info(f"Processing {i+1}/{len(available_dates)}: {d}")
        
        table = storage.read_day_from_parquet(d)
        if table is None:
            continue
        
        df = table.to_pandas()
        if df.empty:
            continue
        
        df['zone'] = df['node'].map(zone_map)
        df['month'] = d.month
        
        zone_df = df[df['zone'].notna()].copy()
        zone_grouped = zone_df.groupby(['zone', 'month', 'opr_hr']).agg(
            total_mw=('mw', 'sum'),
            count=('mw', 'count')
        ).reset_index()
        
        overall_grouped = df.groupby(['month', 'opr_hr']).agg(
            total_mw=('mw', 'sum'),
            count=('mw', 'count')
        ).reset_index()
        overall_grouped['zone'] = 'Overall'
        
        all_chunks.append(zone_grouped)
        all_chunks.append(overall_grouped)
    
    logger.info("Aggregating all chunks...")
    combined = pd.concat(all_chunks, ignore_index=True)
    final = combined.groupby(['zone', 'month', 'opr_hr']).agg(
        total_mw=('total_mw', 'sum'),
        count=('count', 'sum')
    ).reset_index()
    final['avg_price'] = final['total_mw'] / final['count']
    
    logger.info(f"Deleting existing 2024 data and inserting {len(final)} records...")
    db.execute_query("DELETE FROM caiso.month_hour_summary WHERE year = 2024")
    
    for _, row in final.iterrows():
        db.execute_query(
            """INSERT INTO caiso.month_hour_summary 
               (zone, year, month, hour, avg_price, record_count)
               VALUES (%s, %s, %s, %s, %s, %s)""",
            (row['zone'], 2024, int(row['month']), int(row['opr_hr']), 
             float(row['avg_price']), int(row['count']))
        )
    
    logger.info(f"Done! Inserted {len(final)} month-hour summary records")

if __name__ == "__main__":
    backfill_month_hour_summary()
