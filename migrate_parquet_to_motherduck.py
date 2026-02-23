"""
Migrate S3 parquet node data into MotherDuck table.

Creates a 'node_hourly_lmp' table and bulk-loads all parquet files.
Each parquet file has columns: node, mw, opr_hr, month, year
We add opr_dt derived from the filename.
"""
import os
import sys
import duckdb
import time

def migrate():
    token = os.getenv('MOTHERDUCK_TOKEN')
    if not token:
        print("ERROR: MOTHERDUCK_TOKEN not set")
        return False
    
    bucket = os.getenv('AWS_S3_BUCKET', 'oasis-data-for-replit-2025')
    aws_key = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
    
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
    
    conn.execute("""
        CREATE TABLE IF NOT EXISTS node_hourly_lmp (
            node VARCHAR,
            opr_dt DATE,
            opr_hr INTEGER,
            mw DOUBLE
        )
    """)
    
    existing = conn.execute("SELECT COUNT(*) as cnt FROM node_hourly_lmp").fetchone()[0]
    print(f"Existing rows in node_hourly_lmp: {existing:,}")
    
    if existing > 0:
        existing_years = conn.execute(
            "SELECT DISTINCT EXTRACT(YEAR FROM opr_dt)::INT as yr FROM node_hourly_lmp"
        ).fetchdf()
        loaded_years = set(existing_years['yr'].tolist())
    else:
        loaded_years = set()
    
    years_to_load = sys.argv[1:] if len(sys.argv) > 1 else ['2024', '2025', '2026']
    
    for year_str in years_to_load:
        year = int(year_str)
        
        if year in loaded_years:
            count = conn.execute(f"""
                SELECT COUNT(*) as cnt FROM node_hourly_lmp 
                WHERE EXTRACT(YEAR FROM opr_dt) = {year}
            """).fetchone()[0]
            print(f"Year {year}: already loaded ({count:,} rows), skipping")
            continue
        
        for month in range(1, 13):
            month_path = f"s3://{bucket}/lmp_parquet/year={year}/month={month:02d}/*.parquet"
            
            try:
                start = time.time()
                
                conn.execute(f"""
                    INSERT INTO node_hourly_lmp
                    SELECT 
                        node,
                        CAST(regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) AS DATE) as opr_dt,
                        opr_hr,
                        mw
                    FROM read_parquet('{month_path}', filename=true, hive_partitioning=true)
                """)
                
                elapsed = time.time() - start
                
                count = conn.execute(f"""
                    SELECT COUNT(*) as cnt FROM node_hourly_lmp 
                    WHERE EXTRACT(YEAR FROM opr_dt) = {year} 
                    AND EXTRACT(MONTH FROM opr_dt) = {month}
                """).fetchone()[0]
                
                print(f"  {year}-{month:02d}: {count:,} rows ({elapsed:.1f}s)")
                
            except Exception as e:
                if "No files found" in str(e):
                    pass
                else:
                    print(f"  {year}-{month:02d}: ERROR - {e}")
    
    final_count = conn.execute("SELECT COUNT(*) as cnt FROM node_hourly_lmp").fetchone()[0]
    final_dates = conn.execute("SELECT COUNT(DISTINCT opr_dt) as cnt FROM node_hourly_lmp").fetchone()[0]
    final_nodes = conn.execute("SELECT COUNT(DISTINCT node) as cnt FROM node_hourly_lmp").fetchone()[0]
    
    print(f"\n=== Migration Summary ===")
    print(f"Total rows: {final_count:,}")
    print(f"Total dates: {final_dates}")
    print(f"Total nodes: {final_nodes:,}")
    
    conn.close()
    return True

if __name__ == '__main__':
    migrate()
