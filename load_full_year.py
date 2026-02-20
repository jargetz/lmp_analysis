"""
Script to load full year of CAISO LMP data from S3.
Raw data stored as parquet in S3, BX aggregates stored in MotherDuck.
"""

import logging
from s3_data_loader import S3DataLoader

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def progress_callback(current, total, message):
    """Print progress updates"""
    percent = int((current / total) * 100) if total > 0 else 0
    print(f"[{percent}%] {message}")

def main():
    import sys
    print("=" * 60)
    print("CAISO LMP Full Year Data Load (Resumable Batch Mode)")
    print("=" * 60)
    
    # Parse command line arguments
    batch_size = 20  # Default: process 20 files per run
    fresh_start = False
    calculate_bx = True  # Default: calculate BX for each day
    prefix = ''  # Default: no prefix (all files)
    
    for arg in sys.argv[1:]:
        if arg == '--fresh':
            fresh_start = True
        elif arg == '--no-bx':
            calculate_bx = False
        elif arg.startswith('--batch='):
            batch_size = int(arg.split('=')[1])
        elif arg.startswith('--prefix='):
            prefix = arg.split('=')[1]
    
    loader = S3DataLoader()
    
    # Check what's available
    print("\nChecking data freshness...")
    freshness = loader.check_data_freshness()
    print(f"MotherDuck currently has: {freshness.get('db_records', 0):,} summary records")
    print(f"S3 files available: {freshness.get('s3_files_available', 0)}")
    
    # Clear existing data if requested
    if fresh_start:
        print("\n⚠️  FRESH START MODE: Clearing existing data...")
        from motherduck_client import get_motherduck_client
        md = get_motherduck_client(force_new=True)
        md.execute_query("DELETE FROM bx_daily_summary")
        try:
            md.execute_query("DELETE FROM zone_hourly_lmp")
        except Exception:
            pass
        try:
            md.execute_query("DELETE FROM generator_bx_summary")
        except Exception:
            pass
        print("✅ Existing data cleared")
    
    bx_msg = "with BX calculation" if calculate_bx else "(no BX)"
    prefix_msg = f" from '{prefix}'" if prefix else ""
    print(f"\n📦 BATCH MODE: Processing up to {batch_size} files {bx_msg}{prefix_msg}")
    print("💡 TIP: Run this script multiple times to process all files incrementally")
    
    # Start the batch load
    print(f"\nStarting batch data load from S3{prefix_msg}...")
    print("-" * 60)
    
    result = loader.load_all_data(
        progress_callback=progress_callback,
        batch_size=batch_size,
        calculate_bx=calculate_bx,
        prefix=prefix
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("BATCH LOAD COMPLETE")
    print("=" * 60)
    print(f"Files in this batch: {result.get('total_files', 0)}")
    print(f"Files processed: {result.get('processed_files', 0)}")
    print(f"Files skipped: {result.get('skipped_files', 0)}")
    print(f"Records inserted: {result.get('total_records', 0):,}")
    print(f"BX days calculated: {result.get('bx_calculated_days', 0)}")
    
    if result.get('errors'):
        print(f"\nErrors: {len(result['errors'])}")
        for error in result['errors'][:3]:
            print(f"  - {error}")
    
    # Check remaining work
    all_files = loader.list_caiso_files(prefix=prefix)
    processed_count = sum(1 for f in all_files if loader.check_file_already_processed(f))
    remaining = len(all_files) - processed_count
    
    print(f"\n📊 PROGRESS{prefix_msg}:")
    print(f"  - Total files in S3: {len(all_files)}")
    print(f"  - Files processed: {processed_count}")
    print(f"  - Remaining: {remaining}")
    
    if remaining > 0:
        print(f"\n💡 Run again to process next batch (up to {batch_size} files)")
        print(f"   Estimated runs needed: {(remaining + batch_size - 1) // batch_size}")
    else:
        print("\n✅ All files processed!")
        
        # Run post-import aggregation to create monthly/annual summaries
        print("\n📊 Running post-import aggregation...")
        from bx_calculator import BXCalculator
        calc = BXCalculator()
        agg_result = calc.run_post_import_aggregation()
        if agg_result['success']:
            print(f"   Monthly summaries: {agg_result['monthly'].get('rows_affected', 0):,} records")
            print(f"   Annual summaries: {agg_result['annual'].get('rows_affected', 0):,} records")
            print("\n✅ Data ready for analysis!")

if __name__ == "__main__":
    main()
