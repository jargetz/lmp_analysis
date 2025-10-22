"""
Script to load full year of CAISO LMP data from S3 into PostgreSQL database.
This will process all available ZIP files from S3.
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
    print("CAISO LMP Full Year Data Load")
    print("=" * 60)
    
    loader = S3DataLoader()
    
    # Check what's available
    print("\nChecking data freshness...")
    freshness = loader.check_data_freshness()
    print(f"Database currently has: {freshness.get('db_records', 0):,} records")
    print(f"S3 files available: {freshness.get('s3_files_available', 0)}")
    
    # Clear existing data if requested
    if len(sys.argv) > 1 and sys.argv[1] == '--fresh':
        print("\n⚠️  FRESH START MODE: Clearing existing data...")
        from database import DatabaseManager
        db = DatabaseManager()
        db.execute_query("TRUNCATE TABLE caiso.lmp_data RESTART IDENTITY CASCADE", fetch_all=False)
        print("✅ Existing data cleared")
    elif freshness.get('db_records', 0) > 0:
        print("\n⚠️  Database already has data. Use --fresh flag to clear and reload.")
        print("Proceeding will skip already-processed files...")
        response = input("Continue? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Start the load
    print("\nStarting full data load from S3...")
    print("This may take a while (300+ files to process)")
    print("-" * 60)
    
    result = loader.load_all_data(progress_callback)
    
    # Print results
    print("\n" + "=" * 60)
    print("DATA LOAD COMPLETE")
    print("=" * 60)
    print(f"Total files found: {result['total_files']}")
    print(f"Files processed: {result['processed_files']}")
    print(f"Files skipped (already loaded): {result['skipped_files']}")
    print(f"Total records inserted: {result['total_records']:,}")
    
    if result.get('errors'):
        print(f"\nErrors encountered: {len(result['errors'])}")
        for error in result['errors'][:5]:  # Show first 5 errors
            print(f"  - {error}")
    
    if result.get('preprocessing'):
        prep = result['preprocessing']
        print(f"\nPreprocessing:")
        print(f"  - B6 hours calculated: {prep.get('b6_hours', 0)}")
        print(f"  - B8 hours calculated: {prep.get('b8_hours', 0)}")
    
    print("\n✅ Full year data load complete!")

if __name__ == "__main__":
    main()
