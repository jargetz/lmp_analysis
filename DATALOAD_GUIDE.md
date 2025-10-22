# CAISO LMP Full Year Data Loading Guide

## Overview

The system is configured to load 312 ZIP files (full year of CAISO Day Ahead LMP data) from S3 into PostgreSQL. Due to the volume of data and platform execution time limits, the loading process uses a **resumable batch approach**.

## How Batch Loading Works

Each run of the loading script:
1. Processes up to `N` files (default: 20 files per batch)
2. Tracks which files have been processed (via `source_file` column)
3. Automatically skips already-processed files
4. Can be run multiple times until all 312 files are loaded

This approach provides:
- ✅ **Resumability**: If interrupted, just run again to continue
- ✅ **Progress tracking**: See exactly how many files remain
- ✅ **No duplicates**: Already-processed files are automatically skipped

## Loading the Full Year

### Option 1: Default Batch Size (Recommended)

Run the script multiple times. Each run processes up to 20 files:

```bash
# Clear existing data and start fresh
python3 load_full_year.py --fresh

# Continue loading (run this ~15 times to process all 312 files)
python3 load_full_year.py
python3 load_full_year.py
python3 load_full_year.py
...
```

**Estimated time**: ~10-15 minutes per batch, total ~3 hours for full year

### Option 2: Larger Batches (Faster but requires manual monitoring)

Process more files per batch:

```bash
# Fresh start with 50 files per batch
python3 load_full_year.py --fresh --batch=50

# Continue with 50-file batches (run ~6 times)
python3 load_full_year.py --batch=50
```

### Option 3: Shell Loop (Fully Automated)

```bash
# Clear data and load everything automatically
python3 load_full_year.py --fresh
while true; do
    python3 load_full_year.py
    # Check if done (script will indicate when complete)
    sleep 2
done
```

## After Loading: Run Preprocessing

Once all files are loaded, run the B6/B8 preprocessing:

```bash
python3 -c "from preprocessing import CAISOPreprocessor; CAISOPreprocessor().run_full_preprocessing()"
```

This calculates operational hours for B6 (bottom 6 hours) and B8 (bottom 8 hours) pricing analysis.

## Monitoring Progress

Each batch run shows:
```
📊 PROGRESS:
  - Total files in S3: 312
  - Files processed: 40
  - Remaining: 272

💡 Run again to process next batch (up to 20 files)
   Estimated runs needed: 14
```

## Troubleshooting

### "All files already processed!"
✅ You're done! All 312 files are loaded. Run preprocessing next.

### Duplicate key errors
The script automatically skips files that are already in the database. If you see this error, it means the duplicate detection is working correctly.

### Want to start over?
```bash
python3 load_full_year.py --fresh
```
This clears all existing data and starts from scratch.

## Performance Notes

- Each file: ~20-30 seconds to download, process, and insert
- 20-file batch: ~10-15 minutes
- Full year (312 files): ~3-4 hours total across all batches
- Database will contain ~9-11 million records when complete

## Validation

After loading, run the baseline tests:
```bash
pytest test_analytics_baseline.py -v
```

This verifies that all analytics methods work correctly with the full dataset.
