#!/bin/bash
LOG=/tmp/migration_2021_23.log

echo "=== CAISO 2021-2023 Migration Pipeline ===" > "$LOG"
echo "Started: $(date)" >> "$LOG"
echo "" >> "$LOG"

echo "=== Step 1: Loading ZIPs directly into node_hourly_lmp ===" >> "$LOG"
python3 migrate_2021_23_direct.py 2021 2022 2023 >> "$LOG" 2>&1
echo "Step 1 done: $(date)" >> "$LOG"
echo "" >> "$LOG"

echo "=== Step 2: Rebuilding node_bx_monthly_summary ===" >> "$LOG"
python3 rebuild_node_bx_summary.py 2021 2022 2023 >> "$LOG" 2>&1
echo "Step 2 done: $(date)" >> "$LOG"
echo "" >> "$LOG"

echo "=== ALL DONE: $(date) ===" >> "$LOG"
