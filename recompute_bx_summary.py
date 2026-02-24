"""
Recompute bx_daily_summary from zone_hourly_lmp.

For each zone (NP15, SP15, ZP26) and each day:
  - Takes hours 1-24 only (excludes hour 25 on DST days)
  - For each BX type (4-10), sorts hours by LMP ascending, averages cheapest X
  - Also computes "Overall" = simple average across the 3 zones for each day/bx_type

Replaces all zone-level rows in bx_daily_summary (preserves any node-level rows if they exist).
"""
import os
import sys
import duckdb
import json

def main():
    token = os.environ.get('MOTHERDUCK_TOKEN')
    if not token:
        print("ERROR: MOTHERDUCK_TOKEN not set", file=sys.stderr)
        sys.exit(1)
    
    conn = duckdb.connect(f'md:caiso_lmp?motherduck_token={token}')
    
    zones = ['NP15', 'SP15', 'ZP26']
    bx_types = [4, 5, 6, 7, 8, 9, 10]
    
    print("Step 1: Computing BX daily values from zone_hourly_lmp...")
    print(f"  Zones: {zones}")
    print(f"  BX types: {bx_types}")
    print(f"  Hours: 1-24 only (excluding hour 25)")
    
    conn.execute("CREATE TEMPORARY TABLE new_bx_daily AS SELECT * FROM bx_daily_summary WHERE 1=0")
    
    for bx in bx_types:
        print(f"  Computing B{bx}...")
        query = f"""
            INSERT INTO new_bx_daily (node, opr_dt, bx_type, avg_price, min_hour, max_hour)
            WITH ranked AS (
                SELECT 
                    zone,
                    opr_dt,
                    hour_num,
                    lmp,
                    ROW_NUMBER() OVER (PARTITION BY zone, opr_dt ORDER BY lmp ASC) as rn
                FROM zone_hourly_lmp
                WHERE hour_num <= 24
                  AND zone IN ('NP15', 'SP15', 'ZP26')
            ),
            zone_bx AS (
                SELECT 
                    zone as node,
                    opr_dt,
                    {bx} as bx_type,
                    AVG(lmp) as avg_price,
                    MIN(hour_num)::INTEGER as min_hour,
                    MAX(hour_num)::INTEGER as max_hour
                FROM ranked
                WHERE rn <= {bx}
                GROUP BY zone, opr_dt
            )
            SELECT * FROM zone_bx
        """
        conn.execute(query)
    
    print("Step 2: Computing 'Overall' (unweighted average across 3 zones)...")
    for bx in bx_types:
        query = f"""
            INSERT INTO new_bx_daily (node, opr_dt, bx_type, avg_price, min_hour, max_hour)
            SELECT 
                'Overall' as node,
                opr_dt,
                bx_type,
                AVG(avg_price) as avg_price,
                NULL as min_hour,
                NULL as max_hour
            FROM new_bx_daily
            WHERE bx_type = {bx}
              AND node IN ('NP15', 'SP15', 'ZP26')
            GROUP BY opr_dt, bx_type
        """
        conn.execute(query)
    
    new_count = conn.execute("SELECT COUNT(*) FROM new_bx_daily").fetchone()[0]
    new_zones = conn.execute("SELECT node, COUNT(*) as cnt FROM new_bx_daily GROUP BY node ORDER BY node").fetchdf()
    print(f"\nStep 3: New data computed: {new_count} total rows")
    print(new_zones.to_string(index=False))
    
    old_count = conn.execute(
        "SELECT COUNT(*) FROM bx_daily_summary WHERE node IN ('NP15', 'SP15', 'ZP26', 'Overall')"
    ).fetchone()[0]
    print(f"\nStep 4: Replacing old zone-level data ({old_count} rows) with new ({new_count} rows)...")
    
    conn.execute("DELETE FROM bx_daily_summary WHERE node IN ('NP15', 'SP15', 'ZP26', 'Overall')")
    conn.execute("INSERT INTO bx_daily_summary SELECT * FROM new_bx_daily")
    conn.execute("DROP TABLE new_bx_daily")
    
    final_count = conn.execute("SELECT COUNT(*) FROM bx_daily_summary").fetchone()[0]
    print(f"Done. bx_daily_summary now has {final_count} total rows.")
    
    print("\nStep 5: Spot-check B8 for NP15 on 2024-01-15...")
    check = conn.execute("""
        SELECT node, ROUND(avg_price, 4) as b8_price 
        FROM bx_daily_summary 
        WHERE bx_type = 8 AND node = 'NP15' AND opr_dt = '2024-01-15'
    """).fetchone()
    print(f"  Stored B8: ${check[1]}")
    
    hourly = conn.execute("""
        SELECT hour_num, lmp FROM zone_hourly_lmp 
        WHERE zone = 'NP15' AND opr_dt = '2024-01-15' AND hour_num <= 24
        ORDER BY lmp ASC LIMIT 8
    """).fetchdf()
    hand_calc = hourly['lmp'].mean()
    print(f"  Hand-calc B8 (avg of cheapest 8 from hourly): ${hand_calc:.4f}")
    print(f"  Match: {abs(check[1] - hand_calc) < 0.01}")
    
    conn.close()

if __name__ == '__main__':
    main()
