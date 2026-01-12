"""
Import EIA CAISO zone hourly LMP data from CSV files.
Creates zone_hourly_lmp table and computes BX summaries.
"""

import os
import pandas as pd
from datetime import date
from database import DatabaseManager

def create_zone_tables(db: DatabaseManager):
    """Create tables for zone hourly data and BX summaries."""
    
    db.execute_query("""
        CREATE TABLE IF NOT EXISTS caiso.zone_hourly_lmp (
            id SERIAL PRIMARY KEY,
            opr_dt DATE NOT NULL,
            hour_num INTEGER NOT NULL,
            zone VARCHAR(10) NOT NULL,
            lmp DECIMAL(10,5) NOT NULL,
            congestion DECIMAL(10,5),
            energy DECIMAL(10,5),
            loss DECIMAL(10,5),
            UNIQUE(opr_dt, hour_num, zone)
        )
    """)
    
    db.execute_query("""
        CREATE INDEX IF NOT EXISTS idx_zone_hourly_lmp_date 
        ON caiso.zone_hourly_lmp(opr_dt)
    """)
    
    db.execute_query("""
        CREATE INDEX IF NOT EXISTS idx_zone_hourly_lmp_zone 
        ON caiso.zone_hourly_lmp(zone)
    """)
    
    print("Created zone_hourly_lmp table")


def import_csv_file(db: DatabaseManager, filepath: str):
    """Import a single EIA zone CSV file using batch insert."""
    
    print(f"Importing {filepath}...")
    
    df = pd.read_csv(filepath, skiprows=3)
    
    zone_cols = {
        'NP-15 LMP': ('NP15', 'NP-15 (Congestion)', 'NP-15 (Energy)', 'NP-15 (Loss)'),
        'SP-15 LMP': ('SP15', 'SP-15 (Congestion)', 'SP-15 (Energy)', 'SP-15 (Loss)'),
        'ZP-26 LMP': ('ZP26', 'ZP-26 (Congestion)', 'ZP-26 (Energy)', 'ZP-26 (Loss)'),
    }
    
    records = []
    for _, row in df.iterrows():
        opr_dt = pd.to_datetime(row['Local Date']).date()
        hour_num = int(row['Hour Number'])
        
        for lmp_col, (zone, cong_col, energy_col, loss_col) in zone_cols.items():
            lmp = float(row[lmp_col])
            congestion = float(row[cong_col]) if cong_col in row else None
            energy = float(row[energy_col]) if energy_col in row else None
            loss = float(row[loss_col]) if loss_col in row else None
            
            records.append((opr_dt, hour_num, zone, lmp, congestion, energy, loss))
    
    if records:
        import psycopg2.extras
        conn = db.get_connection()
        cur = conn.cursor()
        
        psycopg2.extras.execute_values(
            cur,
            """
            INSERT INTO caiso.zone_hourly_lmp (opr_dt, hour_num, zone, lmp, congestion, energy, loss)
            VALUES %s
            ON CONFLICT (opr_dt, hour_num, zone) DO UPDATE SET
                lmp = EXCLUDED.lmp,
                congestion = EXCLUDED.congestion,
                energy = EXCLUDED.energy,
                loss = EXCLUDED.loss
            """,
            records,
            page_size=1000
        )
        conn.commit()
        cur.close()
        
        print(f"  Imported {len(records)} records")
    
    return len(records)


def compute_zone_bx_summaries(db: DatabaseManager):
    """Compute BX summaries for each zone and day from zone_hourly_lmp using SQL."""
    
    print("Computing zone BX summaries...")
    
    zones = ['NP15', 'SP15', 'ZP26']
    bx_types = [4, 5, 6, 7, 8, 9, 10]
    
    import psycopg2.extras
    conn = db.get_connection()
    cur = conn.cursor()
    
    for zone in zones:
        print(f"  Processing {zone}...")
        
        for bx in bx_types:
            query = f"""
                INSERT INTO caiso.bx_daily_summary (opr_dt, node, bx_type, avg_price)
                SELECT 
                    opr_dt,
                    '{zone}' as node,
                    {bx} as bx_type,
                    AVG(lmp) as avg_price
                FROM (
                    SELECT opr_dt, lmp,
                        ROW_NUMBER() OVER (PARTITION BY opr_dt ORDER BY lmp ASC) as rn
                    FROM caiso.zone_hourly_lmp
                    WHERE zone = '{zone}'
                ) ranked
                WHERE rn <= {bx}
                GROUP BY opr_dt
                ON CONFLICT (opr_dt, node, bx_type) DO UPDATE SET
                    avg_price = EXCLUDED.avg_price
            """
            cur.execute(query)
    
    print("  Processing Overall...")
    for bx in bx_types:
        query = f"""
            INSERT INTO caiso.bx_daily_summary (opr_dt, node, bx_type, avg_price)
            SELECT 
                opr_dt,
                'Overall' as node,
                {bx} as bx_type,
                AVG(avg_lmp) as avg_price
            FROM (
                SELECT opr_dt, hour_num, AVG(lmp) as avg_lmp,
                    ROW_NUMBER() OVER (PARTITION BY opr_dt ORDER BY AVG(lmp) ASC) as rn
                FROM caiso.zone_hourly_lmp
                GROUP BY opr_dt, hour_num
            ) ranked
            WHERE rn <= {bx}
            GROUP BY opr_dt
            ON CONFLICT (opr_dt, node, bx_type) DO UPDATE SET
                avg_price = EXCLUDED.avg_price
        """
        cur.execute(query)
    
    conn.commit()
    cur.close()
    
    count_result = db.execute_query("""
        SELECT COUNT(*) as cnt FROM caiso.bx_daily_summary 
        WHERE node IN ('NP15', 'SP15', 'ZP26', 'Overall')
    """)
    total = count_result[0]['cnt'] if count_result else 0
    
    print(f"Computed {total} zone BX summary records")
    return total


def main():
    db = DatabaseManager()
    
    create_zone_tables(db)
    
    csv_files = [
        'attached_assets/caiso_lmp_da_hr_zones_2020_1768182367068.csv',
        'attached_assets/caiso_lmp_da_hr_zones_2021_1768182367068.csv',
        'attached_assets/caiso_lmp_da_hr_zones_2022_1768182367068.csv',
        'attached_assets/caiso_lmp_da_hr_zones_2023_1768182367068.csv',
        'attached_assets/caiso_lmp_da_hr_zones_2024_1768182367068.csv',
        'attached_assets/caiso_lmp_da_hr_zones_2025_1768182367068.csv',
    ]
    
    total_imported = 0
    for filepath in csv_files:
        if os.path.exists(filepath):
            total_imported += import_csv_file(db, filepath)
        else:
            print(f"Warning: {filepath} not found")
    
    print(f"\nTotal imported: {total_imported} hourly records")
    
    compute_zone_bx_summaries(db)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
