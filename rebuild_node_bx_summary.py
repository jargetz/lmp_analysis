"""
Rebuild node_bx_monthly_summary for specified years from node_hourly_lmp.

For each node, year, and month:
  - For BX types 4-10, ranks hours 1-24 by price ascending per day,
    averages the cheapest X hours, then averages those daily values
    across all days in the month.
  - Records days_count = number of distinct days with data.

Skips any year already present in the summary table unless --force is passed.

Usage:
  python rebuild_node_bx_summary.py 2021 2022 2023
  python rebuild_node_bx_summary.py --force 2024
"""
import os
import sys
import duckdb
import time


def main():
    token = os.environ.get('MOTHERDUCK_TOKEN')
    if not token:
        print("ERROR: MOTHERDUCK_TOKEN not set", file=sys.stderr)
        sys.exit(1)

    force = '--force' in sys.argv
    years_to_build = [int(a) for a in sys.argv[1:] if a != '--force']
    if not years_to_build:
        print("Usage: python rebuild_node_bx_summary.py [--force] <year1> [year2] ...")
        sys.exit(1)

    conn = duckdb.connect(f'md:caiso_lmp?motherduck_token={token}')
    conn.execute("SET enable_progress_bar = false")

    existing_years = set(
        conn.execute(
            "SELECT DISTINCT year FROM node_bx_monthly_summary"
        ).fetchdf()['year'].tolist()
    )
    print(f"Existing years in node_bx_monthly_summary: {sorted(existing_years)}")

    bx_types = [4, 5, 6, 7, 8, 9, 10]

    for year in years_to_build:
        if year in existing_years and not force:
            cnt = conn.execute(
                f"SELECT COUNT(*) FROM node_bx_monthly_summary WHERE year = {year}"
            ).fetchone()[0]
            print(f"\nYear {year}: already has {cnt:,} rows — skipping (use --force to rebuild)")
            continue

        if year in existing_years and force:
            print(f"\nYear {year}: --force specified, deleting existing rows...")
            conn.execute(f"DELETE FROM node_bx_monthly_summary WHERE year = {year}")

        row_check = conn.execute(
            f"SELECT COUNT(*) FROM node_hourly_lmp WHERE EXTRACT(YEAR FROM opr_dt) = {year}"
        ).fetchone()[0]
        if row_check == 0:
            print(f"\nYear {year}: no rows in node_hourly_lmp — skipping")
            continue

        print(f"\nYear {year}: {row_check:,} rows in node_hourly_lmp")

        months_available = conn.execute(f"""
            SELECT DISTINCT EXTRACT(MONTH FROM opr_dt)::INT as m
            FROM node_hourly_lmp
            WHERE EXTRACT(YEAR FROM opr_dt) = {year}
            ORDER BY m
        """).fetchdf()['m'].tolist()

        for month in months_available:
            start = time.time()

            bx_cols = ", ".join(
                [f"AVG(CASE WHEN bx = {bx} THEN daily_avg END) AS b{bx}_avg" for bx in bx_types]
            )

            query = f"""
                INSERT INTO node_bx_monthly_summary
                    (node, year, month, b4_avg, b5_avg, b6_avg, b7_avg, b8_avg, b9_avg, b10_avg, days_count)
                WITH daily_ranked AS (
                    SELECT
                        node,
                        opr_dt,
                        opr_hr,
                        mw,
                        ROW_NUMBER() OVER (PARTITION BY node, opr_dt ORDER BY mw ASC) AS rn
                    FROM node_hourly_lmp
                    WHERE EXTRACT(YEAR FROM opr_dt) = {year}
                      AND EXTRACT(MONTH FROM opr_dt) = {month}
                      AND opr_hr <= 24
                ),
                daily_bx AS (
                    SELECT
                        node,
                        opr_dt,
                        bx,
                        AVG(mw) AS daily_avg
                    FROM daily_ranked
                    CROSS JOIN (VALUES (4),(5),(6),(7),(8),(9),(10)) AS t(bx)
                    WHERE rn <= bx
                    GROUP BY node, opr_dt, bx
                )
                SELECT
                    node,
                    {year} AS year,
                    {month} AS month,
                    {bx_cols},
                    COUNT(DISTINCT opr_dt) AS days_count
                FROM daily_bx
                GROUP BY node
            """
            conn.execute(query)

            elapsed = time.time() - start
            cnt = conn.execute(f"""
                SELECT COUNT(*) FROM node_bx_monthly_summary
                WHERE year = {year} AND month = {month}
            """).fetchone()[0]
            print(f"  {year}-{month:02d}: {cnt:,} nodes ({elapsed:.1f}s)")

        year_total = conn.execute(
            f"SELECT COUNT(*) FROM node_bx_monthly_summary WHERE year = {year}"
        ).fetchone()[0]
        print(f"  Year {year} total: {year_total:,} rows")

    final = conn.execute("SELECT COUNT(*) FROM node_bx_monthly_summary").fetchone()[0]
    final_years = conn.execute(
        "SELECT year, COUNT(*) as rows, COUNT(DISTINCT node) as nodes "
        "FROM node_bx_monthly_summary GROUP BY year ORDER BY year"
    ).fetchdf()
    print(f"\n=== Summary ===")
    print(f"Total rows: {final:,}")
    print(final_years.to_string(index=False))

    conn.close()


if __name__ == '__main__':
    main()
