"""
Subprocess-based MotherDuck query runner.
Runs queries in a separate process to avoid blocking Streamlit.
"""
import os
import sys
import json
import duckdb
import pandas as pd

def run_query():
    """Run a query passed via command line argument"""
    if len(sys.argv) < 2:
        print(json.dumps({'error': 'No query type specified'}))
        return
    
    query_type = sys.argv[1]
    
    token = os.getenv('MOTHERDUCK_TOKEN')
    if not token:
        print(json.dumps({'error': 'MOTHERDUCK_TOKEN not set'}))
        return
    
    try:
        conn = duckdb.connect(f'md:?motherduck_token={token}')
        conn.execute("SET enable_progress_bar = false")
        conn.execute("USE caiso_lmp")
        
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        if aws_key and aws_secret:
            conn.execute(f"""
                CREATE OR REPLACE SECRET s3_secret (
                    TYPE S3,
                    KEY_ID '{aws_key}',
                    SECRET '{aws_secret}',
                    REGION 'us-west-2'
                )
            """)
        
        if query_type == 'node_bx':
            bx = int(sys.argv[2])
            nodes = json.loads(sys.argv[3])
            year = int(sys.argv[4])
            result = get_node_bx(conn, bx, nodes, year)
        elif query_type == 'hourly_avg':
            nodes = json.loads(sys.argv[2])
            year = int(sys.argv[3])
            result = get_hourly_averages(conn, nodes, year)
        elif query_type == 'heatmap':
            nodes = json.loads(sys.argv[2])
            year = int(sys.argv[3])
            result = get_heatmap_data(conn, nodes, year)
        elif query_type == 'bx_trend':
            bx = int(sys.argv[2])
            nodes = json.loads(sys.argv[3])
            year = int(sys.argv[4])
            result = get_bx_trend(conn, bx, nodes, year)
        elif query_type == 'full_year_8760':
            nodes = json.loads(sys.argv[2])
            year = int(sys.argv[3])
            result = get_full_year_8760(conn, nodes, year)
        elif query_type == 'box_stats':
            bx = int(sys.argv[2])
            nodes = json.loads(sys.argv[3])
            year = int(sys.argv[4])
            result = get_box_stats(conn, bx, nodes, year)
        elif query_type == 'data_summary':
            result = get_data_summary(conn)
        elif query_type == 'unique_nodes':
            limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            result = get_unique_nodes(conn, limit)
        elif query_type == 'available_years':
            result = get_available_years(conn)
        elif query_type == 'all_nodes_from_summary':
            result = get_all_nodes_from_summary(conn)
        elif query_type == 'raw_sql':
            sql = sys.argv[2]
            params = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
            result = run_raw_sql(conn, sql, params)
        elif query_type == 'init_dashboard':
            result = init_dashboard(conn)
        elif query_type == 'zone_daily_bx':
            bx = int(sys.argv[2])
            year = int(sys.argv[3])
            result = get_zone_daily_bx(conn, bx, year)
        elif query_type == 'missing_days':
            year = int(sys.argv[2])
            result = get_missing_days(conn, year)
        elif query_type == 'multi_sql':
            queries = json.loads(sys.argv[2])
            result = run_multi_sql(conn, queries)
        else:
            result = {'error': f'Unknown query type: {query_type}'}
        
        conn.close()
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))

def get_node_bx(conn, bx, nodes, year):
    """Compute BX average for nodes with most common BX hours"""
    import re
    from collections import Counter
    nodes = [n for n in nodes if re.match(r'^[A-Za-z0-9_\-\.]+$', n)]
    if not nodes:
        return {'success': False, 'error': 'No valid nodes'}
    
    bucket = os.getenv('AWS_S3_BUCKET', 'oasis-data-for-replit-2025')
    path = f"s3://{bucket}/lmp_parquet/year={year}/**/*.parquet"
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    # Query for BX averages
    avg_query = f"""
        WITH file_data AS (
            SELECT 
                regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) as opr_dt,
                node, opr_hr, mw
            FROM read_parquet('{path}', filename=true, hive_partitioning=true)
            WHERE node IN ({node_list})
        ),
        ranked AS (
            SELECT opr_dt, node, opr_hr, mw,
                ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
            FROM file_data
        ),
        daily_bx AS (
            SELECT opr_dt, node, AVG(mw) as bx_price
            FROM ranked WHERE rn <= {bx}
            GROUP BY opr_dt, node
        )
        SELECT node, AVG(bx_price) as avg_price, MIN(bx_price) as min_price, 
               MAX(bx_price) as max_price, COUNT(DISTINCT opr_dt) as day_count
        FROM daily_bx GROUP BY node
    """
    
    result = conn.execute(avg_query).fetchdf()
    if result.empty:
        return {'success': False, 'error': 'No data found'}
    
    per_node = {row['node']: float(row['avg_price']) for _, row in result.iterrows()}
    
    # Query for most common BX hours per node
    hours_query = f"""
        WITH file_data AS (
            SELECT 
                regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) as opr_dt,
                node, opr_hr, mw
            FROM read_parquet('{path}', filename=true, hive_partitioning=true)
            WHERE node IN ({node_list})
        ),
        ranked AS (
            SELECT opr_dt, node, opr_hr, mw,
                ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
            FROM file_data
        )
        SELECT node, opr_hr, COUNT(*) as cnt
        FROM ranked WHERE rn <= {bx}
        GROUP BY node, opr_hr
        ORDER BY node, cnt DESC
    """
    
    hours_result = conn.execute(hours_query).fetchdf()
    per_node_hours = {}
    for node in nodes:
        node_hours = hours_result[hours_result['node'] == node].head(bx)
        per_node_hours[node] = [int(h) for h in node_hours['opr_hr'].tolist()]
    
    # Safely compute aggregate stats
    try:
        avg_price = float(result['avg_price'].mean()) if len(result) > 0 else 0.0
        min_price = float(result['min_price'].min()) if len(result) > 0 else 0.0
        max_price = float(result['max_price'].max()) if len(result) > 0 else 0.0
        day_count = int(result['day_count'].iloc[0]) if len(result) > 0 else 0
    except Exception as e:
        return {'success': False, 'error': f'Stats calculation error: {str(e)}'}
    
    return {
        'success': True,
        'avg_price': avg_price,
        'min_price': min_price,
        'max_price': max_price,
        'node_count': len(nodes),
        'day_count': day_count,
        'per_node': per_node,
        'per_node_hours': per_node_hours
    }

def get_hourly_averages(conn, nodes, year):
    """Get hourly price averages"""
    import re
    nodes = [n for n in nodes if re.match(r'^[A-Za-z0-9_\-\.]+$', n)]
    if not nodes:
        return []
    
    bucket = os.getenv('AWS_S3_BUCKET', 'oasis-data-for-replit-2025')
    path = f"s3://{bucket}/lmp_parquet/year={year}/**/*.parquet"
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    # Filter out hour 25 (DST transition days have 25 hours)
    query = f"""
        SELECT opr_hr as hour, AVG(mw) as avg_price
        FROM read_parquet('{path}', hive_partitioning=true)
        WHERE node IN ({node_list}) AND opr_hr <= 24
        GROUP BY opr_hr ORDER BY opr_hr
    """
    
    result = conn.execute(query).fetchdf()
    return [{'hour': int(r['hour']), 'avg_price': float(r['avg_price'])} for _, r in result.iterrows()]

def get_heatmap_data(conn, nodes, year):
    """Get month x hour heatmap data"""
    import re
    nodes = [n for n in nodes if re.match(r'^[A-Za-z0-9_\-\.]+$', n)]
    if not nodes:
        return []
    
    bucket = os.getenv('AWS_S3_BUCKET', 'oasis-data-for-replit-2025')
    path = f"s3://{bucket}/lmp_parquet/year={year}/**/*.parquet"
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    # Filter out hour 25 (DST transition days have 25 hours)
    query = f"""
        SELECT month, opr_hr as hour, AVG(mw) as avg_price
        FROM read_parquet('{path}', hive_partitioning=true)
        WHERE node IN ({node_list}) AND opr_hr <= 24
        GROUP BY month, opr_hr ORDER BY month, opr_hr
    """
    
    result = conn.execute(query).fetchdf()
    return [{'month': int(r['month']), 'hour': int(r['hour']), 'avg_price': float(r['avg_price'])} 
            for _, r in result.iterrows()]

def get_bx_trend(conn, bx, nodes, year):
    """Get daily BX trend for nodes"""
    import re
    nodes = [n for n in nodes if re.match(r'^[A-Za-z0-9_\-\.]+$', n)]
    if not nodes:
        return []
    
    bucket = os.getenv('AWS_S3_BUCKET', 'oasis-data-for-replit-2025')
    path = f"s3://{bucket}/lmp_parquet/year={year}/**/*.parquet"
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    query = f"""
        WITH file_data AS (
            SELECT 
                regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) as opr_dt,
                node, opr_hr, mw
            FROM read_parquet('{path}', filename=true, hive_partitioning=true)
            WHERE node IN ({node_list})
        ),
        ranked AS (
            SELECT opr_dt, node, opr_hr, mw,
                ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
            FROM file_data
        ),
        daily_bx AS (
            SELECT opr_dt, node, AVG(mw) as bx_price
            FROM ranked WHERE rn <= {bx}
            GROUP BY opr_dt, node
        )
        SELECT opr_dt as date, node, bx_price as avg_price
        FROM daily_bx ORDER BY opr_dt, node
    """
    
    result = conn.execute(query).fetchdf()
    return [{'date': str(r['date']), 'node': r['node'], 'avg_price': float(r['avg_price'])} 
            for _, r in result.iterrows()]

def get_box_stats(conn, bx, nodes, year):
    """Get summary statistics for box plot"""
    import re
    nodes = [n for n in nodes if re.match(r'^[A-Za-z0-9_\-\.]+$', n)]
    if not nodes:
        return []
    
    bucket = os.getenv('AWS_S3_BUCKET', 'oasis-data-for-replit-2025')
    path = f"s3://{bucket}/lmp_parquet/year={year}/**/*.parquet"
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    query = f"""
        WITH file_data AS (
            SELECT 
                regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) as opr_dt,
                node, opr_hr, mw
            FROM read_parquet('{path}', filename=true, hive_partitioning=true)
            WHERE node IN ({node_list})
        ),
        ranked AS (
            SELECT opr_dt, node, opr_hr, mw,
                ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
            FROM file_data
        ),
        daily_bx AS (
            SELECT opr_dt, node, AVG(mw) as bx_price
            FROM ranked WHERE rn <= {bx}
            GROUP BY opr_dt, node
        )
        SELECT node,
            MIN(bx_price) as min_price,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY bx_price) as q1,
            PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY bx_price) as median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY bx_price) as q3,
            MAX(bx_price) as max_price,
            AVG(bx_price) as avg_price
        FROM daily_bx GROUP BY node
    """
    
    result = conn.execute(query).fetchdf()
    return [{
        'node': r['node'],
        'min': float(r['min_price']),
        'q1': float(r['q1']),
        'median': float(r['median']),
        'q3': float(r['q3']),
        'max': float(r['max_price']),
        'avg': float(r['avg_price'])
    } for _, r in result.iterrows()]

def get_full_year_8760(conn, nodes, year):
    """Get full year hourly data for 8760 heatmap"""
    import re
    nodes = [n for n in nodes if re.match(r'^[A-Za-z0-9_\-\.]+$', n)]
    if not nodes:
        return []
    
    bucket = os.getenv('AWS_S3_BUCKET', 'oasis-data-for-replit-2025')
    path = f"s3://{bucket}/lmp_parquet/year={year}/**/*.parquet"
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    # Filter out hour 25 (DST transition days have 25 hours)
    query = f"""
        SELECT 
            regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) as opr_dt,
            opr_hr, 
            AVG(mw) as avg_price
        FROM read_parquet('{path}', filename=true, hive_partitioning=true)
        WHERE node IN ({node_list}) AND opr_hr <= 24
        GROUP BY 1, opr_hr
        ORDER BY 1, opr_hr
    """
    
    result = conn.execute(query).fetchdf()
    return [{'opr_dt': str(r['opr_dt']), 'opr_hr': int(r['opr_hr']), 'avg_price': float(r['avg_price'])} 
            for _, r in result.iterrows()]

def get_zone_daily_bx(conn, bx, year):
    """Compute daily BX values for NP15, SP15, ZP26 from zone_hourly_lmp"""
    import datetime
    query = f"""
        WITH ranked AS (
            SELECT zone, opr_dt, hour_num, lmp,
                ROW_NUMBER() OVER (PARTITION BY zone, opr_dt ORDER BY lmp ASC) as rn
            FROM zone_hourly_lmp
            WHERE EXTRACT(YEAR FROM opr_dt) = {int(year)}
              AND hour_num <= 24
        )
        SELECT zone, CAST(opr_dt AS VARCHAR) as opr_dt, 
               ROUND(AVG(lmp), 5) as bx_price,
               COUNT(*) as hours_used
        FROM ranked 
        WHERE rn <= {int(bx)}
        GROUP BY zone, opr_dt
        ORDER BY zone, opr_dt
    """
    result = conn.execute(query).fetchdf()
    if result.empty:
        return []
    rows = []
    for _, row in result.iterrows():
        rows.append({
            'zone': str(row['zone']),
            'opr_dt': str(row['opr_dt']),
            'bx_price': float(row['bx_price']),
            'hours_used': int(row['hours_used'])
        })
    return rows

def get_missing_days(conn, year):
    """Find missing days in zone_hourly_lmp for a given year"""
    from datetime import date, timedelta
    query = f"""
        SELECT DISTINCT CAST(opr_dt AS VARCHAR) as dt 
        FROM zone_hourly_lmp 
        WHERE EXTRACT(YEAR FROM opr_dt) = {int(year)}
        ORDER BY dt
    """
    result = conn.execute(query).fetchdf()
    loaded = set(result['dt'].tolist()) if not result.empty else set()
    
    start = date(int(year), 1, 1)
    end = date(int(year), 12, 31)
    all_days = set()
    d = start
    while d <= end:
        all_days.add(d.isoformat())
        d += timedelta(days=1)
    
    missing = sorted(all_days - loaded)
    return {
        'year': int(year),
        'total_expected': len(all_days),
        'total_loaded': len(loaded),
        'missing_count': len(missing),
        'missing_dates': missing
    }

def get_all_individual_nodes(conn):
    """Get all distinct node names from node_zone_mapping + generator_bx_summary for node search"""
    try:
        result = conn.execute("""
            SELECT DISTINCT name FROM (
                SELECT pnode_id as name FROM node_zone_mapping
                UNION
                SELECT node as name FROM generator_bx_summary
            ) ORDER BY name
        """).fetchdf()
        return result['name'].tolist() if not result.empty else []
    except Exception:
        return []

def init_dashboard(conn):
    """Get all data needed for initial dashboard load in one call"""
    result = {}
    result['data_summary'] = get_data_summary(conn)
    result['available_years'] = get_available_years(conn)
    result['all_nodes'] = get_all_nodes_from_summary(conn)
    result['individual_nodes'] = get_all_individual_nodes(conn)
    return result

def run_multi_sql(conn, queries):
    """Run multiple SQL queries in a single subprocess call. 
    queries is a dict of {key: sql_string} or {key: [sql_string, params]}"""
    results = {}
    for key, q in queries.items():
        try:
            if isinstance(q, list):
                sql, params = q[0], q[1] if len(q) > 1 else None
            else:
                sql, params = q, None
            rows = run_raw_sql(conn, sql, params)
            results[key] = rows
        except Exception as e:
            results[key] = {'error': str(e)}
    return results

def run_raw_sql(conn, sql, params=None):
    """Run arbitrary SQL and return results as list of dicts"""
    import datetime
    if params:
        result = conn.execute(sql, params).fetchdf()
    else:
        result = conn.execute(sql).fetchdf()
    if result.empty:
        return []
    rows = []
    for _, row in result.iterrows():
        d = {}
        for col in result.columns:
            val = row[col]
            if isinstance(val, (datetime.date, datetime.datetime)):
                d[col] = str(val)
            elif isinstance(val, float) and pd.isna(val):
                d[col] = None
            elif hasattr(val, 'item'):
                d[col] = val.item()
            else:
                d[col] = val
        rows.append(d)
    return rows

def get_available_years(conn):
    """Get list of years with data available"""
    try:
        result = conn.execute(
            "SELECT DISTINCT EXTRACT(YEAR FROM opr_dt)::INTEGER as year FROM zone_hourly_lmp ORDER BY year DESC"
        ).fetchdf()
        if not result.empty:
            return sorted(result['year'].tolist(), reverse=True)
    except Exception:
        pass
    try:
        result = conn.execute(
            "SELECT DISTINCT EXTRACT(YEAR FROM opr_dt)::INTEGER as year FROM bx_daily_summary ORDER BY year DESC"
        ).fetchdf()
        if not result.empty:
            return sorted(result['year'].tolist(), reverse=True)
    except Exception:
        pass
    return [2024]

def get_all_nodes_from_summary(conn):
    """Get all distinct node names from bx_daily_summary"""
    result = conn.execute("SELECT DISTINCT node FROM bx_daily_summary ORDER BY node").fetchdf()
    return result['node'].tolist() if not result.empty else []

def get_data_summary(conn):
    """Get summary stats from bx_daily_summary table"""
    result = conn.execute("""
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT node) as unique_nodes,
            MIN(opr_dt) as earliest_date,
            MAX(opr_dt) as latest_date,
            AVG(avg_price) as avg_price,
            MIN(avg_price) as min_price,
            MAX(avg_price) as max_price
        FROM bx_daily_summary
    """).fetchdf()
    if result.empty:
        return {}
    row = result.iloc[0]
    return {
        'total_records': int(row['total_records']),
        'unique_nodes': int(row['unique_nodes']),
        'earliest_date': str(row['earliest_date']),
        'latest_date': str(row['latest_date']),
        'avg_price': float(row['avg_price']) if pd.notna(row['avg_price']) else None,
        'min_price': float(row['min_price']) if pd.notna(row['min_price']) else None,
        'max_price': float(row['max_price']) if pd.notna(row['max_price']) else None
    }

def get_unique_nodes(conn, limit=5):
    """Get unique node names from bx_daily_summary"""
    result = conn.execute(f"SELECT DISTINCT node FROM bx_daily_summary ORDER BY node LIMIT {int(limit)}").fetchdf()
    return [str(n) for n in result['node'].tolist()]

if __name__ == '__main__':
    run_query()
