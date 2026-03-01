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
            source = sys.argv[4] if len(sys.argv) > 4 else 'zone_hourly'
            result = get_zone_daily_bx(conn, bx, year, source)
        elif query_type == 'missing_days':
            year = int(sys.argv[2])
            result = get_missing_days(conn, year)
        elif query_type == 'node_coverage':
            year = int(sys.argv[2])
            result = get_node_coverage(conn, year)
        elif query_type == 'monthly_bx_spotcheck':
            bx = int(sys.argv[2])
            year = int(sys.argv[3])
            zone = sys.argv[4] if len(sys.argv) > 4 else 'SP15'
            result = get_monthly_bx_spotcheck(conn, bx, year, zone)
        elif query_type == 'multi_sql':
            queries = json.loads(sys.argv[2])
            result = run_multi_sql(conn, queries)
        elif query_type == 'node_map':
            bx = int(sys.argv[2])
            year = int(sys.argv[3])
            time_period = sys.argv[4]
            month = int(sys.argv[5]) if len(sys.argv) > 5 and sys.argv[5] else None
            result = get_node_map_data(conn, bx, year, time_period, month)
        elif query_type == 'facility_emissions':
            result = get_facility_emissions(conn)
        else:
            result = {'error': f'Unknown query type: {query_type}'}
        
        conn.close()
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))

def get_node_bx(conn, bx, nodes, year):
    """Compute BX average for nodes. Uses pre-computed monthly summary when available,
    falls back to scanning node_hourly_lmp directly for years not in the summary."""
    import re
    nodes = [n for n in nodes if re.match(r'^[A-Za-z0-9_\-\.]+$', n)]
    if not nodes:
        return {'success': False, 'error': 'No valid nodes'}

    col = f'b{bx}_avg'
    node_list = ', '.join(f"'{n}'" for n in nodes)

    summary_query = f"""
        SELECT node,
               SUM({col} * days_count) / SUM(days_count) AS avg_price,
               SUM(days_count) AS day_count
        FROM node_bx_monthly_summary
        WHERE node IN ({node_list})
          AND year = {int(year)}
        GROUP BY node
    """
    result = conn.execute(summary_query).fetchdf()

    if result.empty:
        fallback_query = f"""
            WITH ranked AS (
                SELECT opr_dt, node, opr_hr, mw,
                    ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) AS rn
                FROM node_hourly_lmp
                WHERE node IN ({node_list})
                  AND EXTRACT(YEAR FROM opr_dt) = {int(year)}
                  AND opr_hr BETWEEN 1 AND 24
            ),
            daily_bx AS (
                SELECT node, opr_dt, AVG(mw) AS daily_avg, COUNT(*) AS hrs
                FROM ranked WHERE rn <= {bx}
                GROUP BY node, opr_dt
            )
            SELECT node,
                   AVG(daily_avg) AS avg_price,
                   COUNT(DISTINCT opr_dt) AS day_count
            FROM daily_bx
            GROUP BY node
        """
        result = conn.execute(fallback_query).fetchdf()
        if result.empty:
            return {'success': False, 'error': f'No data found for {year} — year may not be loaded'}

    per_node = {row['node']: float(row['avg_price']) for _, row in result.iterrows()}

    hours_query = f"""
        WITH ranked AS (
            SELECT opr_dt, node, opr_hr, mw,
                ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
            FROM node_hourly_lmp
            WHERE node IN ({node_list})
              AND EXTRACT(YEAR FROM opr_dt) = {int(year)}
              AND opr_hr <= 24
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

    avg_price = float(result['avg_price'].mean()) if len(result) > 0 else 0.0
    day_count = int(result['day_count'].iloc[0]) if len(result) > 0 else 0

    return {
        'success': True,
        'avg_price': avg_price,
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
    
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    query = f"""
        SELECT opr_hr as hour, AVG(mw) as avg_price
        FROM node_hourly_lmp
        WHERE node IN ({node_list}) AND opr_hr <= 24
          AND EXTRACT(YEAR FROM opr_dt) = {int(year)}
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
    
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    query = f"""
        SELECT EXTRACT(MONTH FROM opr_dt)::INT as month, opr_hr as hour, AVG(mw) as avg_price
        FROM node_hourly_lmp
        WHERE node IN ({node_list}) AND opr_hr <= 24
          AND EXTRACT(YEAR FROM opr_dt) = {int(year)}
        GROUP BY 1, opr_hr ORDER BY 1, opr_hr
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
    
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    query = f"""
        WITH ranked AS (
            SELECT opr_dt, node, opr_hr, mw,
                ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
            FROM node_hourly_lmp
            WHERE node IN ({node_list})
              AND EXTRACT(YEAR FROM opr_dt) = {int(year)}
              AND opr_hr <= 24
        ),
        daily_bx AS (
            SELECT opr_dt, node, AVG(mw) as bx_price
            FROM ranked WHERE rn <= {bx}
            GROUP BY opr_dt, node
        )
        SELECT CAST(opr_dt AS VARCHAR) as date, node, bx_price as avg_price
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
    
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    query = f"""
        WITH ranked AS (
            SELECT opr_dt, node, opr_hr, mw,
                ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
            FROM node_hourly_lmp
            WHERE node IN ({node_list})
              AND EXTRACT(YEAR FROM opr_dt) = {int(year)}
              AND opr_hr <= 24
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
    
    node_list = ', '.join(f"'{n}'" for n in nodes)
    
    query = f"""
        SELECT 
            CAST(opr_dt AS VARCHAR) as opr_dt,
            opr_hr, 
            AVG(mw) as avg_price
        FROM node_hourly_lmp
        WHERE node IN ({node_list}) AND opr_hr <= 24
          AND EXTRACT(YEAR FROM opr_dt) = {int(year)}
        GROUP BY opr_dt, opr_hr
        ORDER BY opr_dt, opr_hr
    """
    
    result = conn.execute(query).fetchdf()
    return [{'opr_dt': str(r['opr_dt']), 'opr_hr': int(r['opr_hr']), 'avg_price': float(r['avg_price'])} 
            for _, r in result.iterrows()]

def get_zone_daily_bx(conn, bx, year, source='zone_hourly'):
    """Get daily BX values for NP15, SP15, ZP26.
    source='zone_hourly' computes from zone_hourly_lmp (EIA load-weighted prices).
    source='node_avg' pulls from bx_daily_summary (unweighted node averages).
    """
    if source == 'node_avg':
        query = f"""
            SELECT node as zone, CAST(opr_dt AS VARCHAR) as opr_dt, 
                   avg_price as bx_price
            FROM bx_daily_summary
            WHERE bx_type = {int(bx)}
              AND EXTRACT(YEAR FROM opr_dt) = {int(year)}
              AND node IN ('NP15', 'SP15', 'ZP26')
            ORDER BY node, opr_dt
        """
    else:
        query = f"""
            WITH ranked AS (
                SELECT zone, opr_dt, hour_num, lmp,
                    ROW_NUMBER() OVER (PARTITION BY zone, opr_dt ORDER BY lmp ASC) as rn
                FROM zone_hourly_lmp
                WHERE EXTRACT(YEAR FROM opr_dt) = {int(year)}
                  AND hour_num <= 24
            )
            SELECT zone, CAST(opr_dt AS VARCHAR) as opr_dt, 
                   ROUND(AVG(lmp), 5) as bx_price
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
            'bx_price': float(row['bx_price'])
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

def get_node_coverage(conn, year):
    """Get node_hourly_lmp coverage stats for a given year"""
    from datetime import date, timedelta
    query = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(DISTINCT node) as node_count,
            CAST(MIN(opr_dt) AS VARCHAR) as earliest_date,
            CAST(MAX(opr_dt) AS VARCHAR) as latest_date,
            COUNT(DISTINCT opr_dt) as days_loaded
        FROM node_hourly_lmp
        WHERE EXTRACT(YEAR FROM opr_dt) = {int(year)}
          AND opr_hr <= 24
    """
    result = conn.execute(query).fetchdf()
    if result.empty or result.iloc[0]['total_rows'] == 0:
        return {
            'year': int(year),
            'has_data': False,
            'total_rows': 0,
            'node_count': 0,
            'days_loaded': 0,
            'total_expected': 366 if (int(year) % 4 == 0 and (int(year) % 100 != 0 or int(year) % 400 == 0)) else 365,
            'earliest_date': None,
            'latest_date': None
        }
    row = result.iloc[0]
    start = date(int(year), 1, 1)
    end = date(int(year), 12, 31)
    total_expected = (end - start).days + 1
    return {
        'year': int(year),
        'has_data': True,
        'total_rows': int(row['total_rows']),
        'node_count': int(row['node_count']),
        'days_loaded': int(row['days_loaded']),
        'total_expected': total_expected,
        'earliest_date': str(row['earliest_date']),
        'latest_date': str(row['latest_date'])
    }

def get_all_individual_nodes(conn):
    """Get all distinct node names from node_hourly_lmp + node_zone_mapping + generator_bx_summary"""
    try:
        result = conn.execute("""
            SELECT DISTINCT name FROM (
                SELECT DISTINCT node as name FROM node_hourly_lmp
                UNION
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
    try:
        zone_yr = conn.execute("""
            SELECT DISTINCT EXTRACT(YEAR FROM opr_dt)::INT as yr
            FROM zone_hourly_lmp
            ORDER BY yr DESC
        """).fetchdf()
        result['zone_years'] = [int(y) for y in zone_yr['yr'].tolist()] if not zone_yr.empty else [2024]
    except Exception:
        result['zone_years'] = [2024]
    try:
        node_yr = conn.execute("""
            SELECT DISTINCT EXTRACT(YEAR FROM opr_dt)::INT as yr
            FROM node_hourly_lmp
            ORDER BY yr DESC
        """).fetchdf()
        result['node_years'] = [int(y) for y in node_yr['yr'].tolist()] if not node_yr.empty else [2024]
    except Exception:
        result['node_years'] = [2024]
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
    all_years = set()
    for table in ['node_hourly_lmp', 'zone_hourly_lmp', 'bx_daily_summary']:
        try:
            result = conn.execute(
                f"SELECT DISTINCT EXTRACT(YEAR FROM opr_dt)::INTEGER as year FROM {table}"
            ).fetchdf()
            if not result.empty:
                all_years.update(result['year'].tolist())
        except Exception:
            pass
    return sorted(all_years, reverse=True) if all_years else [2024]

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

def get_monthly_bx_spotcheck(conn, bx, year, zone='SP15'):
    """Get monthly BX averages from all three methods for spot-checking."""
    from calendar import monthrange, isleap
    
    total_cal_days = 366 if isleap(year) else 365
    months = list(range(1, 13))
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    result = {'zone': zone, 'bx': bx, 'year': year, 'months': []}
    
    zone_node_map = {'NP15': 'TH_NP15_GEN-APND', 'SP15': 'TH_SP15_GEN-APND', 'ZP26': 'TH_ZP26_GEN-APND'}
    gen_node = zone_node_map.get(zone, f'TH_{zone}_GEN-APND')
    
    lw_monthly = conn.execute(f"""
        WITH daily_bx AS (
            SELECT opr_dt, 
                   AVG(lmp) as bx_price
            FROM (
                SELECT opr_dt, lmp,
                       ROW_NUMBER() OVER (PARTITION BY opr_dt ORDER BY lmp ASC) as rn
                FROM zone_hourly_lmp
                WHERE zone = ? AND EXTRACT(YEAR FROM opr_dt) = ? AND hour_num <= 24
            ) ranked
            WHERE rn <= ?
            GROUP BY opr_dt
        )
        SELECT EXTRACT(MONTH FROM opr_dt) as month,
               AVG(bx_price) as avg_price,
               COUNT(*) as day_count
        FROM daily_bx
        GROUP BY month
        ORDER BY month
    """, [zone, year, bx]).fetchdf()
    
    gen_monthly = conn.execute(f"""
        SELECT EXTRACT(MONTH FROM opr_dt) as month,
               AVG(avg_price) as avg_price,
               COUNT(*) as day_count
        FROM generator_bx_summary
        WHERE node = ? AND bx_type = ? AND EXTRACT(YEAR FROM opr_dt) = ?
        GROUP BY month
        ORDER BY month
    """, [gen_node, bx, year]).fetchdf()
    
    node_monthly = conn.execute(f"""
        SELECT EXTRACT(MONTH FROM opr_dt) as month,
               AVG(avg_price) as avg_price,
               COUNT(*) as day_count
        FROM bx_daily_summary
        WHERE node = ? AND bx_type = ? AND EXTRACT(YEAR FROM opr_dt) = ?
        GROUP BY month
        ORDER BY month
    """, [zone, bx, year]).fetchdf()
    
    lw_dict = {int(r['month']): {'avg': float(r['avg_price']), 'days': int(r['day_count'])} 
               for _, r in lw_monthly.iterrows()} if not lw_monthly.empty else {}
    gen_dict = {int(r['month']): {'avg': float(r['avg_price']), 'days': int(r['day_count'])} 
                for _, r in gen_monthly.iterrows()} if not gen_monthly.empty else {}
    node_dict = {int(r['month']): {'avg': float(r['avg_price']), 'days': int(r['day_count'])} 
                 for _, r in node_monthly.iterrows()} if not node_monthly.empty else {}
    
    lw_weighted_sum = 0
    gen_weighted_sum = 0
    node_weighted_sum = 0
    
    for m in months:
        _, cal_days = monthrange(year, m)
        lw_data = lw_dict.get(m, {})
        gen_data = gen_dict.get(m, {})
        node_data = node_dict.get(m, {})
        
        lw_avg = lw_data.get('avg')
        gen_avg = gen_data.get('avg')
        node_avg = node_data.get('avg')
        
        if lw_avg is not None:
            lw_weighted_sum += lw_avg * cal_days
        if gen_avg is not None:
            gen_weighted_sum += gen_avg * cal_days
        if node_avg is not None:
            node_weighted_sum += node_avg * cal_days
        
        result['months'].append({
            'month': month_names[m - 1],
            'month_num': m,
            'cal_days': cal_days,
            'load_weighted': round(lw_avg, 2) if lw_avg is not None else None,
            'load_weighted_days': lw_data.get('days', 0),
            'generator': round(gen_avg, 2) if gen_avg is not None else None,
            'generator_days': gen_data.get('days', 0),
            'node_avg': round(node_avg, 2) if node_avg is not None else None,
            'node_avg_days': node_data.get('days', 0),
        })
    
    result['annual'] = {
        'load_weighted': round(lw_weighted_sum / total_cal_days, 2) if lw_weighted_sum else None,
        'generator': round(gen_weighted_sum / total_cal_days, 2) if gen_weighted_sum else None,
        'node_avg': round(node_weighted_sum / total_cal_days, 2) if node_weighted_sum else None,
    }
    
    return result


def get_unique_nodes(conn, limit=5):
    """Get unique node names from bx_daily_summary"""
    result = conn.execute(f"SELECT DISTINCT node FROM bx_daily_summary ORDER BY node LIMIT {int(limit)}").fetchdf()
    return [str(n) for n in result['node'].tolist()]


def get_facility_emissions(conn):
    """Return all CARB facility emissions rows (2023 data)."""
    result = conn.execute('''
        SELECT facility, primary_sector, city, county, district,
               lat, lon, cap_and_trade,
               total_ghg, co2, nox, sox, pm10, pm25, diesel_pm
        FROM facility_emissions
        ORDER BY facility
    ''').fetchdf()
    return [
        {
            'facility':       str(r['facility']),
            'primary_sector': str(r['primary_sector']),
            'city':           str(r['city']),
            'county':         str(r['county']),
            'district':       str(r['district']),
            'lat':            float(r['lat']),
            'lon':            float(r['lon']),
            'cap_and_trade':  str(r['cap_and_trade']),
            'total_ghg':      float(r['total_ghg']) if r['total_ghg'] is not None else 0.0,
            'co2':            float(r['co2']) if r['co2'] is not None else 0.0,
            'nox':            float(r['nox']) if r['nox'] is not None else 0.0,
            'sox':            float(r['sox']) if r['sox'] is not None else 0.0,
            'pm10':           float(r['pm10']) if r['pm10'] is not None else 0.0,
            'pm25':           float(r['pm25']) if r['pm25'] is not None else 0.0,
            'diesel_pm':      float(r['diesel_pm']) if r['diesel_pm'] is not None else 0.0,
        }
        for _, r in result.iterrows()
    ]


def get_node_map_data(conn, bx, year, time_period, month=None):
    """Get node BX prices with coordinates for the geographic map."""
    col = f'b{bx}_avg'

    if time_period == 'Monthly' and month:
        price_sql = f"""
            SELECT node,
                   {col} AS avg_price,
                   days_count
            FROM node_bx_monthly_summary
            WHERE year = {int(year)} AND month = {int(month)}
        """
    else:
        price_sql = f"""
            SELECT node,
                   SUM({col} * days_count) / SUM(days_count) AS avg_price,
                   SUM(days_count) AS days_count
            FROM node_bx_monthly_summary
            WHERE year = {int(year)}
            GROUP BY node
        """

    query = f"""
        WITH prices AS ({price_sql}),
        mapped AS (
            SELECT p.node AS pnode_id,
                   p.avg_price,
                   c.lat,
                   c.lon,
                   c.node_type,
                   c.area,
                   z.zone
            FROM prices p
            JOIN pnode_coordinates c ON c.pnode_id = p.node
            LEFT JOIN node_zone_mapping z ON z.pnode_id = p.node
            WHERE c.lat IS NOT NULL AND c.lon IS NOT NULL
              AND c.lat BETWEEN 30 AND 50
              AND c.lon BETWEEN -130 AND -100
        )
        SELECT pnode_id, avg_price, lat, lon, node_type, area, zone
        FROM mapped
        ORDER BY pnode_id
    """

    result = conn.execute(query).fetchdf()
    return [
        {
            'pnode_id': str(r['pnode_id']),
            'avg_price': float(r['avg_price']) if r['avg_price'] is not None else None,
            'lat': float(r['lat']),
            'lon': float(r['lon']),
            'node_type': str(r['node_type']),
            'area': str(r['area']),
            'zone': str(r['zone']) if r['zone'] is not None else None,
        }
        for _, r in result.iterrows()
    ]


if __name__ == '__main__':
    run_query()
