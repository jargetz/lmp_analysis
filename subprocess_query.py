"""
Subprocess-based MotherDuck query runner.
Runs queries in a separate process to avoid blocking Streamlit.
"""
import os
import sys
import json
import duckdb

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
        else:
            result = {'error': f'Unknown query type: {query_type}'}
        
        conn.close()
        print(json.dumps(result))
    except Exception as e:
        print(json.dumps({'error': str(e)}))

def get_node_bx(conn, bx, nodes, year):
    """Compute BX average for nodes"""
    import re
    nodes = [n for n in nodes if re.match(r'^[A-Za-z0-9_\-\.]+$', n)]
    if not nodes:
        return {'success': False, 'error': 'No valid nodes'}
    
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
        SELECT node, AVG(bx_price) as avg_price, MIN(bx_price) as min_price, 
               MAX(bx_price) as max_price, COUNT(DISTINCT opr_dt) as day_count
        FROM daily_bx GROUP BY node
    """
    
    result = conn.execute(query).fetchdf()
    if result.empty:
        return {'success': False, 'error': 'No data found'}
    
    per_node = {row['node']: float(row['avg_price']) for _, row in result.iterrows()}
    return {
        'success': True,
        'avg_price': float(result['avg_price'].mean()),
        'min_price': float(result['min_price'].min()),
        'max_price': float(result['max_price'].max()),
        'node_count': len(nodes),
        'day_count': int(result['day_count'].iloc[0]),
        'per_node': per_node
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
    
    query = f"""
        SELECT opr_hr as hour, AVG(mw) as avg_price
        FROM read_parquet('{path}', hive_partitioning=true)
        WHERE node IN ({node_list})
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
    
    query = f"""
        SELECT month, opr_hr as hour, AVG(mw) as avg_price
        FROM read_parquet('{path}', hive_partitioning=true)
        WHERE node IN ({node_list})
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

if __name__ == '__main__':
    run_query()
