"""
MotherDuck Client Module

Provides a unified interface for querying CAISO LMP data using MotherDuck (DuckDB cloud).
Replaces slow parquet file reads with fast SQL queries.

MotherDuck can query parquet files directly from S3 and also store summary tables.
"""

import os
import logging
import duckdb
import threading
from datetime import date
from typing import List, Dict, Any, Optional

class MotherDuckClient:
    """Client for querying CAISO LMP data via MotherDuck (thread-safe)"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._conn = None
        self._bucket = self._validate_bucket(os.getenv('AWS_S3_BUCKET'))
        self._s3_configured = False
        self._temp_tables = []
        self._lock = threading.Lock()
        self._thread_id = None
    
    def _validate_bucket(self, bucket: str) -> str:
        """Validate S3 bucket name to prevent injection"""
        import re
        if bucket and re.match(r'^[a-z0-9][a-z0-9\-\.]{1,61}[a-z0-9]$', bucket):
            return bucket
        self.logger.warning(f"Invalid bucket name: {bucket}")
        return None
    
    @property
    def conn(self):
        """Lazy-load MotherDuck connection (thread-safe)"""
        current_thread = threading.get_ident()
        
        # If connection exists but was created in different thread, reconnect
        if self._conn is not None and self._thread_id != current_thread:
            self.logger.info(f"Reconnecting - thread changed from {self._thread_id} to {current_thread}")
            self._conn = None
            self._s3_configured = False
        
        if self._conn is None:
            with self._lock:
                if self._conn is None:
                    self._connect()
                    self._thread_id = current_thread
        return self._conn
    
    def _connect(self):
        """Establish connection to MotherDuck"""
        token = os.getenv('MOTHERDUCK_TOKEN')
        if not token:
            raise ValueError("MOTHERDUCK_TOKEN environment variable not set")
        
        self._conn = duckdb.connect(f'md:?motherduck_token={token}')
        
        # Disable progress bar to prevent Streamlit interference
        self._conn.execute("SET enable_progress_bar = false")
        
        self._conn.execute("CREATE DATABASE IF NOT EXISTS caiso_lmp")
        self._conn.execute("USE caiso_lmp")
        
        self.logger.info("Connected to MotherDuck database: caiso_lmp")
        
        self._configure_s3()
    
    def _configure_s3(self):
        """Configure S3 credentials for parquet access"""
        if self._s3_configured:
            return
            
        aws_key = os.getenv('AWS_ACCESS_KEY_ID')
        aws_secret = os.getenv('AWS_SECRET_ACCESS_KEY')
        
        if aws_key and aws_secret:
            self.conn.execute(f"""
                CREATE OR REPLACE SECRET s3_secret (
                    TYPE S3,
                    KEY_ID '{aws_key}',
                    SECRET '{aws_secret}',
                    REGION 'us-west-2'
                )
            """)
            self._s3_configured = True
            self.logger.info("Configured S3 credentials for MotherDuck")
    
    def _get_parquet_path(self, year: int = None, month: int = None) -> str:
        """Get S3 parquet path with optional year/month filters"""
        base_path = f"s3://{self._bucket}/lmp_parquet"
        
        if year and month:
            return f"{base_path}/year={year}/month={month:02d}/*.parquet"
        elif year:
            return f"{base_path}/year={year}/**/*.parquet"
        else:
            return f"{base_path}/**/*.parquet"
    
    def execute_query(self, query: str, params: tuple = None) -> List[Dict]:
        """Execute a query and return results as list of dicts"""
        try:
            if params:
                result = self.conn.execute(query, params)
            else:
                result = self.conn.execute(query)
            
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            
            return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            self.logger.error(f"Query error: {e}")
            raise
    
    def get_full_year_hourly_data(
        self,
        nodes: List[str],
        year: int = 2024
    ) -> List[Dict]:
        """
        Get full year hourly data for selected nodes from parquet (FAST).
        
        This replaces the slow Python loop in bx_calculator.py that reads
        365 parquet files one by one. MotherDuck queries all files at once.
        
        Args:
            nodes: List of node names to include
            year: Year to query
            
        Returns:
            List of dicts with 'opr_dt', 'opr_hr', 'avg_price'
        """
        if not nodes:
            return []
        
        nodes = self._sanitize_node_list(nodes)
        if not nodes:
            return []
        
        parquet_path = self._get_parquet_path(year=year)
        node_list_sql = ', '.join(f"'{n}'" for n in nodes)
        
        query = f"""
            SELECT 
                DATE(regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1)) as opr_dt,
                opr_hr,
                AVG(mw) as avg_price
            FROM read_parquet('{parquet_path}', filename=true, hive_partitioning=true)
            WHERE node IN ({node_list_sql})
            GROUP BY 1, 2
            ORDER BY opr_dt, opr_hr
        """
        
        try:
            result = self.execute_query(query)
            return result
        except Exception as e:
            self.logger.error(f"Error getting full year hourly data: {e}")
            return []
    
    def _sanitize_node_list(self, nodes: List[str]) -> List[str]:
        """Sanitize node list to prevent injection attacks"""
        import re
        sanitized = []
        for node in nodes:
            if isinstance(node, str) and re.match(r'^[A-Za-z0-9_\-\.]+$', node):
                sanitized.append(node)
            else:
                self.logger.warning(f"Invalid node name filtered: {node}")
        return sanitized
    
    def _create_temp_node_table(self, nodes: List[str]) -> str:
        """Create a unique temp table with nodes for safe querying"""
        import uuid
        import pandas as pd
        
        table_name = f"tmp_nodes_{uuid.uuid4().hex[:8]}"
        self.conn.execute(f"CREATE OR REPLACE TEMP TABLE {table_name} (node VARCHAR)")
        nodes_df = pd.DataFrame({'node': nodes})
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM nodes_df")
        self._temp_tables.append(table_name)
        
        if len(self._temp_tables) > 20:
            self._cleanup_old_temp_tables()
        
        return table_name
    
    def _cleanup_old_temp_tables(self):
        """Clean up old temp tables to prevent memory bloat"""
        tables_to_drop = self._temp_tables[:-10]
        for table in tables_to_drop:
            try:
                self.conn.execute(f"DROP TABLE IF EXISTS {table}")
            except Exception:
                pass
        self._temp_tables = self._temp_tables[-10:]
    
    def _sanitize_zone(self, zone: str) -> Optional[str]:
        """Sanitize zone name to prevent injection"""
        valid_zones = ['NP15', 'SP15', 'ZP26', 'Overall']
        if zone in valid_zones:
            return zone
        self.logger.warning(f"Invalid zone name: {zone}")
        return None
    
    def get_node_bx_from_parquet(
        self,
        bx: int,
        nodes: List[str],
        year: int = 2024
    ) -> Dict[str, Any]:
        """
        Compute BX average for specific nodes from parquet files (FAST).
        
        This replaces the slow Python loop version in bx_calculator.py.
        Uses direct WHERE clause filtering instead of temp table joins for speed.
        
        Args:
            bx: BX value (4-10) - number of cheapest hours
            nodes: List of node names
            year: Year to query
            
        Returns:
            Dict with avg_price, per-node stats, and BX hours distribution
        """
        if not nodes:
            return {'success': False, 'error': 'No nodes specified'}
        
        nodes = self._sanitize_node_list(nodes)
        if not nodes:
            return {'success': False, 'error': 'No valid nodes specified'}
        
        bx = int(bx)
        if bx < 4 or bx > 10:
            return {'success': False, 'error': 'BX must be between 4 and 10'}
        
        parquet_path = self._get_parquet_path(year=year)
        
        # Build node list for IN clause (already sanitized)
        node_list_sql = ', '.join(f"'{n}'" for n in nodes)
        
        query = f"""
            WITH file_data AS (
                SELECT 
                    regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) as opr_dt,
                    node,
                    opr_hr,
                    mw
                FROM read_parquet('{parquet_path}', filename=true, hive_partitioning=true)
                WHERE node IN ({node_list_sql})
            ),
            ranked AS (
                SELECT 
                    opr_dt,
                    node,
                    opr_hr,
                    mw,
                    ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
                FROM file_data
            ),
            daily_bx AS (
                SELECT 
                    opr_dt,
                    node,
                    AVG(mw) as bx_price
                FROM ranked
                WHERE rn <= {bx}
                GROUP BY opr_dt, node
            )
            SELECT 
                AVG(bx_price) as avg_price,
                MIN(bx_price) as min_price,
                MAX(bx_price) as max_price,
                COUNT(DISTINCT opr_dt) as day_count,
                node
            FROM daily_bx
            GROUP BY node
        """
        
        try:
            results = self.execute_query(query)
            
            if not results:
                return {'success': False, 'error': 'No data found for selected nodes'}
            
            per_node = {r['node']: r['avg_price'] for r in results}
            all_prices = [r['avg_price'] for r in results]
            
            return {
                'success': True,
                'bx_type': bx,
                'avg_price': sum(all_prices) / len(all_prices),
                'min_price': min(r['min_price'] for r in results),
                'max_price': max(r['max_price'] for r in results),
                'node_count': len(nodes),
                'day_count': results[0]['day_count'] if results else 0,
                'per_node': per_node
            }
        except Exception as e:
            self.logger.error(f"Error computing BX from parquet: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_hourly_averages_for_nodes(
        self,
        nodes: List[str],
        year: int = 2024
    ) -> List[Dict]:
        """
        Get hourly price averages for nodes from parquet (FAST).
        
        Args:
            nodes: List of node names
            year: Year to query
            
        Returns:
            List of {'hour': int, 'avg_price': float}
        """
        if not nodes:
            return []
        
        nodes = self._sanitize_node_list(nodes)
        if not nodes:
            return []
        
        parquet_path = self._get_parquet_path(year=year)
        node_list_sql = ', '.join(f"'{n}'" for n in nodes)
        
        query = f"""
            SELECT 
                opr_hr as hour,
                AVG(mw) as avg_price
            FROM read_parquet('{parquet_path}', hive_partitioning=true)
            WHERE node IN ({node_list_sql})
            GROUP BY opr_hr
            ORDER BY opr_hr
        """
        
        try:
            return self.execute_query(query)
        except Exception as e:
            self.logger.error(f"Error getting hourly averages: {e}")
            return []
    
    def get_node_summary_statistics(
        self,
        bx: int,
        nodes: List[str],
        year: int = 2024
    ) -> List[Dict]:
        """
        Get summary statistics (for box plot) for each node using MotherDuck.
        
        Returns list of dicts with node, mean, min, max, q1, median, q3, day_count.
        """
        if not nodes:
            return []
        
        nodes = self._sanitize_node_list(nodes)
        if not nodes:
            return []
        
        bx = int(bx)
        if bx < 4 or bx > 10:
            return []
        
        parquet_path = self._get_parquet_path(year=year)
        node_list_sql = ', '.join(f"'{n}'" for n in nodes)
        
        query = f"""
            WITH file_data AS (
                SELECT 
                    regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) as opr_dt,
                    node,
                    opr_hr,
                    mw
                FROM read_parquet('{parquet_path}', filename=true, hive_partitioning=true)
                WHERE node IN ({node_list_sql})
            ),
            ranked AS (
                SELECT 
                    opr_dt,
                    node,
                    opr_hr,
                    mw,
                    ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
                FROM file_data
            ),
            daily_bx AS (
                SELECT 
                    opr_dt,
                    node,
                    AVG(mw) as daily_avg
                FROM ranked
                WHERE rn <= {bx}
                GROUP BY opr_dt, node
            )
            SELECT 
                node,
                AVG(daily_avg) as mean,
                MIN(daily_avg) as min,
                MAX(daily_avg) as max,
                PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY daily_avg) as q1,
                PERCENTILE_CONT(0.50) WITHIN GROUP (ORDER BY daily_avg) as median,
                PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY daily_avg) as q3,
                COUNT(*) as day_count
            FROM daily_bx
            GROUP BY node
            ORDER BY mean ASC
        """
        
        try:
            results = self.execute_query(query)
            return [
                {
                    'node': r['node'],
                    'mean': float(r['mean']),
                    'min': float(r['min']),
                    'max': float(r['max']),
                    'q1': float(r['q1']),
                    'median': float(r['median']),
                    'q3': float(r['q3']),
                    'day_count': int(r['day_count'])
                }
                for r in results
            ]
        except Exception as e:
            self.logger.error(f"Error getting node summary statistics: {e}")
            return []
    
    def get_hourly_averages_per_node(
        self,
        nodes: List[str],
        year: int = 2024
    ) -> Dict[str, List[Dict]]:
        """
        Get hourly price averages for each node individually using MotherDuck.
        
        Returns dict with node names as keys, each containing list of
        {'hour': int, 'avg_price': float} dicts.
        """
        if not nodes:
            return {}
        
        nodes = self._sanitize_node_list(nodes)
        if not nodes:
            return {}
        
        parquet_path = self._get_parquet_path(year=year)
        node_list_sql = ', '.join(f"'{n}'" for n in nodes)
        
        query = f"""
            SELECT 
                node,
                opr_hr as hour,
                AVG(mw) as avg_price
            FROM read_parquet('{parquet_path}', hive_partitioning=true)
            WHERE node IN ({node_list_sql})
            GROUP BY node, opr_hr
            ORDER BY node, opr_hr
        """
        
        try:
            results = self.execute_query(query)
            output = {}
            for r in results:
                node = r['node']
                if node not in output:
                    output[node] = []
                output[node].append({
                    'hour': int(r['hour']),
                    'avg_price': float(r['avg_price'])
                })
            return output
        except Exception as e:
            self.logger.error(f"Error getting per-node hourly averages: {e}")
            return {}
    
    def get_node_month_hour_averages(
        self,
        nodes: List[str],
        year: int = 2024
    ) -> List[Dict]:
        """
        Get month x hour heatmap data for selected nodes using MotherDuck.
        
        Returns list of {'month': int, 'hour': int, 'avg_price': float} dicts.
        """
        if not nodes:
            return []
        
        nodes = self._sanitize_node_list(nodes)
        if not nodes:
            return []
        
        parquet_path = self._get_parquet_path(year=year)
        temp_table = self._create_temp_node_table(nodes)
        
        query = f"""
            SELECT 
                EXTRACT(MONTH FROM DATE(regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1)))::INT as month,
                opr_hr as hour,
                AVG(mw) as avg_price
            FROM read_parquet('{parquet_path}', filename=true)
            WHERE node IN (SELECT node FROM {temp_table})
            GROUP BY 1, opr_hr
            ORDER BY month, hour
        """
        
        try:
            results = self.execute_query(query)
            return [
                {
                    'month': int(r['month']),
                    'hour': int(r['hour']),
                    'avg_price': float(r['avg_price'])
                }
                for r in results
            ]
        except Exception as e:
            self.logger.error(f"Error getting month-hour averages: {e}")
            return []
    
    def get_bx_hour_distribution(
        self,
        bx: int,
        nodes: List[str],
        year: int = 2024
    ) -> Dict[int, int]:
        """
        Get distribution of which hours are most commonly in the cheapest X.
        
        Args:
            bx: BX value (4-10)
            nodes: List of node names
            year: Year to query
            
        Returns:
            Dict mapping hour (1-24) to count of days it was in BX
        """
        if not nodes:
            return {}
        
        nodes = self._sanitize_node_list(nodes)
        if not nodes:
            return {}
        
        bx = int(bx)
        if bx < 4 or bx > 10:
            return {}
        
        parquet_path = self._get_parquet_path(year=year)
        temp_table = self._create_temp_node_table(nodes)
        
        query = f"""
            WITH ranked AS (
                SELECT 
                    DATE(regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1)) as opr_dt,
                    node,
                    opr_hr,
                    mw,
                    ROW_NUMBER() OVER (PARTITION BY filename, node ORDER BY mw ASC) as rn
                FROM read_parquet('{parquet_path}', filename=true)
                WHERE node IN (SELECT node FROM {temp_table})
            )
            SELECT 
                opr_hr as hour,
                COUNT(*) as count
            FROM ranked
            WHERE rn <= {bx}
            GROUP BY opr_hr
            ORDER BY count DESC
        """
        
        try:
            results = self.execute_query(query)
            return {r['hour']: r['count'] for r in results}
        except Exception as e:
            self.logger.error(f"Error getting BX hour distribution: {e}")
            return {}
    
    def get_available_years(self) -> List[int]:
        """Get list of years with parquet data available"""
        parquet_path = self._get_parquet_path()
        
        query = f"""
            SELECT DISTINCT 
                CAST(regexp_extract(filename, 'year=(\\d{{4}})', 1) AS INTEGER) as year
            FROM read_parquet('{parquet_path}', filename=true)
            WHERE year IS NOT NULL
            ORDER BY year DESC
        """
        
        try:
            results = self.execute_query(query)
            return [r['year'] for r in results if r['year']]
        except Exception as e:
            self.logger.error(f"Error getting available years: {e}")
            return [2024]
    
    def get_all_nodes(self, sample_year: int = 2024) -> List[str]:
        """Get all distinct node names from parquet"""
        parquet_path = self._get_parquet_path(year=sample_year)
        
        query = f"""
            SELECT DISTINCT node
            FROM read_parquet('{parquet_path}')
            ORDER BY node
        """
        
        try:
            results = self.execute_query(query)
            return [r['node'] for r in results]
        except Exception as e:
            self.logger.error(f"Error getting nodes: {e}")
            return []
    
    def test_connection(self) -> Dict[str, Any]:
        """Test MotherDuck connection and S3 access"""
        try:
            result = self.conn.execute("SELECT 1 as test").fetchone()
            
            parquet_path = f"s3://{self._bucket}/lmp_parquet/year=2024/month=01/2024-01-01.parquet"
            
            try:
                count_result = self.conn.execute(f"""
                    SELECT COUNT(*) as cnt FROM read_parquet('{parquet_path}')
                """).fetchone()
                s3_accessible = count_result[0] > 0 if count_result else False
                sample_count = count_result[0] if count_result else 0
            except Exception as s3_err:
                return {
                    'success': True,
                    'motherduck_connected': True,
                    's3_accessible': False,
                    's3_error': str(s3_err)
                }
            
            return {
                'success': True,
                'motherduck_connected': True,
                's3_accessible': s3_accessible,
                'sample_row_count': sample_count
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def close(self):
        """Close the connection"""
        if self._conn:
            self._conn.close()
            self._conn = None
    
    def create_summary_tables(self):
        """Create tables for storing migrated PostgreSQL summary data"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS bx_daily_summary (
                node VARCHAR,
                opr_dt DATE,
                bx_type INTEGER,
                avg_price DOUBLE,
                min_hour INTEGER,
                max_hour INTEGER
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS node_zone_mapping (
                pnode_id VARCHAR,
                zone VARCHAR
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS zone_hourly_lmp (
                zone VARCHAR,
                opr_dt DATE,
                hour_num INTEGER,
                lmp DOUBLE,
                congestion DOUBLE,
                energy DOUBLE,
                loss DOUBLE
            )
        """)
        
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS generator_bx_summary (
                zone VARCHAR,
                node VARCHAR,
                opr_dt DATE,
                bx_type INTEGER,
                avg_price DOUBLE
            )
        """)
        
        self.logger.info("Created summary tables in MotherDuck")
    
    def import_from_postgres(self, table_name: str, data: List[Dict]) -> int:
        """Import data from PostgreSQL into MotherDuck table"""
        if not data:
            return 0
        
        import pandas as pd
        from decimal import Decimal
        
        for row in data:
            for key, value in row.items():
                if isinstance(value, Decimal):
                    row[key] = float(value)
        
        df = pd.DataFrame(data)
        
        self.conn.execute(f"DELETE FROM {table_name}")
        
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")
        
        count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        self.logger.info(f"Imported {count} rows into {table_name}")
        return count
    
    def get_bx_average(
        self,
        bx: int,
        start_date: date = None,
        end_date: date = None,
        zone: str = None,
        nodes: List[str] = None
    ) -> Dict[str, Any]:
        """Get average BX price from MotherDuck summary tables"""
        bx = int(bx)
        if bx < 4 or bx > 10:
            return {'success': False, 'error': 'BX must be between 4 and 10'}
        
        conditions = ["s.bx_type = $1"]
        params = [bx]
        
        if start_date:
            conditions.append(f"s.opr_dt >= ${len(params)+1}")
            params.append(start_date)
        
        if end_date:
            conditions.append(f"s.opr_dt <= ${len(params)+1}")
            params.append(end_date)
        
        temp_table = None
        if nodes:
            nodes = self._sanitize_node_list(nodes)
            if nodes:
                temp_table = self._create_temp_node_table(nodes)
                conditions.append(f"s.node IN (SELECT node FROM {temp_table})")
        
        zone_join = ""
        if zone:
            zone = self._sanitize_zone(zone)
            if zone:
                zone_join = "JOIN node_zone_mapping m ON s.node = m.pnode_id"
                conditions.append(f"m.zone = ${len(params)+1}")
                params.append(zone)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                AVG(s.avg_price) as avg_bx_price,
                MIN(s.avg_price) as min_bx_price,
                MAX(s.avg_price) as max_bx_price,
                COUNT(DISTINCT s.node) as node_count,
                COUNT(DISTINCT s.opr_dt) as day_count
            FROM bx_daily_summary s
            {zone_join}
            WHERE {where_clause}
        """
        
        try:
            result = self.conn.execute(query, params).fetchone()
            return {
                'success': True,
                'bx_type': bx,
                'avg_price': float(result[0]) if result and result[0] else None,
                'min_price': float(result[1]) if result and result[1] else None,
                'max_price': float(result[2]) if result and result[2] else None,
                'node_count': result[3] if result else 0,
                'day_count': result[4] if result else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting B{bx} average: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_zone_hourly_averages(self, zone: str = None, year: int = None) -> List[Dict]:
        """Get hourly averages for zone from MotherDuck"""
        zone_name = self._sanitize_zone(zone) if zone else 'Overall'
        if zone and zone_name is None:
            zone_name = 'Overall'
        
        params = [zone_name]
        conditions = ["zone = $1"]
        
        if year:
            year = int(year)
            conditions.append("EXTRACT(YEAR FROM opr_dt) = $2")
            params.append(year)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                hour_num as hour,
                AVG(lmp) as avg_price
            FROM zone_hourly_lmp
            WHERE {where_clause}
            GROUP BY hour_num
            ORDER BY hour_num
        """
        
        try:
            result = self.conn.execute(query, params)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            results = [dict(zip(columns, row)) for row in rows]
            return [{'hour': int(r['hour']), 'avg_price': float(r['avg_price'])} for r in results]
        except Exception as e:
            self.logger.error(f"Error getting zone hourly averages: {e}")
            return []


_client = None
_client_lock = threading.Lock()

def get_motherduck_client(force_new: bool = False) -> MotherDuckClient:
    """Get MotherDuck client instance (thread-safe)
    
    Args:
        force_new: If True, create a fresh connection instead of using singleton
    """
    global _client
    if force_new:
        return MotherDuckClient()
    if _client is None:
        with _client_lock:
            if _client is None:
                _client = MotherDuckClient()
    return _client

def reset_motherduck_client():
    """Reset the singleton client (for connection issues)"""
    global _client
    with _client_lock:
        _client = None
