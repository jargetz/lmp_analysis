import os
import pandas as pd
import psycopg2
from psycopg2.extras import RealDictCursor, execute_batch
from sqlalchemy import create_engine
import logging
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Union, overload

class DatabaseManager:
    """Database connection and query management for CAISO LMP data"""
    
    def __init__(self):
        self.database_url = os.getenv('DATABASE_URL')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable not set")
        
        self.engine = create_engine(self.database_url)
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging for database operations"""
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_connection(self):
        """Get a direct psycopg2 connection"""
        return psycopg2.connect(self.database_url)
    
    def execute_query(self, query: str, params=None, fetch_all: bool = True) -> Union[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Execute a query and return results as regular Python dictionaries"""
        try:
            with self.get_connection() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(query, params)
                    if fetch_all and cur.description:
                        results = cur.fetchall()
                        return [dict(row) for row in results] if results else []
                    elif cur.description:
                        result = cur.fetchone()
                        return dict(result) if result else None
                    return None
        except Exception as e:
            self.logger.error(f"Error executing query: {str(e)}")
            raise
    
    def execute_many(self, query, data_list):
        """Execute a query with multiple parameter sets"""
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    execute_batch(cur, query, data_list, page_size=1000)
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Error executing batch query: {str(e)}")
            raise
    
    def bulk_insert_lmp_data(self, df):
        """Bulk insert LMP data using pandas to_sql for efficiency"""
        try:
            # Prepare the dataframe
            df_clean = df.copy()
            
            # CRITICAL FIX: Remove duplicate columns that cause to_sql errors
            df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
            
            # Rename columns to match database schema
            column_mapping = {
                'NODE': 'node',
                'MW': 'mw',
                'MCC': 'mcc',
                'MLC': 'mlc', 
                'POS': 'pos',
                'HOUR': 'hour_of_day',
                'DAY_OF_WEEK': 'day_of_week',
                'DATE': 'date_only'
            }
            
            # Only keep columns that exist in the dataframe and rename them
            existing_columns = {k: v for k, v in column_mapping.items() if k in df_clean.columns}
            df_clean = df_clean.rename(columns=existing_columns)
            
            # Convert date column to proper format
            if 'date_only' in df_clean.columns:
                df_clean['date_only'] = pd.to_datetime(df_clean['date_only']).dt.date
            
            # Define expected columns for the database table
            expected_columns = [
                'node', 'mw', 'mcc', 'mlc', 'pos',
                'hour_of_day', 'day_of_week', 'date_only', 'source_file', 'opr_hr', 'opr_dt'
            ]
            
            # Keep only columns that should be in the database
            df_final = df_clean[[col for col in expected_columns if col in df_clean.columns]]
            
            # Use to_sql for efficient bulk insert
            df_final.to_sql(
                'lmp_data', 
                self.engine, 
                schema='caiso',
                if_exists='append', 
                index=False,
                method='multi'
            )
            
            return len(df_final)
            
        except Exception as e:
            self.logger.error(f"Error bulk inserting LMP data: {str(e)}")
            raise
    
    def get_data_summary(self) -> Dict[str, Any]:
        """Get summary statistics of stored data"""
        query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(DISTINCT node) as unique_nodes,
            MIN(opr_dt) as earliest_date,
            MAX(opr_dt) as latest_date,
            AVG(mw) as avg_price,
            MIN(mw) as min_price,
            MAX(mw) as max_price
        FROM caiso.lmp_data
        """
        
        result = self.execute_query(query, fetch_all=False)
        return result if result and isinstance(result, dict) else {}
    
    def get_unique_nodes(self) -> List[str]:
        """Get list of unique nodes in the database"""
        query = "SELECT DISTINCT node FROM caiso.lmp_data ORDER BY node"
        results = self.execute_query(query)
        return [row['node'] for row in results] if results and isinstance(results, list) else []
    
    def get_cheapest_hours(self, n_hours=10, node=None, start_date=None, end_date=None):
        """Get N cheapest hours with optional filters"""
        conditions = []
        params = []
        
        if node:
            conditions.append("node = %s")
            params.append(node)
        
        if start_date:
            conditions.append("opr_dt >= %s")
            params.append(start_date)
            
        if end_date:
            conditions.append("opr_dt <= %s")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT 
            opr_dt as operational_date,
            opr_hr as operational_hour,
            node,
            mw,
            mcc,
            mlc,
            hour_of_day
        FROM caiso.lmp_data 
        {where_clause}
        ORDER BY mw ASC
        LIMIT %s
        """
        
        params.append(n_hours)
        results = self.execute_query(query, params)
        
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def get_node_price_statistics(self, node=None, start_date=None, end_date=None):
        """Get price statistics for nodes"""
        conditions = []
        params = []
        
        if node:
            conditions.append("node = %s")
            params.append(node)
            
        if start_date:
            conditions.append("opr_dt >= %s")
            params.append(start_date)
            
        if end_date:
            conditions.append("opr_dt <= %s")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT 
            node,
            COUNT(*) as record_count,
            AVG(mw) as avg_price,
            MIN(mw) as min_price,
            MAX(mw) as max_price,
            STDDEV(mw) as std_price,
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY mw) as p25,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mw) as median,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY mw) as p75,
            PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY mw) as p95
        FROM caiso.lmp_data 
        {where_clause}
        GROUP BY node
        ORDER BY avg_price
        """
        
        results = self.execute_query(query, params)
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def get_hourly_averages(self, start_date=None, end_date=None):
        """Get average prices by hour of day"""
        conditions = []
        params = []
        
        if start_date:
            conditions.append("opr_dt >= %s")
            params.append(start_date)
            
        if end_date:
            conditions.append("opr_dt <= %s")
            params.append(end_date)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
        SELECT 
            hour_of_day,
            AVG(mw) as avg_price,
            STDDEV(mw) as std_price,
            COUNT(*) as record_count
        FROM caiso.lmp_data 
        {where_clause}
        GROUP BY hour_of_day
        ORDER BY hour_of_day
        """
        
        results = self.execute_query(query, params)
        return pd.DataFrame(results) if results else pd.DataFrame()
    
    def update_cheapest_hours_cache(self, date_period=None):
        """Update pre-computed cheapest hours tables"""
        if not date_period:
            date_period = date.today()
        
        try:
            # Update B6 table
            query_b6 = """
            WITH ranked_hours AS (
                SELECT 
                    node,
                    opr_dt,
                    opr_hr,
                    mw,
                    ROW_NUMBER() OVER (PARTITION BY node ORDER BY mw ASC) as rn
                FROM caiso.lmp_data 
                WHERE date_only = %s
            )
            INSERT INTO caiso.cheapest_hours_b6 (node, opr_dt, opr_hr, mw, rank_position, date_computed)
            SELECT node, opr_dt, opr_hr, mw, rn, %s
            FROM ranked_hours 
            WHERE rn <= 6
            ON CONFLICT (node, date_computed, rank_position) DO UPDATE SET
                opr_dt = EXCLUDED.opr_dt,
                opr_hr = EXCLUDED.opr_hr,
                mw = EXCLUDED.mw,
                created_at = CURRENT_TIMESTAMP
            """
            
            # Update B8 table  
            query_b8 = """
            WITH ranked_hours AS (
                SELECT 
                    node,
                    opr_dt,
                    opr_hr,
                    mw,
                    ROW_NUMBER() OVER (PARTITION BY node ORDER BY mw ASC) as rn
                FROM caiso.lmp_data 
                WHERE date_only = %s
            )
            INSERT INTO caiso.cheapest_hours_b8 (node, opr_dt, opr_hr, mw, rank_position, date_computed)
            SELECT node, opr_dt, opr_hr, mw, rn, %s
            FROM ranked_hours 
            WHERE rn <= 8
            ON CONFLICT (node, date_computed, rank_position) DO UPDATE SET
                opr_dt = EXCLUDED.opr_dt,
                opr_hr = EXCLUDED.opr_hr,
                mw = EXCLUDED.mw,
                created_at = CURRENT_TIMESTAMP
            """
            
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query_b6, (date_period, date_period))
                    cur.execute(query_b8, (date_period, date_period))
                    conn.commit()
            
            self.logger.info(f"Updated cheapest hours cache for {date_period}")
            
        except Exception as e:
            self.logger.error(f"Error updating cheapest hours cache: {str(e)}")
            raise
    
    def clear_data(self, start_date=None, end_date=None):
        """Clear data for a specific date range (use with caution)"""
        conditions = []
        params = []
        
        if start_date:
            conditions.append("opr_dt >= %s")
            params.append(start_date)
            
        if end_date:
            conditions.append("opr_dt <= %s")
            params.append(end_date)
        
        if not conditions:
            raise ValueError("Must provide at least start_date or end_date for safety")
        
        where_clause = "WHERE " + " AND ".join(conditions)
        
        query = f"DELETE FROM caiso.lmp_data {where_clause}"
        
        try:
            with self.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    deleted_count = cur.rowcount
                    conn.commit()
            
            self.logger.info(f"Deleted {deleted_count} records")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error clearing data: {str(e)}")
            raise