import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List
from database import DatabaseManager
import pandas as pd

class CAISOPreprocessor:
    """Handles pre-computation of analytics like B6 and B8 hours for CAISO LMP data"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
    def create_preprocessing_tables(self):
        """Create database tables for storing pre-computed analytics"""
        try:
            # Create B6 (cheapest 6 hours) table
            b6_table_sql = """
            CREATE TABLE IF NOT EXISTS caiso.b6_hours (
                id SERIAL PRIMARY KEY,
                node VARCHAR(100) NOT NULL,
                date_only DATE NOT NULL,
                opr_dt DATE NOT NULL,
                opr_hr SMALLINT NOT NULL,
                mw DECIMAL(10,2) NOT NULL,
                hour_rank INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(node, date_only, hour_rank)
            );
            
            CREATE INDEX IF NOT EXISTS idx_b6_node_date ON caiso.b6_hours(node, date_only);
            CREATE INDEX IF NOT EXISTS idx_b6_date ON caiso.b6_hours(date_only);
            """
            
            # Create B8 (cheapest 8 hours) table  
            b8_table_sql = """
            CREATE TABLE IF NOT EXISTS caiso.b8_hours (
                id SERIAL PRIMARY KEY,
                node VARCHAR(100) NOT NULL,
                date_only DATE NOT NULL,
                opr_dt DATE NOT NULL,
                opr_hr SMALLINT NOT NULL,
                mw DECIMAL(10,2) NOT NULL,
                hour_rank INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(node, date_only, hour_rank)
            );
            
            CREATE INDEX IF NOT EXISTS idx_b8_node_date ON caiso.b8_hours(node, date_only);
            CREATE INDEX IF NOT EXISTS idx_b8_date ON caiso.b8_hours(date_only);
            """
            
            # Create node daily summary table
            summary_table_sql = """
            CREATE TABLE IF NOT EXISTS caiso.node_daily_summary (
                id SERIAL PRIMARY KEY,
                node VARCHAR(100) NOT NULL,
                date_only DATE NOT NULL,
                min_price DECIMAL(10,2),
                max_price DECIMAL(10,2),
                avg_price DECIMAL(10,2),
                median_price DECIMAL(10,2),
                total_hours INTEGER,
                b6_avg_price DECIMAL(10,2),
                b8_avg_price DECIMAL(10,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(node, date_only)
            );
            
            CREATE INDEX IF NOT EXISTS idx_summary_node_date ON caiso.node_daily_summary(node, date_only);
            """
            
            # Execute all table creation commands
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(b6_table_sql)
                    cur.execute(b8_table_sql)
                    cur.execute(summary_table_sql)
                    conn.commit()
                    
            self.logger.info("Successfully created preprocessing tables")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating preprocessing tables: {str(e)}")
            return False
    
    def calculate_b6_b8_for_date_range(self, start_date: date, end_date: date, progress_callback=None) -> Dict[str, Any]:
        """Calculate B6 and B8 hours for a specific date range"""
        try:
            # Get list of dates to process
            current_date = start_date
            dates_to_process = []
            
            while current_date <= end_date:
                dates_to_process.append(current_date)
                current_date += timedelta(days=1)
            
            total_dates = len(dates_to_process)
            processed_dates = 0
            total_b6_records = 0
            total_b8_records = 0
            total_summary_records = 0
            
            # Process each date
            for i, process_date in enumerate(dates_to_process):
                if progress_callback:
                    progress_callback(i, total_dates, f"Processing B6/B8 for {process_date}")
                
                result = self.calculate_b6_b8_for_single_date(process_date)
                
                if result['success']:
                    processed_dates += 1
                    total_b6_records += result.get('b6_records', 0)
                    total_b8_records += result.get('b8_records', 0)
                    total_summary_records += result.get('summary_records', 0)
                else:
                    self.logger.warning(f"Failed to process date {process_date}: {result.get('error', 'Unknown error')}")
            
            return {
                'success': True,
                'processed_dates': processed_dates,
                'total_dates': total_dates,
                'total_b6_records': total_b6_records,
                'total_b8_records': total_b8_records,
                'total_summary_records': total_summary_records
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating B6/B8 for date range: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def calculate_b6_b8_for_single_date(self, target_date: date) -> Dict[str, Any]:
        """Calculate B6 and B8 hours for a single date"""
        try:
            # Check if already processed
            if self.is_date_already_processed(target_date):
                return {
                    'success': True,
                    'skipped': True,
                    'b6_records': 0,
                    'b8_records': 0,
                    'summary_records': 0
                }
            
            # Get all nodes that have data for this date
            nodes_query = """
            SELECT DISTINCT node 
            FROM caiso.lmp_data 
            WHERE date_only = %s
            ORDER BY node
            """
            nodes_result = self.db.execute_query(nodes_query, (target_date,))
            
            if not nodes_result:
                return {'success': False, 'error': f'No data found for date {target_date}'}
            
            nodes = [row['node'] for row in nodes_result if isinstance(row, dict)]
            
            b6_records = []
            b8_records = []
            summary_records = []
            
            # Process each node for this date
            for node in nodes:
                # Get all hours for this node and date, ordered by price
                hours_query = """
                SELECT opr_dt, opr_hr, mw
                FROM caiso.lmp_data 
                WHERE date_only = %s AND node = %s
                ORDER BY mw ASC, opr_dt ASC, opr_hr ASC
                """
                hours_result = self.db.execute_query(hours_query, (target_date, node))
                
                if not hours_result or len(hours_result) == 0:
                    continue
                
                # Calculate summary statistics
                prices = [float(row['mw']) for row in hours_result if isinstance(row, dict) and row.get('mw') is not None]
                summary_stats = {
                    'node': node,
                    'date_only': target_date,
                    'min_price': min(prices),
                    'max_price': max(prices),
                    'avg_price': sum(prices) / len(prices),
                    'median_price': sorted(prices)[len(prices) // 2],
                    'total_hours': len(prices)
                }
                
                # Get B6 (cheapest 6 hours)
                b6_hours = hours_result[:6] if isinstance(hours_result, list) else []
                b6_avg = sum(float(row['mw']) for row in b6_hours if isinstance(row, dict)) / len(b6_hours) if b6_hours else 0
                summary_stats['b6_avg_price'] = b6_avg
                
                for rank, hour_data in enumerate(b6_hours, 1):
                    b6_records.append({
                        'node': node,
                        'date_only': target_date,
                        'opr_dt': hour_data['opr_dt'],
                        'opr_hr': hour_data['opr_hr'],
                        'mw': hour_data['mw'],
                        'hour_rank': rank
                    })
                
                # Get B8 (cheapest 8 hours)
                b8_hours = hours_result[:8] if isinstance(hours_result, list) else []
                b8_avg = sum(float(row['mw']) for row in b8_hours if isinstance(row, dict)) / len(b8_hours) if b8_hours else 0
                summary_stats['b8_avg_price'] = b8_avg
                
                for rank, hour_data in enumerate(b8_hours, 1):
                    b8_records.append({
                        'node': node,
                        'date_only': target_date,
                        'opr_dt': hour_data['opr_dt'],
                        'opr_hr': hour_data['opr_hr'],
                        'mw': hour_data['mw'],
                        'hour_rank': rank
                    })
                
                summary_records.append(summary_stats)
            
            # Bulk insert all records
            b6_inserted = self.bulk_insert_b6_records(b6_records) if b6_records else 0
            b8_inserted = self.bulk_insert_b8_records(b8_records) if b8_records else 0
            summary_inserted = self.bulk_insert_summary_records(summary_records) if summary_records else 0
            
            return {
                'success': True,
                'b6_records': b6_inserted,
                'b8_records': b8_inserted,
                'summary_records': summary_inserted,
                'nodes_processed': len(nodes)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating B6/B8 for date {target_date}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def is_date_already_processed(self, target_date: date) -> bool:
        """Check if B6/B8 calculations already exist for this date"""
        try:
            query = "SELECT COUNT(*) as count FROM caiso.b6_hours WHERE date_only = %s LIMIT 1"
            result = self.db.execute_query(query, (target_date,), fetch_all=False)
            return result and isinstance(result, dict) and result.get('count', 0) > 0
        except:
            return False
    
    def bulk_insert_b6_records(self, records: List[Dict]) -> int:
        """Bulk insert B6 hours records"""
        if not records:
            return 0
        
        try:
            query = """
            INSERT INTO caiso.b6_hours (node, date_only, opr_dt, opr_hr, mw, hour_rank)
            VALUES (%(node)s, %(date_only)s, %(opr_dt)s, %(opr_hr)s, %(mw)s, %(hour_rank)s)
            ON CONFLICT (node, date_only, hour_rank) DO NOTHING
            """
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    from psycopg2.extras import execute_batch
                    execute_batch(cur, query, records, page_size=1000)
                    conn.commit()
                    
            return len(records)
            
        except Exception as e:
            self.logger.error(f"Error inserting B6 records: {str(e)}")
            return 0
    
    def bulk_insert_b8_records(self, records: List[Dict]) -> int:
        """Bulk insert B8 hours records"""
        if not records:
            return 0
            
        try:
            query = """
            INSERT INTO caiso.b8_hours (node, date_only, opr_dt, opr_hr, mw, hour_rank)
            VALUES (%(node)s, %(date_only)s, %(opr_dt)s, %(opr_hr)s, %(mw)s, %(hour_rank)s)
            ON CONFLICT (node, date_only, hour_rank) DO NOTHING
            """
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    from psycopg2.extras import execute_batch
                    execute_batch(cur, query, records, page_size=1000)
                    conn.commit()
                    
            return len(records)
            
        except Exception as e:
            self.logger.error(f"Error inserting B8 records: {str(e)}")
            return 0
    
    def bulk_insert_summary_records(self, records: List[Dict]) -> int:
        """Bulk insert node daily summary records"""
        if not records:
            return 0
            
        try:
            query = """
            INSERT INTO caiso.node_daily_summary 
            (node, date_only, min_price, max_price, avg_price, median_price, total_hours, b6_avg_price, b8_avg_price)
            VALUES (%(node)s, %(date_only)s, %(min_price)s, %(max_price)s, %(avg_price)s, 
                    %(median_price)s, %(total_hours)s, %(b6_avg_price)s, %(b8_avg_price)s)
            ON CONFLICT (node, date_only) DO UPDATE SET
                min_price = EXCLUDED.min_price,
                max_price = EXCLUDED.max_price,
                avg_price = EXCLUDED.avg_price,
                median_price = EXCLUDED.median_price,
                total_hours = EXCLUDED.total_hours,
                b6_avg_price = EXCLUDED.b6_avg_price,
                b8_avg_price = EXCLUDED.b8_avg_price,
                created_at = CURRENT_TIMESTAMP
            """
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    from psycopg2.extras import execute_batch
                    execute_batch(cur, query, records, page_size=1000)
                    conn.commit()
                    
            return len(records)
            
        except Exception as e:
            self.logger.error(f"Error inserting summary records: {str(e)}")
            return 0
    
    def get_date_range_from_data(self) -> Dict[str, Any]:
        """Get the date range of available LMP data"""
        try:
            query = """
            SELECT 
                MIN(date_only) as earliest_date,
                MAX(date_only) as latest_date,
                COUNT(DISTINCT date_only) as total_days
            FROM caiso.lmp_data
            """
            
            result = self.db.execute_query(query, fetch_all=False)
            return result if result and isinstance(result, dict) else {}
            
        except Exception as e:
            self.logger.error(f"Error getting date range: {str(e)}")
            return {}
    
    def run_full_preprocessing(self, progress_callback=None) -> Dict[str, Any]:
        """Run complete preprocessing on all available data"""
        try:
            # Create tables if they don't exist
            if not self.create_preprocessing_tables():
                return {'success': False, 'error': 'Failed to create preprocessing tables'}
            
            # Get date range of available data
            date_range = self.get_date_range_from_data()
            if not date_range or not date_range.get('earliest_date'):
                return {'success': False, 'error': 'No LMP data found for preprocessing'}
            
            start_date = date_range['earliest_date']
            end_date = date_range['latest_date']
            
            if progress_callback:
                progress_callback(0, 100, f"Starting B6/B8 preprocessing for {start_date} to {end_date}")
            
            # Calculate B6/B8 for the entire date range
            result = self.calculate_b6_b8_for_date_range(start_date, end_date, progress_callback)
            
            if result['success']:
                self.logger.info(f"Preprocessing completed: {result}")
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in full preprocessing: {str(e)}")
            return {'success': False, 'error': str(e)}