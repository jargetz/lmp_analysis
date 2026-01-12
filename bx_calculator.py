"""
BX Hours Calculator Module

Calculates the cheapest X hours per day (BX) for CAISO LMP data.
Supports B4 through B10 with a unified, parameterized approach.

The BX value represents the average price of the X cheapest hours
in a given day for each node. This is commonly used for evaluating
battery storage charging strategies.

Usage:
    calculator = BXCalculator()
    
    # Calculate B6 for a specific date
    result = calculator.calculate_bx_for_date(date(2024, 1, 15), bx=6)
    
    # Calculate all BX values (4-10) for a date range
    result = calculator.calculate_all_bx_for_range(start_date, end_date)
"""

import logging
import pandas as pd
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
from database import DatabaseManager
from parquet_storage import ParquetStorage

# Supported BX values
SUPPORTED_BX_VALUES = [4, 5, 6, 7, 8, 9, 10]

# Minimum/maximum supported values
MIN_BX = 4
MAX_BX = 10


class BXCalculator:
    """Calculates cheapest X hours (BX) for CAISO LMP data"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.logger = logging.getLogger(__name__)
        self._parquet = None
    
    @property
    def parquet(self):
        """Lazy-load parquet storage"""
        if self._parquet is None:
            self._parquet = ParquetStorage()
        return self._parquet
    
    def create_bx_table(self) -> bool:
        """
        Create unified BX hours table.
        
        Uses a single table with bx_type column instead of separate tables
        for each BX value. This is more flexible and easier to extend.
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS caiso.bx_hours (
            id SERIAL PRIMARY KEY,
            node VARCHAR(100) NOT NULL,
            opr_dt DATE NOT NULL,
            opr_hr SMALLINT NOT NULL,
            mw DECIMAL(10,2) NOT NULL,
            hour_rank INTEGER NOT NULL,
            bx_type SMALLINT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(node, opr_dt, hour_rank, bx_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_bx_node_date ON caiso.bx_hours(node, opr_dt);
        CREATE INDEX IF NOT EXISTS idx_bx_date ON caiso.bx_hours(opr_dt);
        CREATE INDEX IF NOT EXISTS idx_bx_type ON caiso.bx_hours(bx_type);
        CREATE INDEX IF NOT EXISTS idx_bx_lookup ON caiso.bx_hours(bx_type, opr_dt, node);
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_sql)
                    conn.commit()
            self.logger.info("Created bx_hours table")
            return True
        except Exception as e:
            self.logger.error(f"Error creating bx_hours table: {str(e)}")
            return False
    
    def store_daily_bx_batch(self, opr_date: date, node_bx_prices: Dict[str, float], bx_type: int) -> int:
        """Store pre-computed BX averages for multiple nodes in a single batch.
        
        Args:
            opr_date: The operating date
            node_bx_prices: Dict mapping node name to BX average price
            bx_type: The BX type (4-10)
            
        Returns:
            Number of records inserted
        """
        if not node_bx_prices:
            return 0
        
        try:
            import io
            import csv
            
            output = io.StringIO()
            writer = csv.writer(output)
            for node, avg_price in node_bx_prices.items():
                writer.writerow([node, opr_date.isoformat(), bx_type, round(avg_price, 4)])
            
            output.seek(0)
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        CREATE TEMP TABLE tmp_bx_import (
                            node VARCHAR(100),
                            opr_dt DATE,
                            bx_type INTEGER,
                            avg_price NUMERIC(12,4)
                        ) ON COMMIT DROP
                    """)
                    
                    cur.copy_expert(
                        "COPY tmp_bx_import (node, opr_dt, bx_type, avg_price) FROM STDIN WITH CSV",
                        output
                    )
                    
                    cur.execute("""
                        INSERT INTO caiso.bx_daily_summary (node, opr_dt, bx_type, avg_price)
                        SELECT node, opr_dt, bx_type, avg_price FROM tmp_bx_import
                        ON CONFLICT (opr_dt, node, bx_type) DO UPDATE SET
                            avg_price = EXCLUDED.avg_price
                    """)
                    
                    inserted = cur.rowcount
                    conn.commit()
                    
            return inserted
            
        except Exception as e:
            self.logger.error(f"Error storing BX batch for {opr_date}: {str(e)}")
            return 0

    def create_bx_summary_table(self) -> bool:
        """
        Create BX summary table with average prices for each BX type.
        
        This table stores pre-computed averages for quick dashboard queries.
        """
        create_sql = """
        CREATE TABLE IF NOT EXISTS caiso.bx_daily_summary (
            id SERIAL PRIMARY KEY,
            node VARCHAR(100) NOT NULL,
            opr_dt DATE NOT NULL,
            bx_type SMALLINT NOT NULL,
            avg_price DECIMAL(10,2) NOT NULL,
            min_hour SMALLINT,
            max_hour SMALLINT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(node, opr_dt, bx_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_bx_summary_lookup 
            ON caiso.bx_daily_summary(bx_type, opr_dt, node);
        CREATE INDEX IF NOT EXISTS idx_bx_summary_date ON caiso.bx_daily_summary(opr_dt);
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_sql)
                    conn.commit()
            self.logger.info("Created bx_daily_summary table")
            return True
        except Exception as e:
            self.logger.error(f"Error creating bx_daily_summary table: {str(e)}")
            return False
    
    def create_monthly_summary_table(self) -> bool:
        """Create monthly BX summary table for fast dashboard queries."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS caiso.bx_monthly_summary (
            id SERIAL PRIMARY KEY,
            node VARCHAR(100) NOT NULL,
            year_month VARCHAR(7) NOT NULL,
            bx_type SMALLINT NOT NULL,
            avg_price DECIMAL(10,2) NOT NULL,
            min_price DECIMAL(10,2),
            max_price DECIMAL(10,2),
            day_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(node, year_month, bx_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_bx_monthly_lookup 
            ON caiso.bx_monthly_summary(bx_type, year_month);
        CREATE INDEX IF NOT EXISTS idx_bx_monthly_node 
            ON caiso.bx_monthly_summary(node, bx_type);
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_sql)
                    conn.commit()
            self.logger.info("Created bx_monthly_summary table")
            return True
        except Exception as e:
            self.logger.error(f"Error creating bx_monthly_summary table: {str(e)}")
            return False
    
    def create_annual_summary_table(self) -> bool:
        """Create annual BX summary table for fast dashboard defaults."""
        create_sql = """
        CREATE TABLE IF NOT EXISTS caiso.bx_annual_summary (
            id SERIAL PRIMARY KEY,
            node VARCHAR(100) NOT NULL,
            year INTEGER NOT NULL,
            bx_type SMALLINT NOT NULL,
            avg_price DECIMAL(10,2) NOT NULL,
            min_price DECIMAL(10,2),
            max_price DECIMAL(10,2),
            day_count INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(node, year, bx_type)
        );
        
        CREATE INDEX IF NOT EXISTS idx_bx_annual_lookup 
            ON caiso.bx_annual_summary(bx_type, year);
        CREATE INDEX IF NOT EXISTS idx_bx_annual_node 
            ON caiso.bx_annual_summary(node, bx_type);
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(create_sql)
                    conn.commit()
            self.logger.info("Created bx_annual_summary table")
            return True
        except Exception as e:
            self.logger.error(f"Error creating bx_annual_summary table: {str(e)}")
            return False
    
    def calculate_all_bx_for_date(
        self, 
        target_date: date,
        bx_values: List[int] = None,
        force_recalculate: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate multiple BX values for a single date efficiently.
        
        This is the primary calculation method. It fetches hourly data once
        per node and derives all BX values from that single query, avoiding
        redundant database scans.
        
        Args:
            target_date: The date to process
            bx_values: List of BX values to calculate (default: all 4-10)
            force_recalculate: If True, delete existing data and recalculate
            
        Returns:
            Dict with results for each BX type
        """
        bx_values = bx_values or SUPPORTED_BX_VALUES
        max_bx = max(bx_values)
        
        # Validate all BX values
        for bx in bx_values:
            if bx < MIN_BX or bx > MAX_BX:
                return {
                    'success': False,
                    'error': f'BX must be between {MIN_BX} and {MAX_BX}, got {bx}'
                }
        
        try:
            # If force recalculate, delete existing data for this date and these BX types
            if force_recalculate:
                self._delete_bx_data_for_date(target_date, bx_values)
            else:
                # Filter to only BX values that aren't already calculated
                bx_values = [bx for bx in bx_values if not self._is_bx_calculated(target_date, bx)]
                if not bx_values:
                    return {
                        'success': True,
                        'skipped': True,
                        'date': target_date,
                        'message': 'All BX values already calculated'
                    }
                max_bx = max(bx_values)
            
            # Fetch ALL hourly data for this date in one query
            # Order by node, then price (cheapest first)
            all_hours_query = """
                SELECT node, opr_dt, opr_hr, mw
                FROM caiso.lmp_data 
                WHERE opr_dt = %s
                ORDER BY node, mw ASC
            """
            all_hours = self.db.execute_query(all_hours_query, (target_date,))
            
            if not all_hours:
                return {'success': False, 'error': f'No data found for {target_date}'}
            
            # Group hours by node (already sorted by price within each node)
            node_hours: Dict[str, List[Dict]] = {}
            for row in all_hours:
                node = row['node']
                if node not in node_hours:
                    node_hours[node] = []
                node_hours[node].append(row)
            
            # Build summary records only (skip detailed hour records for performance)
            summary_records = []
            nodes_processed = 0
            
            for node, hours in node_hours.items():
                # Skip if not enough hours for the smallest BX we need
                if len(hours) < min(bx_values):
                    continue
                
                nodes_processed += 1
                
                # For each BX type, extract the cheapest X hours and compute summary
                for bx in bx_values:
                    if len(hours) < bx:
                        continue
                    
                    cheapest_hours = hours[:bx]
                    prices = [float(h['mw']) for h in cheapest_hours]
                    hours_used = [h['opr_hr'] for h in cheapest_hours]
                    
                    # Build summary record only
                    summary_records.append({
                        'node': node,
                        'opr_dt': target_date,
                        'bx_type': bx,
                        'avg_price': sum(prices) / len(prices),
                        'min_hour': min(hours_used),
                        'max_hour': max(hours_used)
                    })
            
            # Bulk insert summary records only
            summary_inserted = self._insert_bx_summary(summary_records) if summary_records else 0
            
            return {
                'success': True,
                'date': target_date,
                'bx_values': bx_values,
                'nodes_processed': nodes_processed,
                'summaries_inserted': summary_inserted
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating BX for {target_date}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def _delete_bx_data_for_date(self, target_date: date, bx_values: List[int]) -> None:
        """Delete existing BX data for a date to allow recalculation"""
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    # Delete from bx_daily_summary only (we no longer use bx_hours)
                    cur.execute(
                        "DELETE FROM caiso.bx_daily_summary WHERE opr_dt = %s AND bx_type = ANY(%s)",
                        (target_date, bx_values)
                    )
                    conn.commit()
        except Exception as e:
            self.logger.error(f"Error deleting BX data for {target_date}: {str(e)}")
    
    def calculate_bx_for_date_range(
        self,
        start_date: date,
        end_date: date,
        bx_values: List[int] = None,
        progress_callback=None
    ) -> Dict[str, Any]:
        """
        Calculate BX values for a date range.
        
        Args:
            start_date: Start of date range
            end_date: End of date range
            bx_values: List of BX values to calculate (default: all 4-10)
            progress_callback: Optional callback(current, total, message)
            
        Returns:
            Summary of all calculations
        """
        bx_values = bx_values or SUPPORTED_BX_VALUES
        
        # Ensure summary table exists (we only use summaries now)
        self.create_bx_summary_table()
        
        # Build list of dates
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        
        total_dates = len(dates)
        processed = 0
        total_summaries = 0
        
        for i, d in enumerate(dates):
            if progress_callback:
                progress_callback(i, total_dates, f"Processing {d} for B{min(bx_values)}-B{max(bx_values)}")
            
            result = self.calculate_all_bx_for_date(d, bx_values)
            
            if result.get('success'):
                processed += 1
                total_summaries += result.get('summaries_inserted', 0)
        
        return {
            'success': True,
            'dates_processed': processed,
            'total_dates': total_dates,
            'total_summaries': total_summaries,
            'bx_values': bx_values
        }
    
    def calculate_bx_for_date(
        self, 
        target_date: date, 
        bx: int,
        force_recalculate: bool = False
    ) -> Dict[str, Any]:
        """
        Calculate a single BX value for a date.
        
        Convenience wrapper around calculate_all_bx_for_date for when
        you only need one BX type.
        
        Args:
            target_date: The date to calculate
            bx: The BX value (4-10)
            force_recalculate: If True, recalculate even if exists
            
        Returns:
            Dict with success status and record counts
        """
        return self.calculate_all_bx_for_date(target_date, [bx], force_recalculate)
    
    def _is_bx_calculated(self, target_date: date, bx: int) -> bool:
        """Check if BX calculation exists for this date and type"""
        try:
            query = """
                SELECT COUNT(*) as count 
                FROM caiso.bx_daily_summary 
                WHERE opr_dt = %s AND bx_type = %s 
                LIMIT 1
            """
            result = self.db.execute_query(query, (target_date, bx), fetch_all=False)
            return result and result.get('count', 0) > 0
        except Exception:
            return False
    
    def _insert_bx_hours(self, records: List[Dict]) -> int:
        """Insert BX hour records"""
        if not records:
            return 0
        
        try:
            query = """
                INSERT INTO caiso.bx_hours 
                (node, opr_dt, opr_hr, mw, hour_rank, bx_type)
                VALUES (%(node)s, %(opr_dt)s, %(opr_hr)s, %(mw)s, %(hour_rank)s, %(bx_type)s)
                ON CONFLICT (node, opr_dt, hour_rank, bx_type) DO NOTHING
            """
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    from psycopg2.extras import execute_batch
                    execute_batch(cur, query, records, page_size=1000)
                    conn.commit()
            
            return len(records)
        except Exception as e:
            self.logger.error(f"Error inserting BX hours: {str(e)}")
            return 0
    
    def _insert_bx_summary(self, records: List[Dict]) -> int:
        """Insert BX summary records"""
        if not records:
            return 0
        
        try:
            query = """
                INSERT INTO caiso.bx_daily_summary 
                (node, opr_dt, bx_type, avg_price, min_hour, max_hour)
                VALUES (%(node)s, %(opr_dt)s, %(bx_type)s, %(avg_price)s, %(min_hour)s, %(max_hour)s)
                ON CONFLICT (node, opr_dt, bx_type) DO UPDATE SET
                    avg_price = EXCLUDED.avg_price,
                    min_hour = EXCLUDED.min_hour,
                    max_hour = EXCLUDED.max_hour,
                    created_at = CURRENT_TIMESTAMP
            """
            
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    from psycopg2.extras import execute_batch
                    execute_batch(cur, query, records, page_size=1000)
                    conn.commit()
            
            return len(records)
        except Exception as e:
            self.logger.error(f"Error inserting BX summary: {str(e)}")
            return 0
    
    def get_bx_average(
        self,
        bx: int,
        start_date: date = None,
        end_date: date = None,
        zone: str = None,
        nodes: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get average BX price with optional filters.
        
        Args:
            bx: BX type (4-10)
            start_date: Optional start date filter
            end_date: Optional end date filter
            zone: Optional zone filter (requires node_zone_mapping table)
            nodes: Optional list of node names to filter by
            
        Returns:
            Dict with average price and supporting stats
        """
        conditions = ["bx_type = %s"]
        params = [bx]
        
        if start_date:
            conditions.append("s.opr_dt >= %s")
            params.append(start_date)
        
        if end_date:
            conditions.append("s.opr_dt <= %s")
            params.append(end_date)
        
        if nodes:
            if isinstance(nodes, str):
                nodes = [nodes]
            placeholders = ','.join(['%s'] * len(nodes))
            conditions.append(f"s.node IN ({placeholders})")
            params.extend(nodes)
        
        # Build zone join if needed
        zone_join = ""
        if zone:
            zone_join = "JOIN caiso.node_zone_mapping m ON s.node = m.pnode_id"
            conditions.append("m.zone = %s")
            params.append(zone)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                AVG(s.avg_price) as avg_bx_price,
                MIN(s.avg_price) as min_bx_price,
                MAX(s.avg_price) as max_bx_price,
                COUNT(DISTINCT s.node) as node_count,
                COUNT(DISTINCT s.opr_dt) as day_count
            FROM caiso.bx_daily_summary s
            {zone_join}
            WHERE {where_clause}
        """
        
        try:
            result = self.db.execute_query(query, params, fetch_all=False)
            return {
                'success': True,
                'bx_type': bx,
                'avg_price': float(result['avg_bx_price']) if result and result.get('avg_bx_price') else None,
                'min_price': float(result['min_bx_price']) if result and result.get('min_bx_price') else None,
                'max_price': float(result['max_bx_price']) if result and result.get('max_bx_price') else None,
                'node_count': result['node_count'] if result else 0,
                'day_count': result['day_count'] if result else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting B{bx} average: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def get_bx_trend(
        self,
        bx: int,
        start_date: date = None,
        end_date: date = None,
        zone: str = None,
        nodes: List[str] = None,
        aggregation: str = 'daily'
    ) -> List[Dict]:
        """
        Get BX price trend over time.
        
        Args:
            bx: BX type (4-10)
            start_date: Optional start date
            end_date: Optional end date
            zone: Optional zone filter
            nodes: Optional list of node names to filter by
            aggregation: 'daily', 'weekly', or 'monthly'
            
        Returns:
            List of dicts with date and average price
        """
        conditions = ["bx_type = %s"]
        params = [bx]
        
        if start_date:
            conditions.append("s.opr_dt >= %s")
            params.append(start_date)
        
        if end_date:
            conditions.append("s.opr_dt <= %s")
            params.append(end_date)
        
        if nodes:
            if isinstance(nodes, str):
                nodes = [nodes]
            placeholders = ','.join(['%s'] * len(nodes))
            conditions.append(f"s.node IN ({placeholders})")
            params.extend(nodes)
        
        zone_join = ""
        if zone:
            zone_join = "JOIN caiso.node_zone_mapping m ON s.node = m.pnode_id"
            conditions.append("m.zone = %s")
            params.append(zone)
        
        where_clause = " AND ".join(conditions)
        
        # Date truncation based on aggregation
        if aggregation == 'weekly':
            date_expr = "DATE_TRUNC('week', s.opr_dt)"
        elif aggregation == 'monthly':
            date_expr = "DATE_TRUNC('month', s.opr_dt)"
        else:
            date_expr = "s.opr_dt"
        
        query = f"""
            SELECT 
                {date_expr} as period,
                AVG(s.avg_price) as avg_price,
                COUNT(DISTINCT s.node) as node_count
            FROM caiso.bx_daily_summary s
            {zone_join}
            WHERE {where_clause}
            GROUP BY {date_expr}
            ORDER BY {date_expr}
        """
        
        try:
            results = self.db.execute_query(query, params)
            return [
                {
                    'date': row['period'],
                    'avg_price': float(row['avg_price']) if row['avg_price'] else None,
                    'node_count': row['node_count']
                }
                for row in results
            ]
        except Exception as e:
            self.logger.error(f"Error getting B{bx} trend: {str(e)}")
            return []
    
    def aggregate_monthly_summaries(self, year: int = None, month: int = None) -> Dict[str, Any]:
        """
        Aggregate daily BX summaries into monthly summaries.
        
        Args:
            year: Optional year filter (aggregates all years if not provided)
            month: Optional month filter (1-12, aggregates all months if not provided)
            
        Returns:
            Dict with aggregation results
        """
        self.create_monthly_summary_table()
        
        conditions = []
        params = []
        
        if year:
            conditions.append("EXTRACT(YEAR FROM opr_dt) = %s")
            params.append(year)
        if month:
            conditions.append("EXTRACT(MONTH FROM opr_dt) = %s")
            params.append(month)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            INSERT INTO caiso.bx_monthly_summary (node, year_month, bx_type, avg_price, min_price, max_price, day_count)
            SELECT 
                node,
                TO_CHAR(opr_dt, 'YYYY-MM') as year_month,
                bx_type,
                ROUND(AVG(avg_price)::numeric, 2) as avg_price,
                ROUND(MIN(avg_price)::numeric, 2) as min_price,
                ROUND(MAX(avg_price)::numeric, 2) as max_price,
                COUNT(DISTINCT opr_dt) as day_count
            FROM caiso.bx_daily_summary
            {where_clause}
            GROUP BY node, TO_CHAR(opr_dt, 'YYYY-MM'), bx_type
            ON CONFLICT (node, year_month, bx_type) DO UPDATE SET
                avg_price = EXCLUDED.avg_price,
                min_price = EXCLUDED.min_price,
                max_price = EXCLUDED.max_price,
                day_count = EXCLUDED.day_count,
                created_at = CURRENT_TIMESTAMP
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows_affected = cur.rowcount
                    conn.commit()
            
            self.logger.info(f"Aggregated {rows_affected} monthly summary records")
            return {'success': True, 'rows_affected': rows_affected}
        except Exception as e:
            self.logger.error(f"Error aggregating monthly summaries: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def aggregate_annual_summaries(self, year: int = None) -> Dict[str, Any]:
        """
        Aggregate daily BX summaries into annual summaries.
        
        Args:
            year: Optional year filter (aggregates all years if not provided)
            
        Returns:
            Dict with aggregation results
        """
        self.create_annual_summary_table()
        
        conditions = []
        params = []
        
        if year:
            conditions.append("EXTRACT(YEAR FROM opr_dt) = %s")
            params.append(year)
        
        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
        
        query = f"""
            INSERT INTO caiso.bx_annual_summary (node, year, bx_type, avg_price, min_price, max_price, day_count)
            SELECT 
                node,
                EXTRACT(YEAR FROM opr_dt)::integer as year,
                bx_type,
                ROUND(AVG(avg_price)::numeric, 2) as avg_price,
                ROUND(MIN(avg_price)::numeric, 2) as min_price,
                ROUND(MAX(avg_price)::numeric, 2) as max_price,
                COUNT(DISTINCT opr_dt) as day_count
            FROM caiso.bx_daily_summary
            {where_clause}
            GROUP BY node, EXTRACT(YEAR FROM opr_dt), bx_type
            ON CONFLICT (node, year, bx_type) DO UPDATE SET
                avg_price = EXCLUDED.avg_price,
                min_price = EXCLUDED.min_price,
                max_price = EXCLUDED.max_price,
                day_count = EXCLUDED.day_count,
                created_at = CURRENT_TIMESTAMP
        """
        
        try:
            with self.db.get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(query, params)
                    rows_affected = cur.rowcount
                    conn.commit()
            
            self.logger.info(f"Aggregated {rows_affected} annual summary records")
            return {'success': True, 'rows_affected': rows_affected}
        except Exception as e:
            self.logger.error(f"Error aggregating annual summaries: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def run_post_import_aggregation(self) -> Dict[str, Any]:
        """
        Run full aggregation after data import.
        Creates monthly and annual summaries from daily data.
        
        Returns:
            Dict with aggregation results
        """
        self.logger.info("Running post-import aggregation...")
        
        monthly_result = self.aggregate_monthly_summaries()
        annual_result = self.aggregate_annual_summaries()
        
        return {
            'success': monthly_result['success'] and annual_result['success'],
            'monthly': monthly_result,
            'annual': annual_result
        }
    
    def get_annual_bx_average(
        self,
        bx: int,
        year: int = None,
        zone: str = None,
        nodes: List[str] = None
    ) -> Dict[str, Any]:
        """
        Get average BX price from annual summary table (fast).
        
        Args:
            bx: BX type (4-10)
            year: Year to query (defaults to most recent)
            zone: Optional zone filter
            nodes: Optional list of node names
            
        Returns:
            Dict with average price and stats
        """
        conditions = ["bx_type = %s"]
        params = [bx]
        
        if year:
            conditions.append("s.year = %s")
            params.append(year)
        
        if nodes:
            if isinstance(nodes, str):
                nodes = [nodes]
            placeholders = ','.join(['%s'] * len(nodes))
            conditions.append(f"s.node IN ({placeholders})")
            params.extend(nodes)
        
        zone_join = ""
        if zone:
            zone_join = "JOIN caiso.node_zone_mapping m ON s.node = m.pnode_id"
            conditions.append("m.zone = %s")
            params.append(zone)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                s.year,
                AVG(s.avg_price) as avg_bx_price,
                MIN(s.min_price) as min_bx_price,
                MAX(s.max_price) as max_bx_price,
                COUNT(DISTINCT s.node) as node_count,
                SUM(s.day_count) / COUNT(DISTINCT s.node) as avg_days
            FROM caiso.bx_annual_summary s
            {zone_join}
            WHERE {where_clause}
            GROUP BY s.year
            ORDER BY s.year DESC
            LIMIT 1
        """
        
        try:
            result = self.db.execute_query(query, params, fetch_all=False)
            return {
                'success': True,
                'bx_type': bx,
                'year': result['year'] if result else None,
                'avg_price': float(result['avg_bx_price']) if result and result.get('avg_bx_price') else None,
                'min_price': float(result['min_bx_price']) if result and result.get('min_bx_price') else None,
                'max_price': float(result['max_bx_price']) if result and result.get('max_bx_price') else None,
                'node_count': result['node_count'] if result else 0,
                'avg_days': int(result['avg_days']) if result and result.get('avg_days') else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting annual B{bx} average: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_node_bx_from_parquet(
        self,
        bx: int,
        nodes: List[str],
        year: int = 2024
    ) -> Dict[str, Any]:
        """
        Compute BX average for specific nodes from parquet files.
        
        Used when individual node data is requested (not available in summary tables).
        """
        if not nodes:
            return {'success': False, 'error': 'No nodes specified'}
        
        available_dates = self.parquet.list_available_dates(year=year)
        if not available_dates:
            return {'success': False, 'error': 'No parquet data available'}
        
        all_bx_prices = []
        nodes_set = set(nodes)
        
        for d in available_dates:
            try:
                table = self.parquet.read_day_from_parquet(d)
                if table is None:
                    continue
                
                df = table.to_pandas()
                node_data = df[df['node'].isin(nodes_set)]
                if node_data.empty:
                    continue
                
                for node in nodes_set:
                    node_df = node_data[node_data['node'] == node]
                    if len(node_df) >= bx:
                        cheapest = node_df.nsmallest(bx, 'mw')['mw'].mean()
                        all_bx_prices.append(cheapest)
            except Exception as e:
                self.logger.debug(f"Error processing {d}: {e}")
                continue
        
        if not all_bx_prices:
            return {'success': False, 'error': 'No data found for selected nodes'}
        
        return {
            'success': True,
            'bx_type': bx,
            'avg_price': sum(all_bx_prices) / len(all_bx_prices),
            'min_price': min(all_bx_prices),
            'max_price': max(all_bx_prices),
            'node_count': len(nodes),
            'day_count': len(available_dates)
        }

    def get_zone_level_bx(
        self,
        bx: int,
        zone: str = None,
        year: int = None,
        start_date: date = None,
        end_date: date = None
    ) -> Dict[str, Any]:
        """
        Get zone-level BX average from bx_daily_summary table.
        
        The bx_daily_summary table now stores zone-level data directly
        with zone names in the 'node' column (NP15, SP15, ZP26, Overall).
        
        Args:
            bx: BX type (4-10)
            zone: Zone filter (NP15, SP15, ZP26) or None for overall
            year: Year to query (used if start/end not specified)
            start_date: Optional start date
            end_date: Optional end date
            
        Returns:
            Dict with avg_price and stats
        """
        conditions = ["bx_type = %s"]
        params = [bx]
        
        zone_name = zone if zone else 'Overall'
        conditions.append("node = %s")
        params.append(zone_name)
        
        if start_date and end_date:
            conditions.append("opr_dt >= %s AND opr_dt <= %s")
            params.extend([start_date, end_date])
        elif year:
            conditions.append("EXTRACT(YEAR FROM opr_dt) = %s")
            params.append(year)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                AVG(avg_price) as avg_bx_price,
                MIN(avg_price) as min_bx_price,
                MAX(avg_price) as max_bx_price,
                COUNT(*) as day_count
            FROM caiso.bx_daily_summary
            WHERE {where_clause}
        """
        
        try:
            result = self.db.execute_query(query, params, fetch_all=False)
            return {
                'success': True,
                'bx_type': bx,
                'avg_price': float(result['avg_bx_price']) if result and result.get('avg_bx_price') else None,
                'min_price': float(result['min_bx_price']) if result and result.get('min_bx_price') else None,
                'max_price': float(result['max_bx_price']) if result and result.get('max_bx_price') else None,
                'day_count': result['day_count'] if result else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting zone-level B{bx}: {str(e)}")
            return {'success': False, 'error': str(e)}


    def get_zone_level_bx_OLD(
        self,
        bx: int,
        zone: str = None,
        year: int = None,
        start_date: date = None,
        end_date: date = None
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Old method that computed from raw lmp_data.
        Kept for reference only.
        """
        zone_join = ""
        zone_condition = ""
        params = []
        
        if zone:
            zone_join = "JOIN caiso.node_zone_mapping m ON l.node = m.pnode_id"
            zone_condition = "AND m.zone = %s"
            params.append(zone)
        
        date_condition = ""
        if start_date and end_date:
            date_condition = "AND l.opr_dt >= %s AND l.opr_dt <= %s"
            params.extend([start_date, end_date])
        elif year:
            date_condition = "AND EXTRACT(YEAR FROM l.opr_dt) = %s"
            params.append(year)
        
        query = f"""
            WITH zone_hourly AS (
                SELECT 
                    l.opr_dt,
                    l.opr_hr as hour,
                    AVG(l.mw) as zone_avg_price
                FROM caiso.lmp_data l
                {zone_join}
                WHERE 1=1 {zone_condition} {date_condition}
                GROUP BY l.opr_dt, l.opr_hr
            ),
            ranked_hours AS (
                SELECT 
                    opr_dt,
                    hour,
                    zone_avg_price,
                    ROW_NUMBER() OVER (PARTITION BY opr_dt ORDER BY zone_avg_price ASC) as rn
                FROM zone_hourly
            ),
            daily_bx AS (
                SELECT 
                    opr_dt,
                    AVG(zone_avg_price) as bx_price
                FROM ranked_hours
                WHERE rn <= %s
                GROUP BY opr_dt
            )
            SELECT 
                AVG(bx_price) as avg_bx_price,
                MIN(bx_price) as min_bx_price,
                MAX(bx_price) as max_bx_price,
                COUNT(*) as day_count
            FROM daily_bx
        """
        params.append(bx)
        
        try:
            result = self.db.execute_query(query, params, fetch_all=False)
            return {
                'success': True,
                'bx_type': bx,
                'avg_price': float(result['avg_bx_price']) if result and result.get('avg_bx_price') else None,
                'min_price': float(result['min_bx_price']) if result and result.get('min_bx_price') else None,
                'max_price': float(result['max_bx_price']) if result and result.get('max_bx_price') else None,
                'day_count': result['day_count'] if result else 0
            }
        except Exception as e:
            self.logger.error(f"Error getting zone-level B{bx}: {str(e)}")
            return {'success': False, 'error': str(e)}

    def get_all_zones_bx_average(
        self,
        bx: int,
        year: int = None,
        time_period: str = "Annual",
        month: int = None
    ) -> Dict[str, Any]:
        """
        Get zone-level BX average for all zones plus overall.
        
        Uses zone-level BX calculation (cheapest X hours based on zone-average prices)
        rather than averaging per-node BX values.
        
        Returns dict with keys: 'NP15', 'SP15', 'ZP26', 'Overall'
        """
        from calendar import monthrange
        
        zones = ['NP15', 'SP15', 'ZP26']
        results = {}
        
        if time_period == "Annual":
            start_date = None
            end_date = None
        else:
            start_date = date(year, month, 1)
            _, last_day = monthrange(year, month)
            end_date = date(year, month, last_day)
        
        for zone in zones:
            stats = self.get_zone_level_bx(
                bx=bx,
                zone=zone,
                year=year if time_period == "Annual" else None,
                start_date=start_date,
                end_date=end_date
            )
            results[zone] = stats
        
        overall = self.get_zone_level_bx(
            bx=bx,
            zone=None,
            year=year if time_period == "Annual" else None,
            start_date=start_date,
            end_date=end_date
        )
        results['Overall'] = overall
        
        return results

    def get_hourly_averages_by_zone(self, year: int = None) -> Dict[str, Any]:
        """
        Get hourly price averages for each zone plus overall.
        
        Returns dict with zone names as keys, each containing list of 24 hourly averages.
        """
        zones = ['NP15', 'SP15', 'ZP26']
        results = {}
        
        year_filter = f"AND EXTRACT(YEAR FROM l.opr_dt) = {year}" if year else ""
        
        # Query for each zone
        for zone in zones:
            query = f"""
                SELECT 
                    l.opr_hr as hour,
                    AVG(l.mw) as avg_price
                FROM caiso.lmp_data l
                JOIN caiso.node_zone_mapping m ON l.node = m.pnode_id
                WHERE m.zone = %s {year_filter}
                GROUP BY l.opr_hr
                ORDER BY hour
            """
            try:
                data = self.db.execute_query(query, (zone,))
                results[zone] = [{'hour': int(r['hour']), 'avg_price': float(r['avg_price'])} for r in data] if data else []
            except Exception as e:
                self.logger.error(f"Error getting hourly averages for {zone}: {str(e)}")
                results[zone] = []
        
        # Overall (no zone filter)
        query = f"""
            SELECT 
                opr_hr as hour,
                AVG(mw) as avg_price
            FROM caiso.lmp_data
            WHERE 1=1 {year_filter}
            GROUP BY opr_hr
            ORDER BY hour
        """
        try:
            data = self.db.execute_query(query)
            results['Overall'] = [{'hour': int(r['hour']), 'avg_price': float(r['avg_price'])} for r in data] if data else []
        except Exception as e:
            self.logger.error(f"Error getting overall hourly averages: {str(e)}")
            results['Overall'] = []
        
        return results

    def get_month_hour_averages(self, zone: str = None, year: int = None) -> List[Dict]:
        """
        Get average prices by month and hour for heatmap display.
        
        Uses pre-computed month_hour_summary table.
        
        Args:
            zone: Optional zone filter (NP15, SP15, ZP26), None for Overall
            year: Optional year filter
            
        Returns:
            List of dicts with 'month', 'hour', 'avg_price'
        """
        params = []
        zone_filter = "zone = %s"
        params.append(zone if zone else 'Overall')
        
        year_filter = ""
        if year:
            year_filter = "AND year = %s"
            params.append(year)
        
        query = f"""
            SELECT month, hour, avg_price
            FROM caiso.month_hour_summary
            WHERE {zone_filter} {year_filter}
            ORDER BY month, hour
        """
        
        try:
            result = self.db.execute_query(query, params)
            return [
                {
                    'month': int(r['month']),
                    'hour': int(r['hour']),
                    'avg_price': float(r['avg_price'])
                }
                for r in result
            ] if result else []
        except Exception as e:
            self.logger.error(f"Error getting month/hour averages: {str(e)}")
            return []

    def get_all_nodes(self) -> List[str]:
        """Get all distinct PNODE names for autocomplete. Sorted alphabetically."""
        try:
            query = "SELECT DISTINCT pnode_id FROM caiso.node_zone_mapping ORDER BY pnode_id"
            results = self.db.execute_query(query)
            return [r['pnode_id'] for r in results] if results else []
        except Exception as e:
            self.logger.error(f"Error getting all nodes: {str(e)}")
            return []

    def get_available_years(self) -> List[int]:
        """Get list of years with data in annual summary table."""
        try:
            query = "SELECT DISTINCT year FROM caiso.bx_annual_summary ORDER BY year DESC"
            results = self.db.execute_query(query)
            return [r['year'] for r in results] if results else []
        except Exception:
            # Fallback to daily summary if annual not populated
            try:
                query = "SELECT DISTINCT EXTRACT(YEAR FROM opr_dt)::integer as year FROM caiso.bx_daily_summary ORDER BY year DESC"
                results = self.db.execute_query(query)
                return [r['year'] for r in results] if results else [2024]
            except Exception:
                return [2024]
    
    def get_available_months(self, year: int) -> List[str]:
        """Get list of months with data for a given year."""
        try:
            query = """
                SELECT DISTINCT year_month 
                FROM caiso.bx_monthly_summary 
                WHERE year_month LIKE %s
                ORDER BY year_month
            """
            results = self.db.execute_query(query, (f"{year}-%",))
            return [r['year_month'] for r in results] if results else []
        except Exception:
            return []


    def get_hourly_averages_for_nodes(self, nodes: List[str], year: int = None) -> List[Dict]:
        """
        Get hourly price averages for a list of nodes from parquet files.
        
        Returns list of {'hour': int, 'avg_price': float} dicts.
        """
        if not nodes:
            return []
        
        year = year or 2024
        available_dates = self.parquet.list_available_dates(year=year)
        if not available_dates:
            return []
        
        nodes_set = set(nodes)
        hour_totals = {h: {'sum': 0, 'count': 0} for h in range(1, 25)}
        
        for d in available_dates:
            try:
                table = self.parquet.read_day_from_parquet(d)
                if table is None:
                    continue
                df = table.to_pandas()
                node_data = df[df['node'].isin(nodes_set)]
                if node_data.empty:
                    continue
                for hour in range(1, 25):
                    hour_df = node_data[node_data['opr_hr'] == hour]
                    if not hour_df.empty:
                        hour_totals[hour]['sum'] += hour_df['mw'].sum()
                        hour_totals[hour]['count'] += len(hour_df)
            except Exception:
                continue
        
        return [
            {'hour': h, 'avg_price': t['sum'] / t['count']}
            for h, t in hour_totals.items() if t['count'] > 0
        ]
    
    def get_bx_trend_by_zone(
        self,
        bx: int,
        year: int,
        aggregation: str = 'monthly'
    ) -> Dict[str, List[Dict]]:
        """
        Get BX price trend over time for each zone.
        
        Returns dict with zone names as keys, each containing list of
        {'date': date, 'avg_price': float} dicts.
        """
        zones = ['NP15', 'SP15', 'ZP26']
        results = {}
        
        for zone in zones:
            trend = self.get_bx_trend(
                bx=bx,
                start_date=date(year, 1, 1),
                end_date=date(year, 12, 31),
                zone=zone,
                aggregation=aggregation
            )
            results[zone] = trend
        
        # Overall (no zone filter)
        overall = self.get_bx_trend(
            bx=bx,
            start_date=date(year, 1, 1),
            end_date=date(year, 12, 31),
            aggregation=aggregation
        )
        results['Overall'] = overall
        
        return results
    
    def get_bx_trend_per_node(
        self,
        bx: int,
        nodes: List[str],
        year: int,
        aggregation: str = 'monthly'
    ) -> Dict[str, List[Dict]]:
        """
        Get BX price trend for each specified node from parquet files.
        
        Returns dict with node names as keys, each containing list of
        {'date': date, 'avg_price': float} dicts.
        """
        if not nodes:
            return {}
        
        available_dates = self.parquet.list_available_dates(year=year)
        if not available_dates:
            return {}
        
        nodes_set = set(nodes[:20])
        from collections import defaultdict
        node_monthly = defaultdict(lambda: defaultdict(list))
        
        for d in available_dates:
            try:
                table = self.parquet.read_day_from_parquet(d)
                if table is None:
                    continue
                df = table.to_pandas()
                node_data = df[df['node'].isin(nodes_set)]
                if node_data.empty:
                    continue
                
                month_key = date(d.year, d.month, 1)
                for node in nodes_set:
                    node_df = node_data[node_data['node'] == node]
                    if len(node_df) >= bx:
                        bx_price = node_df.nsmallest(bx, 'mw')['mw'].mean()
                        node_monthly[node][month_key].append(bx_price)
            except Exception:
                continue
        
        results = {}
        for node in nodes_set:
            monthly_data = node_monthly[node]
            results[node] = [
                {'date': m, 'avg_price': sum(prices) / len(prices)}
                for m, prices in sorted(monthly_data.items()) if prices
            ]
        
        return results
    
    def get_node_summary_statistics(
        self,
        bx: int,
        nodes: List[str],
        year: int
    ) -> List[Dict]:
        """
        Get summary statistics (for box plot) for each node from parquet files.
        
        Returns list of dicts with node, avg, min, max, q1, median, q3.
        """
        if not nodes:
            return []
        
        import numpy as np
        available_dates = self.parquet.list_available_dates(year=year)
        if not available_dates:
            return []
        
        nodes_set = set(nodes[:20])
        from collections import defaultdict
        node_bx_prices = defaultdict(list)
        
        for d in available_dates:
            try:
                table = self.parquet.read_day_from_parquet(d)
                if table is None:
                    continue
                df = table.to_pandas()
                node_data = df[df['node'].isin(nodes_set)]
                if node_data.empty:
                    continue
                
                for node in nodes_set:
                    node_df = node_data[node_data['node'] == node]
                    if len(node_df) >= bx:
                        bx_price = node_df.nsmallest(bx, 'mw')['mw'].mean()
                        node_bx_prices[node].append(bx_price)
            except Exception:
                continue
        
        results = []
        for node in nodes:
            if node not in node_bx_prices or not node_bx_prices[node]:
                continue
            prices = node_bx_prices[node]
            results.append({
                'node': node,
                'mean': float(np.mean(prices)),
                'min': float(np.min(prices)),
                'max': float(np.max(prices)),
                'q1': float(np.percentile(prices, 25)),
                'median': float(np.median(prices)),
                'q3': float(np.percentile(prices, 75)),
                'day_count': len(prices)
            })
        
        return sorted(results, key=lambda x: x['mean'])


if __name__ == "__main__":
    # Test the module
    logging.basicConfig(level=logging.INFO)
    
    calculator = BXCalculator()
    
    print("Creating tables...")
    calculator.create_bx_table()
    calculator.create_bx_summary_table()
    
    print("\nTesting single date calculation...")
    from datetime import date
    test_date = date(2024, 1, 15)
    result = calculator.calculate_all_bx_for_date(test_date)
    print(f"Result: {result}")
