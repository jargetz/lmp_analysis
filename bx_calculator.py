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
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
from database import DatabaseManager

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
            
            # Build records for all BX values from the cached data
            bx_records = []
            summary_records = []
            nodes_processed = 0
            
            for node, hours in node_hours.items():
                # Skip if not enough hours for the smallest BX we need
                if len(hours) < min(bx_values):
                    continue
                
                nodes_processed += 1
                
                # For each BX type, extract the cheapest X hours
                for bx in bx_values:
                    if len(hours) < bx:
                        continue
                    
                    cheapest_hours = hours[:bx]
                    prices = [float(h['mw']) for h in cheapest_hours]
                    hours_used = [h['opr_hr'] for h in cheapest_hours]
                    
                    # Build hour records
                    for rank, hour_data in enumerate(cheapest_hours, 1):
                        bx_records.append({
                            'node': node,
                            'opr_dt': hour_data['opr_dt'],
                            'opr_hr': hour_data['opr_hr'],
                            'mw': hour_data['mw'],
                            'hour_rank': rank,
                            'bx_type': bx
                        })
                    
                    # Build summary record
                    summary_records.append({
                        'node': node,
                        'opr_dt': target_date,
                        'bx_type': bx,
                        'avg_price': sum(prices) / len(prices),
                        'min_hour': min(hours_used),
                        'max_hour': max(hours_used)
                    })
            
            # Bulk insert all records
            hours_inserted = self._insert_bx_hours(bx_records) if bx_records else 0
            summary_inserted = self._insert_bx_summary(summary_records) if summary_records else 0
            
            return {
                'success': True,
                'date': target_date,
                'bx_values': bx_values,
                'nodes_processed': nodes_processed,
                'hours_inserted': hours_inserted,
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
                    # Delete from bx_hours
                    cur.execute(
                        "DELETE FROM caiso.bx_hours WHERE opr_dt = %s AND bx_type = ANY(%s)",
                        (target_date, bx_values)
                    )
                    # Delete from bx_daily_summary
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
        
        # Ensure tables exist
        self.create_bx_table()
        self.create_bx_summary_table()
        
        # Build list of dates
        dates = []
        current = start_date
        while current <= end_date:
            dates.append(current)
            current += timedelta(days=1)
        
        total_dates = len(dates)
        processed = 0
        total_records = 0
        
        for i, d in enumerate(dates):
            if progress_callback:
                progress_callback(i, total_dates, f"Processing {d} for B{min(bx_values)}-B{max(bx_values)}")
            
            result = self.calculate_all_bx_for_date(d, bx_values)
            
            if result.get('success'):
                processed += 1
                total_records += result.get('hours_inserted', 0)
        
        return {
            'success': True,
            'dates_processed': processed,
            'total_dates': total_dates,
            'total_records': total_records,
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
                FROM caiso.bx_hours 
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
        node: str = None
    ) -> Dict[str, Any]:
        """
        Get average BX price with optional filters.
        
        Args:
            bx: BX type (4-10)
            start_date: Optional start date filter
            end_date: Optional end date filter
            zone: Optional zone filter (requires node_zone_mapping table)
            node: Optional specific node filter
            
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
        
        if node:
            conditions.append("s.node = %s")
            params.append(node)
        
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
        aggregation: str = 'daily'
    ) -> List[Dict]:
        """
        Get BX price trend over time.
        
        Args:
            bx: BX type (4-10)
            start_date: Optional start date
            end_date: Optional end date
            zone: Optional zone filter
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
