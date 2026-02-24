"""
BX Hours Calculator Module

Calculates the cheapest X hours per day (BX) for CAISO LMP data.
Supports B4 through B10 with a unified, parameterized approach.

The BX value represents the average price of the X cheapest hours
in a given day for each node. This is commonly used for evaluating
battery storage charging strategies.

All queries now use MotherDuck (DuckDB cloud) for both summary tables
and S3 parquet raw data. PostgreSQL is no longer required.

Usage:
    calculator = BXCalculator()
    
    # Get zone-level BX average
    result = calculator.get_zone_level_bx(bx=8, year=2024)
    
    # Get node-level BX from parquet
    result = calculator.get_node_bx_from_parquet(bx=8, nodes=['TH_NP15_GEN-APND'], year=2024)
"""

import logging
import pandas as pd
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
from parquet_storage import ParquetStorage

SUPPORTED_BX_VALUES = [4, 5, 6, 7, 8, 9, 10]

MIN_BX = 4
MAX_BX = 10


class BXCalculator:
    """Calculates cheapest X hours (BX) for CAISO LMP data using MotherDuck"""
    
    def __init__(self, use_motherduck: bool = True):
        self.logger = logging.getLogger(__name__)
        self._parquet = None
        self._motherduck = None
        self._use_motherduck = use_motherduck
    
    @property
    def parquet(self):
        """Lazy-load parquet storage"""
        if self._parquet is None:
            self._parquet = ParquetStorage()
        return self._parquet
    
    def _md_query(self, query: str, params: list = None) -> List[Dict]:
        """Execute a query via MotherDuck subprocess and return list of dicts"""
        import subprocess
        import json
        cmd = ['python3', 'subprocess_query.py', 'raw_sql', query]
        if params:
            cmd.append(json.dumps(params))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout.strip())
                if isinstance(data, dict) and 'error' in data:
                    raise RuntimeError(data['error'])
                return data
            return []
        except subprocess.TimeoutExpired:
            self.logger.error("MotherDuck query timed out")
            return []

    def _md_query_one(self, query: str, params: list = None) -> Optional[Dict]:
        """Execute a query via MotherDuck and return first row as dict"""
        results = self._md_query(query, params)
        return results[0] if results else None

    def _sanitize_zone(self, zone: str) -> str:
        """Validate zone name against whitelist"""
        import re
        valid = ['NP15', 'SP15', 'ZP26', 'Overall']
        if zone in valid:
            return zone
        raise ValueError(f"Invalid zone: {zone}")

    def get_zone_level_bx(
        self,
        bx: int,
        zone: str = None,
        year: int = None,
        start_date: date = None,
        end_date: date = None
    ) -> Dict[str, Any]:
        """
        Get zone-level BX average from bx_daily_summary table in MotherDuck.
        """
        bx = int(bx)
        zone_name = self._sanitize_zone(zone if zone else 'Overall')
        
        conditions = [f"bx_type = {bx}", f"node = '{zone_name}'"]
        
        if start_date and end_date:
            conditions.append(f"opr_dt >= '{start_date}' AND opr_dt <= '{end_date}'")
        elif year:
            conditions.append(f"EXTRACT(YEAR FROM opr_dt) = {int(year)}")
        
        query = f"""
            SELECT 
                AVG(avg_price) as avg_bx_price,
                MIN(avg_price) as min_bx_price,
                MAX(avg_price) as max_bx_price,
                COUNT(*) as day_count
            FROM bx_daily_summary
            WHERE {' AND '.join(conditions)}
        """
        
        try:
            result = self._md_query_one(query)
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

    def _monthly_weighted_avg(self, rows, price_col='monthly_avg', year=2024):
        """Compute monthly-weighted annual average.
        Weight each month's average by the number of calendar days in that month.
        Formula: sum(month_avg * calendar_days_in_month) / total_calendar_days_in_year
        """
        from calendar import monthrange, isleap
        total_calendar_days = 366 if isleap(year) else 365
        weighted_sum = 0
        weighted_days = 0
        for row in rows:
            month_num = int(row['month_num'])
            _, cal_days = monthrange(year, month_num)
            weighted_sum += float(row[price_col]) * cal_days
            weighted_days += cal_days
        if weighted_days == 0:
            return None
        return weighted_sum / total_calendar_days

    def get_all_zones_load_weighted_bx(
        self,
        bx: int,
        year: int = None,
        time_period: str = "Annual",
        month: int = None
    ) -> Dict[str, Any]:
        """
        Get zone-level BX average computed from zone_hourly_lmp (EIA load-weighted zone prices).
        Uses monthly weighting for annual averages.
        """
        from calendar import monthrange
        
        bx = int(bx)
        year = int(year) if year else 2024
        
        if time_period == "Monthly" and month:
            start_date = date(year, month, 1)
            _, last_day = monthrange(year, month)
            end_date = date(year, month, last_day)
            query = f"""
                WITH ranked AS (
                    SELECT zone, opr_dt, hour_num, lmp,
                        ROW_NUMBER() OVER (PARTITION BY zone, opr_dt ORDER BY lmp ASC) as rn
                    FROM zone_hourly_lmp
                    WHERE opr_dt >= '{start_date}' AND opr_dt <= '{end_date}'
                      AND hour_num <= 24
                ),
                daily_bx AS (
                    SELECT zone, opr_dt, AVG(lmp) as bx_price
                    FROM ranked WHERE rn <= {bx}
                    GROUP BY zone, opr_dt
                )
                SELECT zone as node, AVG(bx_price) as avg_bx_price,
                       MIN(bx_price) as min_bx_price, MAX(bx_price) as max_bx_price,
                       COUNT(*) as day_count
                FROM daily_bx GROUP BY zone
            """
            try:
                rows = self._md_query(query)
                results = {}
                for row in rows:
                    zone_name = row['node']
                    results[zone_name] = {
                        'success': True, 'bx_type': bx,
                        'avg_price': float(row['avg_bx_price']) if row.get('avg_bx_price') else None,
                        'min_price': float(row['min_bx_price']) if row.get('min_bx_price') else None,
                        'max_price': float(row['max_bx_price']) if row.get('max_bx_price') else None,
                        'day_count': row['day_count'] if row.get('day_count') else 0
                    }
                if 'NP15' in results and 'SP15' in results and 'ZP26' in results:
                    valid_prices = [results[z]['avg_price'] for z in ['NP15','SP15','ZP26'] if results[z].get('avg_price') is not None]
                    results['Overall'] = {
                        'success': True, 'bx_type': bx,
                        'avg_price': sum(valid_prices) / len(valid_prices) if valid_prices else None,
                        'day_count': max(results[z].get('day_count', 0) for z in ['NP15','SP15','ZP26'])
                    }
                for zone in ['NP15', 'SP15', 'ZP26', 'Overall']:
                    if zone not in results:
                        results[zone] = {'success': False, 'error': 'No data'}
                return results
            except Exception as e:
                self.logger.error(f"Error getting load-weighted B{bx}: {str(e)}")
                return {z: {'success': False, 'error': str(e)} for z in ['NP15', 'SP15', 'ZP26', 'Overall']}
        else:
            query = f"""
                WITH ranked AS (
                    SELECT zone, opr_dt, hour_num, lmp,
                        ROW_NUMBER() OVER (PARTITION BY zone, opr_dt ORDER BY lmp ASC) as rn
                    FROM zone_hourly_lmp
                    WHERE EXTRACT(YEAR FROM opr_dt) = {year}
                      AND hour_num <= 24
                ),
                daily_bx AS (
                    SELECT zone, opr_dt, AVG(lmp) as bx_price
                    FROM ranked WHERE rn <= {bx}
                    GROUP BY zone, opr_dt
                )
                SELECT zone as node,
                       EXTRACT(MONTH FROM opr_dt) as month_num,
                       AVG(bx_price) as monthly_avg,
                       MIN(bx_price) as monthly_min,
                       MAX(bx_price) as monthly_max,
                       COUNT(*) as days_with_data
                FROM daily_bx
                GROUP BY zone, EXTRACT(MONTH FROM opr_dt)
                ORDER BY zone, month_num
            """
            try:
                rows = self._md_query(query)
                by_zone = {}
                for row in rows:
                    zone_name = row['node']
                    if zone_name not in by_zone:
                        by_zone[zone_name] = []
                    by_zone[zone_name].append(row)
                
                results = {}
                for zone_name, zone_rows in by_zone.items():
                    avg_price = self._monthly_weighted_avg(zone_rows, 'monthly_avg', year)
                    total_days = sum(int(r['days_with_data']) for r in zone_rows)
                    all_mins = [float(r['monthly_min']) for r in zone_rows]
                    all_maxs = [float(r['monthly_max']) for r in zone_rows]
                    results[zone_name] = {
                        'success': True, 'bx_type': bx,
                        'avg_price': avg_price,
                        'min_price': min(all_mins) if all_mins else None,
                        'max_price': max(all_maxs) if all_maxs else None,
                        'day_count': total_days
                    }
                
                if 'NP15' in results and 'SP15' in results and 'ZP26' in results:
                    valid_prices = [results[z]['avg_price'] for z in ['NP15','SP15','ZP26'] if results[z].get('avg_price') is not None]
                    results['Overall'] = {
                        'success': True, 'bx_type': bx,
                        'avg_price': sum(valid_prices) / len(valid_prices) if valid_prices else None,
                        'day_count': max(results[z].get('day_count', 0) for z in ['NP15','SP15','ZP26'])
                    }
                for zone in ['NP15', 'SP15', 'ZP26', 'Overall']:
                    if zone not in results:
                        results[zone] = {'success': False, 'error': 'No data'}
                return results
            except Exception as e:
                self.logger.error(f"Error getting load-weighted B{bx}: {str(e)}")
                return {z: {'success': False, 'error': str(e)} for z in ['NP15', 'SP15', 'ZP26', 'Overall']}

    def get_all_zones_bx_average(
        self,
        bx: int,
        year: int = None,
        time_period: str = "Annual",
        month: int = None
    ) -> Dict[str, Any]:
        """
        Get zone-level BX average from bx_daily_summary (unweighted node average).
        Uses monthly weighting for annual averages.
        """
        from calendar import monthrange
        
        bx = int(bx)
        conditions = [f"bx_type = {bx}"]
        
        if time_period == "Annual" and year:
            conditions.append(f"EXTRACT(YEAR FROM opr_dt) = {int(year)}")
        elif time_period == "Monthly" and year and month:
            start_date = date(year, month, 1)
            _, last_day = monthrange(year, month)
            end_date = date(year, month, last_day)
            conditions.append(f"opr_dt >= '{start_date}' AND opr_dt <= '{end_date}'")
        
        if time_period == "Monthly":
            query = f"""
                SELECT 
                    node,
                    AVG(avg_price) as avg_bx_price,
                    MIN(avg_price) as min_bx_price,
                    MAX(avg_price) as max_bx_price,
                    COUNT(*) as day_count
                FROM bx_daily_summary
                WHERE {' AND '.join(conditions)}
                    AND node IN ('NP15', 'SP15', 'ZP26', 'Overall')
                GROUP BY node
            """
            try:
                rows = self._md_query(query)
                results = {}
                for row in rows:
                    zone_name = row['node']
                    results[zone_name] = {
                        'success': True, 'bx_type': bx,
                        'avg_price': float(row['avg_bx_price']) if row.get('avg_bx_price') else None,
                        'min_price': float(row['min_bx_price']) if row.get('min_bx_price') else None,
                        'max_price': float(row['max_bx_price']) if row.get('max_bx_price') else None,
                        'day_count': row['day_count'] if row.get('day_count') else 0
                    }
                for zone in ['NP15', 'SP15', 'ZP26', 'Overall']:
                    if zone not in results:
                        results[zone] = {'success': False, 'error': 'No data'}
                return results
            except Exception as e:
                self.logger.error(f"Error getting all zones B{bx}: {str(e)}")
                return {z: {'success': False, 'error': str(e)} for z in ['NP15', 'SP15', 'ZP26', 'Overall']}
        else:
            year_int = int(year) if year else 2024
            query = f"""
                SELECT 
                    node,
                    EXTRACT(MONTH FROM opr_dt) as month_num,
                    AVG(avg_price) as monthly_avg,
                    MIN(avg_price) as monthly_min,
                    MAX(avg_price) as monthly_max,
                    COUNT(*) as days_with_data
                FROM bx_daily_summary
                WHERE {' AND '.join(conditions)}
                    AND node IN ('NP15', 'SP15', 'ZP26', 'Overall')
                GROUP BY node, EXTRACT(MONTH FROM opr_dt)
                ORDER BY node, month_num
            """
            try:
                rows = self._md_query(query)
                by_zone = {}
                for row in rows:
                    zone_name = row['node']
                    if zone_name not in by_zone:
                        by_zone[zone_name] = []
                    by_zone[zone_name].append(row)
                
                results = {}
                for zone_name, zone_rows in by_zone.items():
                    avg_price = self._monthly_weighted_avg(zone_rows, 'monthly_avg', year_int)
                    total_days = sum(int(r['days_with_data']) for r in zone_rows)
                    all_mins = [float(r['monthly_min']) for r in zone_rows]
                    all_maxs = [float(r['monthly_max']) for r in zone_rows]
                    results[zone_name] = {
                        'success': True, 'bx_type': bx,
                        'avg_price': avg_price,
                        'min_price': min(all_mins) if all_mins else None,
                        'max_price': max(all_maxs) if all_maxs else None,
                        'day_count': total_days
                    }
                for zone in ['NP15', 'SP15', 'ZP26', 'Overall']:
                    if zone not in results:
                        results[zone] = {'success': False, 'error': 'No data'}
                return results
            except Exception as e:
                self.logger.error(f"Error getting all zones B{bx}: {str(e)}")
                return {z: {'success': False, 'error': str(e)} for z in ['NP15', 'SP15', 'ZP26', 'Overall']}

    def get_generator_bx_average(
        self,
        bx: int,
        year: int,
        zone: str = None,
        time_period: str = "Annual",
        month: int = None
    ) -> Dict[str, Any]:
        """
        Get pre-computed BX averages for generator nodes from MotherDuck.
        """
        bx = int(bx)
        year = int(year)
        conditions = [f"bx_type = {bx}", f"EXTRACT(YEAR FROM opr_dt) = {year}"]
        
        if zone:
            zone = self._sanitize_zone(zone)
            conditions.append(f"zone = '{zone}'")
        
        if time_period == "Monthly" and month:
            month = int(month)
            if 1 <= month <= 12:
                conditions.append(f"EXTRACT(MONTH FROM opr_dt) = {month}")
        
        query = f"""
            SELECT 
                zone,
                node,
                EXTRACT(MONTH FROM opr_dt) as month_num,
                AVG(avg_price) as monthly_avg,
                MIN(avg_price) as monthly_min,
                MAX(avg_price) as monthly_max,
                COUNT(*) as days_with_data
            FROM generator_bx_summary
            WHERE {' AND '.join(conditions)}
            GROUP BY zone, node, EXTRACT(MONTH FROM opr_dt)
            ORDER BY zone, month_num
        """
        
        try:
            results = self._md_query(query)
            
            if not results:
                return {'success': False, 'error': 'No generator data found'}
            
            by_zone_months = {}
            for row in results:
                zone_name = row['zone']
                if zone_name not in by_zone_months:
                    by_zone_months[zone_name] = {'node': row['node'], 'months': []}
                by_zone_months[zone_name]['months'].append(row)
            
            by_zone = {}
            for zone_name, data in by_zone_months.items():
                zone_rows = data['months']
                if time_period == "Monthly" and month:
                    avg_price = float(zone_rows[0]['monthly_avg']) if zone_rows else None
                else:
                    avg_price = self._monthly_weighted_avg(zone_rows, 'monthly_avg', year)
                total_days = sum(int(r['days_with_data']) for r in zone_rows)
                all_mins = [float(r['monthly_min']) for r in zone_rows]
                all_maxs = [float(r['monthly_max']) for r in zone_rows]
                by_zone[zone_name] = {
                    'node': data['node'],
                    'avg_price': avg_price,
                    'min_price': min(all_mins) if all_mins else None,
                    'max_price': max(all_maxs) if all_maxs else None,
                    'day_count': total_days,
                    'success': True
                }
            
            return {'success': True, 'zones': by_zone}
            
        except Exception as e:
            self.logger.error(f"Error getting generator BX: {e}")
            return {'success': False, 'error': str(e)}

    def get_month_hour_averages(self, zone: str = None, year: int = None) -> List[Dict]:
        """
        Get average prices by month and hour for heatmap display.
        
        Derives from zone_hourly_lmp table in MotherDuck.
        """
        zone_name = self._sanitize_zone(zone if zone else 'Overall')
        conditions = [f"zone = '{zone_name}'"]
        
        if year:
            conditions.append(f"EXTRACT(YEAR FROM opr_dt) = {int(year)}")
        
        query = f"""
            SELECT 
                EXTRACT(MONTH FROM opr_dt)::INT as month, 
                hour_num as hour, 
                AVG(lmp) as avg_price
            FROM zone_hourly_lmp
            WHERE {' AND '.join(conditions)}
            GROUP BY 1, hour_num
            ORDER BY month, hour
        """
        
        try:
            results = self._md_query(query)
            return [
                {
                    'month': int(r['month']),
                    'hour': int(r['hour']),
                    'avg_price': float(r['avg_price'])
                }
                for r in results
            ] if results else []
        except Exception as e:
            self.logger.error(f"Error getting month/hour averages: {str(e)}")
            return []

    def get_all_zones_month_hour(self, year: int = None) -> Dict[str, List[Dict]]:
        """Get month/hour averages for all zones in one query."""
        conditions = []
        if year:
            conditions.append(f"EXTRACT(YEAR FROM opr_dt) = {int(year)}")
        
        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        
        hour_filter = "hour_num <= 24"
        if where_clause:
            zone_where = f"{where_clause} AND {hour_filter}"
        else:
            zone_where = f"WHERE {hour_filter}"
        
        query = f"""
            SELECT 
                zone,
                EXTRACT(MONTH FROM opr_dt)::INT as month, 
                hour_num as hour, 
                AVG(lmp) as avg_price
            FROM zone_hourly_lmp
            {zone_where}
            GROUP BY zone, month, hour_num
            UNION ALL
            SELECT 
                'Overall' as zone,
                EXTRACT(MONTH FROM opr_dt)::INT as month, 
                hour_num as hour, 
                AVG(lmp) as avg_price
            FROM zone_hourly_lmp
            {zone_where}
            GROUP BY month, hour_num
            ORDER BY zone, month, hour
        """
        
        try:
            results = self._md_query(query)
            by_zone = {}
            for r in results:
                z = r.get('zone', 'Overall')
                if z not in by_zone:
                    by_zone[z] = []
                by_zone[z].append({
                    'month': int(r['month']),
                    'hour': int(r['hour']),
                    'avg_price': float(r['avg_price'])
                })
            return by_zone
        except Exception as e:
            self.logger.error(f"Error getting all zones month/hour: {str(e)}")
            return {}

    def get_bx_trend_by_zone(
        self,
        bx: int,
        year: int,
        aggregation: str = 'monthly'
    ) -> Dict[str, List[Dict]]:
        """
        Get BX price trend over time for each zone from MotherDuck.
        
        The bx_daily_summary table stores zone names directly in the 'node' column.
        """
        zones = ['NP15', 'SP15', 'ZP26', 'Overall']
        results = {}
        bx = int(bx)
        year = int(year)
        
        if aggregation == 'weekly':
            date_expr = "DATE_TRUNC('week', opr_dt)"
        elif aggregation == 'monthly':
            date_expr = "DATE_TRUNC('month', opr_dt)"
        else:
            date_expr = "opr_dt"
        
        query = f"""
            SELECT 
                node as zone,
                {date_expr} as period,
                AVG(avg_price) as avg_price,
                COUNT(*) as day_count
            FROM bx_daily_summary
            WHERE bx_type = {bx}
              AND node IN ('NP15', 'SP15', 'ZP26', 'Overall')
              AND EXTRACT(YEAR FROM opr_dt) = {year}
            GROUP BY node, {date_expr}
            ORDER BY node, period
        """
        try:
            data = self._md_query(query)
            for zone in zones:
                results[zone] = [
                    {'date': r['period'], 'avg_price': float(r['avg_price'])}
                    for r in data if r.get('zone') == zone
                ]
        except Exception as e:
                self.logger.error(f"Error getting BX trend for {zone}: {e}")
                results[zone] = []
        
        return results

    def get_available_years(self) -> List[int]:
        """Get list of years with zone data available from MotherDuck."""
        try:
            results = self._md_query(
                "SELECT DISTINCT EXTRACT(YEAR FROM opr_dt)::INTEGER as year FROM zone_hourly_lmp ORDER BY year DESC"
            )
            return [r['year'] for r in results] if results else [2024]
        except Exception:
            try:
                results = self._md_query(
                    "SELECT DISTINCT EXTRACT(YEAR FROM opr_dt)::INTEGER as year FROM bx_daily_summary ORDER BY year DESC"
                )
                return [r['year'] for r in results] if results else [2024]
            except Exception:
                return [2024]

    def get_available_parquet_years(self) -> List[int]:
        """Get list of years with parquet node data available."""
        try:
            all_dates = self.parquet.list_available_dates()
            if not all_dates:
                return [2024]
            years = sorted(set(d.year for d in all_dates if 2015 <= d.year <= 2100), reverse=True)
            return years if years else [2024]
        except Exception:
            return [2024]

    def get_all_nodes(self) -> List[str]:
        """Get all distinct PNODE names from parquet files for autocomplete. Sorted alphabetically.
        
        Samples from all available years to ensure all node names are included.
        """
        try:
            all_nodes = set()
            available_years = self.get_available_parquet_years()
            
            for year in available_years:
                available_dates = self.parquet.list_available_dates(year=year)
                if not available_dates:
                    continue
                
                sample_date = available_dates[len(available_dates) // 2]
                table = self.parquet.read_day_from_parquet(sample_date)
                if table is not None:
                    df = table.to_pandas()
                    all_nodes.update(df['node'].unique().tolist())
            
            if not all_nodes:
                return self._get_nodes_from_mapping()
            
            return sorted(all_nodes)
        except Exception as e:
            self.logger.error(f"Error getting nodes from parquet: {str(e)}")
            return self._get_nodes_from_mapping()
    
    def _get_nodes_from_mapping(self) -> List[str]:
        """Fallback: get nodes from zone mapping table in MotherDuck."""
        try:
            results = self._md_query("SELECT DISTINCT pnode_id FROM node_zone_mapping ORDER BY pnode_id")
            return [r['pnode_id'] for r in results] if results else []
        except Exception:
            return []

    def get_node_bx_from_parquet(
        self,
        bx: int,
        nodes: List[str],
        year: int = 2024
    ) -> Dict[str, Any]:
        """
        Compute BX average for specific nodes from parquet files.
        
        Uses MotherDuck for fast SQL queries if available, falls back to parquet.
        """
        if not nodes:
            return {'success': False, 'error': 'No nodes specified'}
        
        if self.motherduck:
            try:
                result = self.motherduck.get_node_bx_from_parquet(bx, nodes, year)
                if result.get('success'):
                    return result
            except Exception as e:
                self.logger.warning(f"MotherDuck BX query failed, falling back to parquet: {e}")
        
        available_dates = self.parquet.list_available_dates(year=year)
        if not available_dates:
            return {'success': False, 'error': 'No parquet data available'}
        
        all_bx_prices = []
        nodes_set = set(nodes)
        per_node_totals = {node: {'sum': 0, 'count': 0} for node in nodes}
        per_node_hour_counts = {node: {h: 0 for h in range(1, 25)} for node in nodes}
        
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
                        cheapest_rows = node_df.nsmallest(bx, 'mw')
                        cheapest = cheapest_rows['mw'].mean()
                        all_bx_prices.append(cheapest)
                        per_node_totals[node]['sum'] += cheapest
                        per_node_totals[node]['count'] += 1
                        for hour in cheapest_rows['opr_hr'].values:
                            per_node_hour_counts[node][int(hour)] += 1
            except Exception as e:
                self.logger.debug(f"Error processing {d}: {e}")
                continue
        
        if not all_bx_prices:
            return {'success': False, 'error': 'No data found for selected nodes'}
        
        per_node_averages = {
            node: t['sum'] / t['count'] 
            for node, t in per_node_totals.items() if t['count'] > 0
        }
        
        per_node_bx_hours = {}
        for node, hour_counts in per_node_hour_counts.items():
            if any(hour_counts.values()):
                top_hours = sorted(hour_counts.items(), key=lambda x: -x[1])[:bx]
                per_node_bx_hours[node] = sorted([h for h, c in top_hours if c > 0])
        
        return {
            'success': True,
            'bx_type': bx,
            'avg_price': sum(all_bx_prices) / len(all_bx_prices),
            'min_price': min(all_bx_prices),
            'max_price': max(all_bx_prices),
            'node_count': len(nodes),
            'day_count': len(available_dates),
            'per_node': per_node_averages,
            'per_node_hours': per_node_bx_hours
        }

    def get_full_year_hourly_data(
        self,
        nodes: List[str],
        year: int = 2024
    ) -> List[Dict]:
        """
        Get full year hourly data for selected nodes from parquet.
        
        Uses MotherDuck for fast SQL queries if available (~20s vs ~60s).
        """
        if not nodes:
            return []
        
        if self.motherduck:
            try:
                result = self.motherduck.get_full_year_hourly_data(nodes, year)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"MotherDuck query failed, falling back to parquet: {e}")
        
        available_dates = self.parquet.list_available_dates(year=year)
        if not available_dates:
            return []
        
        nodes_set = set(nodes)
        hourly_data = []
        
        for d in available_dates:
            try:
                table = self.parquet.read_day_from_parquet(d)
                if table is None:
                    continue
                
                df = table.to_pandas()
                node_data = df[df['node'].isin(nodes_set)]
                if node_data.empty:
                    continue
                
                hourly_avg = node_data.groupby('opr_hr')['mw'].mean().reset_index()
                
                for _, row in hourly_avg.iterrows():
                    hourly_data.append({
                        'opr_dt': d,
                        'opr_hr': int(row['opr_hr']),
                        'avg_price': float(row['mw'])
                    })
                    
            except Exception as e:
                self.logger.debug(f"Error processing {d}: {e}")
                continue
        
        return hourly_data

    def get_hourly_averages_for_nodes(self, nodes: List[str], year: int = None) -> List[Dict]:
        """
        Get hourly price averages for a list of nodes from parquet files.
        
        Uses MotherDuck for fast SQL queries if available.
        """
        if not nodes:
            return []
        
        year = year or 2024
        
        if self.motherduck:
            try:
                result = self.motherduck.get_hourly_averages_for_nodes(nodes, year)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"MotherDuck hourly query failed, falling back to parquet: {e}")
        
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
            except Exception as e:
                self.logger.debug(f"Error processing {d}: {e}")
                continue
        
        return [
            {'hour': h, 'avg_price': t['sum'] / t['count']}
            for h, t in sorted(hour_totals.items())
            if t['count'] > 0
        ]

    def get_bx_trend_per_node(
        self,
        bx: int,
        nodes: List[str],
        year: int,
        aggregation: str = 'monthly'
    ) -> Dict[str, List[Dict]]:
        """
        Get BX price trend for each specified node from parquet files.
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
        Get summary statistics (for box plot) for each node using MotherDuck.
        """
        if not nodes:
            return []
        
        try:
            from motherduck_client import get_motherduck_client
            client = get_motherduck_client()
            result = client.get_node_summary_statistics(bx, nodes, year)
            if result:
                return result
        except Exception as e:
            self.logger.warning(f"MotherDuck stats failed: {e}, falling back to parquet")
        
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

    def get_hourly_averages_per_node(
        self,
        nodes: List[str],
        year: int = 2024
    ) -> Dict[str, List[Dict]]:
        """
        Get hourly price averages for each node individually using MotherDuck.
        """
        if not nodes:
            return {}
        
        if self.motherduck:
            try:
                result = self.motherduck.get_hourly_averages_per_node(nodes, year)
                if result:
                    return result
            except Exception as e:
                self.logger.warning(f"MotherDuck per-node hourly query failed: {e}")
        
        return {}
