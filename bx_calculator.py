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

    def get_all_zones_bx_average(
        self,
        bx: int,
        year: int = None,
        time_period: str = "Annual",
        month: int = None
    ) -> Dict[str, Any]:
        """
        Get zone-level BX average for all zones plus overall in one query.
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
                    'success': True,
                    'bx_type': bx,
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

    def get_generator_bx_average(
        self,
        bx: int,
        year: int,
        zone: str = None
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
        
        query = f"""
            SELECT 
                zone,
                node,
                AVG(avg_price) as avg_price,
                MIN(avg_price) as min_price,
                MAX(avg_price) as max_price,
                COUNT(*) as day_count
            FROM generator_bx_summary
            WHERE {' AND '.join(conditions)}
            GROUP BY zone, node
            ORDER BY zone
        """
        
        try:
            results = self._md_query(query)
            
            if not results:
                return {'success': False, 'error': 'No generator data found'}
            
            by_zone = {}
            for row in results:
                zone_name = row['zone']
                by_zone[zone_name] = {
                    'node': row['node'],
                    'avg_price': float(row['avg_price']) if row['avg_price'] else None,
                    'min_price': float(row['min_price']) if row['min_price'] else None,
                    'max_price': float(row['max_price']) if row['max_price'] else None,
                    'day_count': row['day_count'],
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
