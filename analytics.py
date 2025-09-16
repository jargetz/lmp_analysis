import pandas as pd
from datetime import datetime, timedelta
from database import DatabaseManager
import logging

class LMPAnalytics:
    """Core analytics functions for CAISO LMP data analysis with PostgreSQL backend"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.logger = logging.getLogger(__name__)
    
    def get_cheapest_operational_hours(self, n_hours=5, start_date=None, end_date=None):
        """Get the N cheapest operational hours (0-23) averaged across all nodes"""
        try:
            # Use hour_of_day as temporary fallback until opr_hr is properly loaded
            conditions = ["mw > 0"]
            params = []
            
            if start_date:
                conditions.append("date_only >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("date_only <= %s")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            query = f"""
            SELECT 
                hour_of_day as opr_hr,
                ROUND(AVG(mw)::numeric, 2) as avg_price,
                COUNT(*) as records,
                COUNT(DISTINCT node) as unique_nodes
            FROM caiso.lmp_data 
            {where_clause}
            GROUP BY hour_of_day
            ORDER BY avg_price ASC
            LIMIT %s
            """
            params.append(n_hours)
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting cheapest operational hours: {str(e)}")
            return pd.DataFrame()

    def get_cheapest_hours(self, n_hours, node=None, aggregate_nodes=None, start_date=None, end_date=None, exclude_zero=True):
        """Get the N cheapest hours overall or for specific node(s) from database"""
        try:
            if aggregate_nodes and len(aggregate_nodes) > 1:
                # Query for aggregated nodes
                placeholders = ','.join(['%s'] * len(aggregate_nodes))
                conditions = [f"node IN ({placeholders})"]
                params = list(aggregate_nodes)
                
                # Exclude zero prices by default
                if exclude_zero:
                    conditions.append("mw > 0")
                
                if start_date:
                    conditions.append("interval_start_time_gmt >= %s")
                    params.append(start_date)
                if end_date:
                    conditions.append("interval_start_time_gmt <= %s") 
                    params.append(end_date)
                
                where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                
                query = f"""
                WITH aggregated_prices AS (
                    SELECT 
                        interval_start_time_gmt,
                        AVG(mw) as mw,
                        'Aggregated_{len(aggregate_nodes)}_nodes' as node
                    FROM caiso.lmp_data 
                    {where_clause}
                    GROUP BY interval_start_time_gmt
                )
                SELECT interval_start_time_gmt, node, ROUND(mw::numeric, 2) as mw
                FROM aggregated_prices
                ORDER BY mw ASC
                LIMIT %s
                """
                params.append(n_hours)
                
            else:
                # Query for single node or all nodes
                conditions = []
                params = []
                
                # Exclude zero prices by default
                if exclude_zero:
                    conditions.append("mw > 0")
                
                if node:
                    conditions.append("node = %s")
                    params.append(node)
                elif aggregate_nodes and len(aggregate_nodes) == 1:
                    conditions.append("node = %s")
                    params.append(aggregate_nodes[0])
                    
                if start_date:
                    conditions.append("interval_start_time_gmt >= %s")
                    params.append(start_date)
                if end_date:
                    conditions.append("interval_start_time_gmt <= %s")
                    params.append(end_date)
                
                where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                
                query = f"""
                SELECT 
                    interval_start_time_gmt,
                    node,
                    ROUND(mw::numeric, 2) as mw
                FROM caiso.lmp_data 
                {where_clause}
                ORDER BY mw ASC
                LIMIT %s
                """
                params.append(n_hours)
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting cheapest hours: {str(e)}")
            return pd.DataFrame()

    def get_cheapest_nodes_by_hour(self, n_nodes=10, operational_date=None, operational_hour=None, exclude_zero=True):
        """Get the N cheapest nodes for a specific operational date and hour"""
        try:
            conditions = []
            params = []
            
            # Always exclude zero prices by default unless specified otherwise
            if exclude_zero:
                conditions.append("mw > 0")
            
            # Filter by operational date
            if operational_date:
                conditions.append("date_only = %s")
                params.append(operational_date)
            
            # Filter by operational hour (using hour_of_day column as proxy for opr_hr)
            if operational_hour is not None:
                conditions.append("hour_of_day = %s")
                params.append(operational_hour)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            SELECT 
                date_only as operational_date,
                node,
                ROUND(mw::numeric, 2) as mw,
                hour_of_day as opr_hr
            FROM caiso.lmp_data 
            {where_clause}
            ORDER BY mw ASC
            LIMIT %s
            """
            params.append(n_nodes)
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting cheapest nodes by hour: {str(e)}")
            return pd.DataFrame()
    
    def get_lowest_congestion_hours(self, n_hours, during_cheap_hours=False, start_date=None, end_date=None):
        """Get hours with lowest congestion component from database"""
        try:
            conditions = ["mcc IS NOT NULL"]
            params = []
            
            # Add date filters
            if start_date:
                conditions.append("interval_start_time_gmt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("interval_start_time_gmt <= %s")
                params.append(end_date)
            
            if during_cheap_hours:
                # First get the 20% price threshold
                threshold_query = f"""
                SELECT PERCENTILE_CONT(0.2) WITHIN GROUP (ORDER BY mw) as cheap_threshold
                FROM caiso.lmp_data 
                WHERE {' AND '.join(conditions)}
                """
                threshold_result = self.db.execute_query(threshold_query, params, fetch_all=False)
                
                if threshold_result and isinstance(threshold_result, dict) and threshold_result.get('cheap_threshold'):
                    conditions.append("mw <= %s")
                    params.append(float(threshold_result['cheap_threshold']))
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            query = f"""
            SELECT 
                interval_start_time_gmt,
                node,
                ROUND(mw::numeric, 2) as mw,
                ROUND(mcc::numeric, 2) as mcc
            FROM caiso.lmp_data 
            {where_clause}
            ORDER BY mcc ASC
            LIMIT %s
            """
            params.append(n_hours)
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting lowest congestion hours: {str(e)}")
            return pd.DataFrame()
    
    def get_nodes_by_price_percentile(self, percentile=10, period_start=None, period_end=None):
        """Get nodes in the lowest X percentile of prices from database"""
        try:
            conditions = []
            params = []
            
            if period_start:
                conditions.append("interval_start_time_gmt >= %s")
                params.append(period_start)
            if period_end:
                conditions.append("interval_start_time_gmt <= %s")
                params.append(period_end)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            WITH node_stats AS (
                SELECT 
                    node,
                    AVG(mw) as avg_price,
                    STDDEV(mw) as price_std,
                    COUNT(*) as data_points
                FROM caiso.lmp_data 
                {where_clause}
                GROUP BY node
            ),
            price_threshold AS (
                SELECT PERCENTILE_CONT(%s / 100.0) WITHIN GROUP (ORDER BY avg_price) as threshold
                FROM node_stats
            )
            SELECT 
                n.node,
                ROUND(n.avg_price::numeric, 2) as avg_price,
                ROUND(n.price_std::numeric, 2) as price_std,
                n.data_points
            FROM node_stats n, price_threshold t
            WHERE n.avg_price <= t.threshold
            ORDER BY n.avg_price
            """
            params.append(percentile)
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting nodes by price percentile: {str(e)}")
            return pd.DataFrame()
    
    def get_peak_vs_offpeak_analysis(self, start_date=None, end_date=None):
        """Analyze peak vs off-peak pricing patterns from database"""
        try:
            conditions = []
            params = []
            
            if start_date:
                conditions.append("interval_start_time_gmt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("interval_start_time_gmt <= %s")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            WITH period_stats AS (
                SELECT 
                    node,
                    CASE WHEN hour_of_day BETWEEN 7 AND 22 THEN 'Peak' ELSE 'Off-Peak' END as period,
                    AVG(mw) as avg_price
                FROM caiso.lmp_data 
                {where_clause}
                GROUP BY node, CASE WHEN hour_of_day BETWEEN 7 AND 22 THEN 'Peak' ELSE 'Off-Peak' END
            )
            SELECT 
                node,
                ROUND(MAX(CASE WHEN period = 'Peak' THEN avg_price END)::numeric, 2) as peak,
                ROUND(MAX(CASE WHEN period = 'Off-Peak' THEN avg_price END)::numeric, 2) as off_peak,
                ROUND((MAX(CASE WHEN period = 'Peak' THEN avg_price END) - 
                       MAX(CASE WHEN period = 'Off-Peak' THEN avg_price END))::numeric, 2) as peak_premium,
                ROUND((((MAX(CASE WHEN period = 'Peak' THEN avg_price END) - 
                         MAX(CASE WHEN period = 'Off-Peak' THEN avg_price END)) / 
                         NULLIF(MAX(CASE WHEN period = 'Off-Peak' THEN avg_price END), 0)) * 100)::numeric, 2) as peak_premium_pct
            FROM period_stats
            GROUP BY node
            HAVING COUNT(DISTINCT period) = 2
            ORDER BY peak_premium_pct DESC
            """
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting peak vs off-peak analysis: {str(e)}")
            return pd.DataFrame()
    
    def get_price_statistics(self, start_date=None, end_date=None):
        """Get comprehensive price statistics from database"""
        try:
            conditions = []
            params = []
            
            if start_date:
                conditions.append("interval_start_time_gmt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("interval_start_time_gmt <= %s")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            SELECT 
                node,
                COUNT(*) as count,
                ROUND(AVG(mw)::numeric, 2) as mean,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mw)::numeric, 2) as median,
                ROUND(STDDEV(mw)::numeric, 2) as std,
                ROUND(MIN(mw)::numeric, 2) as min,
                ROUND(MAX(mw)::numeric, 2) as max,
                ROUND(PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY mw)::numeric, 2) as p25,
                ROUND(PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY mw)::numeric, 2) as p75,
                ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY mw)::numeric, 2) as p95
            FROM caiso.lmp_data 
            {where_clause}
            GROUP BY node
            ORDER BY mean
            """
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting price statistics: {str(e)}")
            return pd.DataFrame()
    
    def get_hourly_averages(self, start_date=None, end_date=None):
        """Get average prices by hour of day from database"""
        try:
            conditions = []
            params = []
            
            if start_date:
                conditions.append("interval_start_time_gmt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("interval_start_time_gmt <= %s")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            SELECT 
                hour_of_day as hour,
                ROUND(AVG(mw)::numeric, 2) as mw,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mw)::numeric, 2) as median,
                ROUND(STDDEV(mw)::numeric, 2) as std
            FROM caiso.lmp_data 
            {where_clause}
            GROUP BY hour_of_day
            ORDER BY hour_of_day
            """
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting hourly averages: {str(e)}")
            return pd.DataFrame()
    
    def get_node_summary(self, start_date=None, end_date=None):
        """Get summary statistics for all nodes from database"""
        try:
            conditions = []
            params = []
            
            if start_date:
                conditions.append("interval_start_time_gmt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("interval_start_time_gmt <= %s")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            SELECT 
                node,
                COUNT(*) as mw_count,
                ROUND(AVG(mw)::numeric, 2) as mw_mean,
                ROUND(MIN(mw)::numeric, 2) as mw_min,
                ROUND(MAX(mw)::numeric, 2) as mw_max,
                MIN(interval_start_time_gmt) as intervalstarttime_gmt_min,
                MAX(interval_start_time_gmt) as intervalstarttime_gmt_max
            FROM caiso.lmp_data 
            {where_clause}
            GROUP BY node
            ORDER BY node
            """
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting node summary: {str(e)}")
            return pd.DataFrame()
    
    def detect_price_spikes(self, threshold_std=3, start_date=None, end_date=None):
        """Detect price spikes using standard deviation threshold from database"""
        try:
            conditions = []
            params = []
            
            if start_date:
                conditions.append("interval_start_time_gmt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("interval_start_time_gmt <= %s")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            WITH rolling_stats AS (
                SELECT 
                    interval_start_time_gmt,
                    node,
                    mw,
                    AVG(mw) OVER (
                        PARTITION BY node 
                        ORDER BY interval_start_time_gmt 
                        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                    ) as rolling_mean,
                    STDDEV(mw) OVER (
                        PARTITION BY node 
                        ORDER BY interval_start_time_gmt 
                        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                    ) as rolling_std
                FROM caiso.lmp_data 
                {where_clause}
            )
            SELECT 
                interval_start_time_gmt,
                node,
                ROUND(mw::numeric, 2) as mw,
                ROUND(rolling_mean::numeric, 2) as rolling_mean
            FROM rolling_stats
            WHERE mw > rolling_mean + %s * COALESCE(rolling_std, 0)
              AND rolling_std IS NOT NULL
              AND rolling_std > 0
            ORDER BY interval_start_time_gmt DESC
            """
            params.append(threshold_std)
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error detecting price spikes: {str(e)}")
            return pd.DataFrame()
    
    def get_congestion_analysis(self, start_date=None, end_date=None):
        """Analyze congestion patterns if MCC data is available from database"""
        try:
            conditions = ["mcc IS NOT NULL"]
            params = []
            
            if start_date:
                conditions.append("interval_start_time_gmt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("interval_start_time_gmt <= %s")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            query = f"""
            SELECT 
                node,
                ROUND(AVG(mcc)::numeric, 2) as avg_congestion,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mcc)::numeric, 2) as median_congestion,
                ROUND(STDDEV(mcc)::numeric, 2) as congestion_std,
                ROUND(MIN(mcc)::numeric, 2) as min_congestion,
                ROUND(MAX(mcc)::numeric, 2) as max_congestion,
                COUNT(CASE WHEN mcc > 0 THEN 1 END) as positive_congestion_hours,
                COUNT(*) as total_hours,
                ROUND((COUNT(CASE WHEN mcc > 0 THEN 1 END) * 100.0 / COUNT(*))::numeric, 2) as congestion_frequency
            FROM caiso.lmp_data 
            {where_clause}
            GROUP BY node
            ORDER BY congestion_frequency DESC
            """
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting congestion analysis: {str(e)}")
            return pd.DataFrame()
