import pandas as pd
from datetime import datetime, timedelta
from database import DatabaseManager
import logging
import inspect
from functools import wraps

# Module-level registry for analytics methods
_analytics_registry = {}

def register_analytics(description, parameters=None, example_questions=None):
    """
    Decorator to register analytics methods with metadata for dynamic discovery.
    
    Args:
        description (str): Clear description of what the method does
        parameters (list): List of expected parameter names
        example_questions (list): Example natural language questions this method can answer
    
    Returns:
        Decorated method with registration metadata
    """
    def decorator(func):
        # Get method signature for additional metadata
        sig = inspect.signature(func)
        param_names = [param for param in sig.parameters.keys() if param != 'self']
        
        # Store registration metadata
        _analytics_registry[func.__name__] = {
            'method_name': func.__name__,
            'description': description,
            'parameters': parameters or param_names,
            'example_questions': example_questions or [],
            'signature': str(sig)
        }
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        return wrapper
    return decorator

def get_registered_analytics():
    """
    Get all registered analytics methods with their metadata.
    
    Returns:
        dict: Dictionary of all registered methods with metadata for chatbot discovery
    """
    return _analytics_registry.copy()

class LMPAnalytics:
    """Core analytics functions for CAISO LMP data analysis with PostgreSQL backend"""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.logger = logging.getLogger(__name__)
    
    @register_analytics(
        description="Find the cheapest operational hours (0-23) averaged across all nodes",
        parameters=["n_hours", "start_date", "end_date"],
        example_questions=[
            "What are the 5 cheapest hours of the day?",
            "Show me the top 10 cheapest operational hours between Jan 1-15",
            "Which hours have the lowest average electricity prices?"
        ]
    )
    def get_cheapest_operational_hours(self, n_hours=5, start_date=None, end_date=None):
        """Get the N cheapest operational hours (0-23) averaged across all nodes"""
        try:
            conditions = ["mw > 0"]
            params = []
            
            if start_date:
                conditions.append("opr_dt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("opr_dt <= %s")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            query = f"""
            SELECT 
                opr_hr,
                ROUND(AVG(mw)::numeric, 2) as avg_price,
                COUNT(*) as records,
                COUNT(DISTINCT node) as unique_nodes
            FROM caiso.lmp_data 
            {where_clause}
            GROUP BY opr_hr
            ORDER BY avg_price ASC
            LIMIT %s
            """
            params.append(n_hours)
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting cheapest operational hours: {str(e)}")
            return pd.DataFrame()

    @register_analytics(
        description="Get the cheapest specific hours overall or for specific node(s)",
        parameters=["n_hours", "node", "aggregate_nodes", "start_date", "end_date", "exclude_zero"],
        example_questions=[
            "What were the 20 cheapest hours last week?",
            "Show me the cheapest 50 hours for node CAISO_EHV",
            "Find the cheapest hours across multiple nodes in January"
        ]
    )
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
                    conditions.append("opr_dt >= %s")
                    params.append(start_date)
                if end_date:
                    conditions.append("opr_dt <= %s") 
                    params.append(end_date)
                
                where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                
                query = f"""
                WITH aggregated_prices AS (
                    SELECT 
                        opr_dt as operational_date,
                        opr_hr as operational_hour,
                        AVG(mw) as mw,
                        'Aggregated_{len(aggregate_nodes)}_nodes' as node
                    FROM caiso.lmp_data 
                    {where_clause}
                    GROUP BY opr_dt, opr_hr
                )
                SELECT operational_date, operational_hour, node, ROUND(mw::numeric, 2) as mw
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

    @register_analytics(
        description="Find the cheapest nodes during a specific operational date and hour",
        parameters=["n_nodes", "operational_date", "operational_hour", "exclude_zero"],
        example_questions=[
            "What were the 10 cheapest nodes on Jan 1 at hour 13?",
            "Show me the cheapest nodes during peak hours on Dec 25",
            "Which nodes had the lowest prices yesterday at 3 PM?"
        ]
    )
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
                conditions.append("opr_dt = %s")
                params.append(operational_date)
            
            # Filter by operational hour
            if operational_hour is not None:
                conditions.append("opr_hr = %s")
                params.append(operational_hour)
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            query = f"""
            SELECT 
                opr_dt as operational_date,
                node,
                ROUND(mw::numeric, 2) as mw,
                opr_hr
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
    
    @register_analytics(
        description="Get hours with the lowest congestion component (MCC)",
        parameters=["n_hours", "during_cheap_hours", "start_date", "end_date"],
        example_questions=[
            "What are the 25 hours with lowest congestion?",
            "Show me low congestion hours during cheap price periods",
            "Find times with minimal transmission constraints last month"
        ]
    )
    def get_lowest_congestion_hours(self, n_hours, during_cheap_hours=False, start_date=None, end_date=None):
        """Get hours with lowest congestion component from database"""
        try:
            conditions = ["mcc IS NOT NULL"]
            params = []
            
            # Add date filters
            if start_date:
                conditions.append("opr_dt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("opr_dt <= %s")
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
                opr_dt as operational_date,
                opr_hr as operational_hour,
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
    
    @register_analytics(
        description="Get nodes in the lowest X percentile of average prices",
        parameters=["percentile", "period_start", "period_end"],
        example_questions=[
            "Which nodes are in the bottom 10% for prices?",
            "Show me the cheapest 5% of nodes last quarter",
            "Find nodes with consistently low prices in the 20th percentile"
        ]
    )
    def get_nodes_by_price_percentile(self, percentile=10, period_start=None, period_end=None):
        """Get nodes in the lowest X percentile of prices from database"""
        try:
            conditions = []
            params = []
            
            if period_start:
                conditions.append("opr_dt >= %s")
                params.append(period_start)
            if period_end:
                conditions.append("opr_dt <= %s")
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
    
    @register_analytics(
        description="Analyze peak vs off-peak pricing patterns and premiums",
        parameters=["start_date", "end_date"],
        example_questions=[
            "Compare peak and off-peak prices for all nodes",
            "What's the peak premium percentage by node?",
            "Show peak vs off-peak price differences last month"
        ]
    )
    def get_peak_vs_offpeak_analysis(self, start_date=None, end_date=None):
        """Analyze peak vs off-peak pricing patterns from database"""
        try:
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
            WITH period_stats AS (
                SELECT 
                    node,
                    CASE WHEN opr_hr BETWEEN 7 AND 22 THEN 'Peak' ELSE 'Off-Peak' END as period,
                    AVG(mw) as avg_price
                FROM caiso.lmp_data 
                {where_clause}
                GROUP BY node, CASE WHEN opr_hr BETWEEN 7 AND 22 THEN 'Peak' ELSE 'Off-Peak' END
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
    
    @register_analytics(
        description="Get comprehensive statistical summary of prices by node",
        parameters=["start_date", "end_date"],
        example_questions=[
            "Show me detailed price statistics for all nodes",
            "What are the mean, median, and percentiles by node?",
            "Get statistical overview of prices last week"
        ]
    )
    def get_price_statistics(self, start_date=None, end_date=None):
        """Get comprehensive price statistics from database"""
        try:
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
    
    @register_analytics(
        description="Get average prices by hour of day across all nodes",
        parameters=["start_date", "end_date"],
        example_questions=[
            "What are the average prices by hour of day?",
            "Show me hourly price patterns last month",
            "Which hours typically have the highest/lowest prices?"
        ]
    )
    def get_hourly_averages(self, start_date=None, end_date=None):
        """Get average prices by hour of day from database"""
        try:
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
                opr_hr as hour,
                ROUND(AVG(mw)::numeric, 2) as mw,
                ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mw)::numeric, 2) as median,
                ROUND(STDDEV(mw)::numeric, 2) as std
            FROM caiso.lmp_data 
            {where_clause}
            GROUP BY opr_hr
            ORDER BY opr_hr
            """
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting hourly averages: {str(e)}")
            return pd.DataFrame()
    
    @register_analytics(
        description="Get summary statistics and data coverage for each node",
        parameters=["start_date", "end_date"],
        example_questions=[
            "Give me a summary of all nodes and their price ranges",
            "What nodes have data and what are their min/max prices?",
            "Show node overview with data counts and date ranges"
        ]
    )
    def get_node_summary(self, start_date=None, end_date=None):
        """Get summary statistics for all nodes from database"""
        try:
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
                node,
                COUNT(*) as mw_count,
                ROUND(AVG(mw)::numeric, 2) as mw_mean,
                ROUND(MIN(mw)::numeric, 2) as mw_min,
                ROUND(MAX(mw)::numeric, 2) as mw_max,
                MIN(opr_dt) as opr_dt_min,
                MAX(opr_dt) as opr_dt_max
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
    
    @register_analytics(
        description="Detect price spikes using rolling standard deviation threshold",
        parameters=["threshold_std", "start_date", "end_date"],
        example_questions=[
            "Find all price spikes above 3 standard deviations",
            "What are the unusual price events last week?",
            "Detect price anomalies and spikes in the data"
        ]
    )
    def detect_price_spikes(self, threshold_std=3, start_date=None, end_date=None):
        """Detect price spikes using standard deviation threshold from database"""
        try:
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
            WITH rolling_stats AS (
                SELECT 
                    opr_dt as operational_date,
                    opr_hr as operational_hour,
                    node,
                    mw,
                    AVG(mw) OVER (
                        PARTITION BY node 
                        ORDER BY opr_dt, opr_hr 
                        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                    ) as rolling_mean,
                    STDDEV(mw) OVER (
                        PARTITION BY node 
                        ORDER BY opr_dt, opr_hr 
                        ROWS BETWEEN 23 PRECEDING AND CURRENT ROW
                    ) as rolling_std
                FROM caiso.lmp_data 
                {where_clause}
            )
            SELECT 
                operational_date,
                operational_hour,
                node,
                ROUND(mw::numeric, 2) as mw,
                ROUND(rolling_mean::numeric, 2) as rolling_mean
            FROM rolling_stats
            WHERE mw > rolling_mean + %s * COALESCE(rolling_std, 0)
              AND rolling_std IS NOT NULL
              AND rolling_std > 0
            ORDER BY operational_date DESC, operational_hour DESC
            """
            params.append(threshold_std)
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error detecting price spikes: {str(e)}")
            return pd.DataFrame()
    
    @register_analytics(
        description="Analyze congestion patterns using MCC (marginal congestion component) data",
        parameters=["start_date", "end_date"],
        example_questions=[
            "Which nodes have the highest congestion?",
            "Show me congestion analysis by node",
            "What are the congestion patterns and frequencies?"
        ]
    )
    def get_congestion_analysis(self, start_date=None, end_date=None):
        """Analyze congestion patterns if MCC data is available from database"""
        try:
            conditions = ["mcc IS NOT NULL"]
            params = []
            
            if start_date:
                conditions.append("opr_dt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("opr_dt <= %s")
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
    
    @register_analytics(
        description="Get prices at each operational hour (0-23) for a specific node",
        parameters=["node", "start_date", "end_date", "aggregation_method"],
        example_questions=[
            "Show me a chart of prices at each operational hour at node CSADIAB_7_N001",
            "What are the hourly prices for node CAISO_EHV across all 24 hours?",
            "Get the price pattern by hour for a specific node"
        ]
    )
    def get_node_hourly_prices(self, node, start_date=None, end_date=None, aggregation_method="avg"):
        """Get prices for a specific node across all operational hours (0-23)"""
        try:
            conditions = ["node = %s"]
            params = [node]
            
            if start_date:
                conditions.append("opr_dt >= %s")
                params.append(start_date)
            if end_date:
                conditions.append("opr_dt <= %s")
                params.append(end_date)
            
            where_clause = "WHERE " + " AND ".join(conditions)
            
            # Choose aggregation method
            agg_func = "AVG" if aggregation_method.lower() == "avg" else "MEDIAN"
            if aggregation_method.lower() == "median":
                agg_func = "PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY mw)"
            
            query = f"""
            SELECT 
                opr_hr as operational_hour,
                node,
                ROUND({agg_func}(mw)::numeric, 2) as price,
                COUNT(*) as data_points,
                ROUND(MIN(mw)::numeric, 2) as min_price,
                ROUND(MAX(mw)::numeric, 2) as max_price,
                ROUND(STDDEV(mw)::numeric, 2) as price_std
            FROM caiso.lmp_data 
            {where_clause}
            GROUP BY opr_hr, node
            ORDER BY opr_hr
            """
            
            results = self.db.execute_query(query, params)
            return pd.DataFrame(results) if results else pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error getting node hourly prices: {str(e)}")
            return pd.DataFrame()
