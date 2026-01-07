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
        self._node_cache = None
    
    def get_all_nodes(self, limit=None):
        """
        Get all unique node names from the database.
        Used for autocomplete node selection.
        
        Args:
            limit: Maximum number of nodes to return (for performance)
            
        Returns:
            list: List of node names sorted alphabetically
        """
        if self._node_cache is not None:
            return self._node_cache[:limit] if limit else self._node_cache
            
        try:
            query = "SELECT DISTINCT node FROM caiso.lmp_data ORDER BY node"
            if limit:
                query += f" LIMIT {int(limit)}"
            
            results = self.db.execute_query(query)
            self._node_cache = [r['node'] for r in results] if results else []
            return self._node_cache
        except Exception as e:
            self.logger.error(f"Error getting nodes: {str(e)}")
            return []
    
    def search_nodes(self, search_term, limit=50):
        """
        Search nodes by partial name match for autocomplete.
        
        Args:
            search_term: Search string to match against node names
            limit: Maximum results to return
            
        Returns:
            list: Matching node names
        """
        try:
            query = """
                SELECT DISTINCT node 
                FROM caiso.lmp_data 
                WHERE node ILIKE %s
                ORDER BY node
                LIMIT %s
            """
            results = self.db.execute_query(query, (f"%{search_term}%", limit))
            return [r['node'] for r in results] if results else []
        except Exception as e:
            self.logger.error(f"Error searching nodes: {str(e)}")
            return []
    
    def _build_where_conditions(self, start_date=None, end_date=None, nodes=None, exclude_zero=True):
        """
        Build standardized WHERE conditions for analytics queries.
        
        Args:
            start_date: Filter start date (optional)
            end_date: Filter end date (optional) 
            nodes: Single node or list of nodes to filter (optional)
            exclude_zero: Whether to exclude zero/negative prices (default True)
            
        Returns:
            tuple: (conditions_list, params_list)
        """
        conditions = []
        params = []
        
        # Exclude zero/negative prices by default for consistency
        if exclude_zero:
            conditions.append("mw > 0")
        
        # Date range filters
        if start_date:
            conditions.append("opr_dt >= %s")
            params.append(start_date)
        if end_date:
            conditions.append("opr_dt <= %s")
            params.append(end_date)
        
        # Node filters
        if nodes:
            if isinstance(nodes, (list, tuple)) and len(nodes) > 1:
                placeholders = ','.join(['%s'] * len(nodes))
                conditions.append(f"node IN ({placeholders})")
                params.extend(nodes)
            elif isinstance(nodes, (list, tuple)) and len(nodes) == 1:
                conditions.append("node = %s")
                params.append(nodes[0])
            else:
                conditions.append("node = %s")
                params.append(nodes)
        
        return conditions, params
    
    def _build_market_aggregation_wrapper(self, base_query, market_summary=False):
        """
        Wrap a query with market-level aggregation if market_summary=True.
        
        Args:
            base_query: The base SQL query to potentially wrap
            market_summary: Whether to aggregate to market level
            
        Returns:
            str: Original query or market-aggregated version
        """
        if not market_summary:
            return base_query
        
        # For market summaries, we aggregate across all nodes
        # This wrapper handles the common pattern of converting per-node results to market-wide
        market_wrapper = f"""
        WITH base_results AS (
            {base_query}
        )
        SELECT 
            'MARKET_SUMMARY' as node,
            ROUND(AVG(CASE WHEN CAST(peak AS NUMERIC) > 0 THEN CAST(peak AS NUMERIC) END)::numeric, 2) as peak,
            ROUND(AVG(CASE WHEN CAST(off_peak AS NUMERIC) > 0 THEN CAST(off_peak AS NUMERIC) END)::numeric, 2) as off_peak,
            ROUND(AVG(CASE WHEN CAST(peak_premium AS NUMERIC) > 0 THEN CAST(peak_premium AS NUMERIC) END)::numeric, 2) as peak_premium,
            ROUND(AVG(CASE WHEN CAST(peak_premium_pct AS NUMERIC) > 0 THEN CAST(peak_premium_pct AS NUMERIC) END)::numeric, 2) as peak_premium_pct
        FROM base_results
        WHERE peak IS NOT NULL AND off_peak IS NOT NULL
        """
        
        return market_wrapper
    
    @register_analytics(
        description="Find the cheapest operational hours (0-23) averaged across all nodes",
        parameters=["n_hours", "start_date", "end_date", "exclude_zero"],
        example_questions=[
            "What are the 5 cheapest hours of the day?",
            "Show me the top 10 cheapest operational hours between Jan 1-15",
            "Which hours have the lowest average electricity prices?"
        ]
    )
    def get_cheapest_operational_hours(self, n_hours=5, start_date=None, end_date=None, exclude_zero=True):
        """Get the N cheapest operational hours (0-23) averaged across all nodes"""
        try:
            # Use shared helper for consistent condition building
            conditions, params = self._build_where_conditions(
                start_date=start_date, 
                end_date=end_date, 
                exclude_zero=exclude_zero
            )
            
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
        parameters=["n_hours", "during_cheap_hours", "start_date", "end_date", "exclude_zero"],
        example_questions=[
            "What are the 25 hours with lowest congestion?",
            "Show me low congestion hours during cheap price periods",
            "Find times with minimal transmission constraints last month"
        ]
    )
    def get_lowest_congestion_hours(self, n_hours, during_cheap_hours=False, start_date=None, end_date=None, exclude_zero=True):
        """Get hours with lowest congestion component from database"""
        try:
            # Start with MCC constraint and use shared helper for other conditions
            base_conditions, params = self._build_where_conditions(
                start_date=start_date, 
                end_date=end_date, 
                exclude_zero=exclude_zero
            )
            
            # Add MCC-specific constraint
            conditions = ["mcc IS NOT NULL"] + base_conditions
            
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
        parameters=["percentile", "period_start", "period_end", "exclude_zero"],
        example_questions=[
            "Which nodes are in the bottom 10% for prices?",
            "Show me the cheapest 5% of nodes last quarter",
            "Find nodes with consistently low prices in the 20th percentile"
        ]
    )
    def get_nodes_by_price_percentile(self, percentile=10, period_start=None, period_end=None, exclude_zero=True):
        """Get nodes in the lowest X percentile of prices from database"""
        try:
            # Use shared helper for consistent condition building
            conditions, params = self._build_where_conditions(
                start_date=period_start, 
                end_date=period_end, 
                exclude_zero=exclude_zero
            )
            
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
        parameters=["start_date", "end_date", "exclude_zero", "market_summary"],
        example_questions=[
            "Compare peak and off-peak prices for all nodes",
            "What's the peak premium percentage by node?",
            "Show peak vs off-peak price differences last month"
        ]
    )
    def get_peak_vs_offpeak_analysis(self, start_date=None, end_date=None, exclude_zero=True, market_summary=False):
        """Analyze peak vs off-peak pricing patterns from database"""
        try:
            # Use shared helper for consistent condition building
            conditions, params = self._build_where_conditions(
                start_date=start_date, 
                end_date=end_date, 
                exclude_zero=exclude_zero
            )
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            # Build base query for peak vs off-peak analysis
            if market_summary:
                # Market-wide aggregation
                query = f"""
                WITH period_stats AS (
                    SELECT 
                        CASE WHEN opr_hr BETWEEN 0 AND 6 THEN 'Peak' ELSE 'Off-Peak' END as period,
                        AVG(mw) as avg_price
                    FROM caiso.lmp_data 
                    {where_clause}
                    GROUP BY CASE WHEN opr_hr BETWEEN 0 AND 6 THEN 'Peak' ELSE 'Off-Peak' END
                )
                SELECT 
                    'MARKET_SUMMARY' as node,
                    ROUND(MAX(CASE WHEN period = 'Peak' THEN avg_price END)::numeric, 2) as peak,
                    ROUND(MAX(CASE WHEN period = 'Off-Peak' THEN avg_price END)::numeric, 2) as off_peak,
                    ROUND((MAX(CASE WHEN period = 'Peak' THEN avg_price END) - 
                           MAX(CASE WHEN period = 'Off-Peak' THEN avg_price END))::numeric, 2) as peak_premium,
                    ROUND((((MAX(CASE WHEN period = 'Peak' THEN avg_price END) - 
                             MAX(CASE WHEN period = 'Off-Peak' THEN avg_price END)) / 
                             NULLIF(MAX(CASE WHEN period = 'Off-Peak' THEN avg_price END), 0)) * 100)::numeric, 2) as peak_premium_pct
                FROM period_stats
                HAVING COUNT(DISTINCT period) = 2
                """
            else:
                # Per-node breakdown (original behavior)
                query = f"""
                WITH period_stats AS (
                    SELECT 
                        node,
                        CASE WHEN opr_hr BETWEEN 0 AND 6 THEN 'Peak' ELSE 'Off-Peak' END as period,
                        AVG(mw) as avg_price
                    FROM caiso.lmp_data 
                    {where_clause}
                    GROUP BY node, CASE WHEN opr_hr BETWEEN 0 AND 6 THEN 'Peak' ELSE 'Off-Peak' END
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
        parameters=["start_date", "end_date", "exclude_zero", "market_summary"],
        example_questions=[
            "Show me detailed price statistics for all nodes",
            "What are the mean, median, and percentiles by node?",
            "Get statistical overview of prices last week"
        ]
    )
    def get_price_statistics(self, start_date=None, end_date=None, exclude_zero=True, market_summary=False):
        """Get comprehensive price statistics from database"""
        try:
            # Use shared helper for consistent condition building
            conditions, params = self._build_where_conditions(
                start_date=start_date, 
                end_date=end_date, 
                exclude_zero=exclude_zero
            )
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            # Build query based on market_summary preference
            if market_summary:
                # Market-wide statistics aggregation
                query = f"""
                SELECT 
                    'MARKET_SUMMARY' as node,
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
                """
            else:
                # Per-node statistics (original behavior)
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
        parameters=["start_date", "end_date", "exclude_zero", "market_summary"],
        example_questions=[
            "What are the average prices by hour of day?",
            "Show me hourly price patterns last month",
            "Which hours typically have the highest/lowest prices?"
        ]
    )
    def get_hourly_averages(self, start_date=None, end_date=None, exclude_zero=True, market_summary=False):
        """Get average prices by hour of day from database"""
        try:
            # Use shared helper for consistent condition building
            conditions, params = self._build_where_conditions(
                start_date=start_date, 
                end_date=end_date, 
                exclude_zero=exclude_zero
            )
            
            where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
            
            # Market summary doesn't change the structure for hourly averages since it's already aggregated across nodes
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
        parameters=["start_date", "end_date", "exclude_zero"],
        example_questions=[
            "Give me a summary of all nodes and their price ranges",
            "What nodes have data and what are their min/max prices?",
            "Show node overview with data counts and date ranges"
        ]
    )
    def get_node_summary(self, start_date=None, end_date=None, exclude_zero=True):
        """Get summary statistics for all nodes from database"""
        try:
            # Use shared helper for consistent condition building
            conditions, params = self._build_where_conditions(
                start_date=start_date, 
                end_date=end_date, 
                exclude_zero=exclude_zero
            )
            
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
        parameters=["threshold_std", "start_date", "end_date", "exclude_zero"],
        example_questions=[
            "Find all price spikes above 3 standard deviations",
            "What are the unusual price events last week?",
            "Detect price anomalies and spikes in the data"
        ]
    )
    def detect_price_spikes(self, threshold_std=3, start_date=None, end_date=None, exclude_zero=True):
        """Detect price spikes using standard deviation threshold from database"""
        try:
            # Use shared helper for consistent condition building
            conditions, params = self._build_where_conditions(
                start_date=start_date, 
                end_date=end_date, 
                exclude_zero=exclude_zero
            )
            
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
        parameters=["start_date", "end_date", "exclude_zero"],
        example_questions=[
            "Which nodes have the highest congestion?",
            "Show me congestion analysis by node",
            "What are the congestion patterns and frequencies?"
        ]
    )
    def get_congestion_analysis(self, start_date=None, end_date=None, exclude_zero=True):
        """Analyze congestion patterns if MCC data is available from database"""
        try:
            # Start with MCC constraint and use shared helper for other conditions
            base_conditions, params = self._build_where_conditions(
                start_date=start_date, 
                end_date=end_date, 
                exclude_zero=exclude_zero
            )
            
            # Add MCC-specific constraint
            conditions = ["mcc IS NOT NULL"] + base_conditions
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
        parameters=["node", "start_date", "end_date", "aggregation_method", "exclude_zero"],
        example_questions=[
            "Show me a chart of prices at each operational hour at node CSADIAB_7_N001",
            "What are the hourly prices for node CAISO_EHV across all 24 hours?",
            "Get the price pattern by hour for a specific node"
        ]
    )
    def get_node_hourly_prices(self, node, start_date=None, end_date=None, aggregation_method="avg", exclude_zero=True):
        """Get prices for a specific node across all operational hours (0-23)"""
        try:
            # Use shared helper for consistent condition building, then add node constraint
            base_conditions, base_params = self._build_where_conditions(
                start_date=start_date, 
                end_date=end_date, 
                exclude_zero=exclude_zero
            )
            
            # Add node-specific constraint
            conditions = ["node = %s"] + base_conditions
            params = [node] + base_params
            
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
