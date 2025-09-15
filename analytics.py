import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class LMPAnalytics:
    """Core analytics functions for CAISO LMP data analysis"""
    
    def __init__(self):
        pass
    
    def get_cheapest_hours(self, df, n_hours, node=None, aggregate_nodes=None):
        """Get the N cheapest hours overall or for specific node(s)"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        working_df = df.copy()
        
        if node:
            working_df = working_df[working_df['NODE'] == node]
        elif aggregate_nodes:
            working_df = working_df[working_df['NODE'].isin(aggregate_nodes)]
            # Group by time and calculate mean price across selected nodes
            working_df = working_df.groupby('INTERVALSTARTTIME_GMT').agg({
                'MW': 'mean',
                'NODE': lambda x: f"Aggregated_{len(aggregate_nodes)}_nodes"
            }).reset_index()
        
        if working_df.empty:
            return pd.DataFrame()
        
        # Sort by price and return top N
        cheapest = working_df.nsmallest(n_hours, 'MW')
        
        return cheapest[['INTERVALSTARTTIME_GMT', 'NODE', 'MW']].round(2)
    
    def get_lowest_congestion_hours(self, df, n_hours, during_cheap_hours=False):
        """Get hours with lowest congestion component"""
        if df is None or df.empty or 'MCC' not in df.columns:
            return pd.DataFrame()
        
        working_df = df.copy()
        
        if during_cheap_hours:
            # First find the cheapest 20% of hours
            cheap_threshold = working_df['MW'].quantile(0.2)
            working_df = working_df[working_df['MW'] <= cheap_threshold]
        
        if working_df.empty:
            return pd.DataFrame()
        
        # Sort by congestion component (MCC)
        lowest_congestion = working_df.nsmallest(n_hours, 'MCC')
        
        columns = ['INTERVALSTARTTIME_GMT', 'NODE', 'MW', 'MCC']
        return lowest_congestion[columns].round(2)
    
    def get_nodes_by_price_percentile(self, df, percentile=10, period_start=None, period_end=None):
        """Get nodes in the lowest X percentile of prices"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        working_df = df.copy()
        
        # Filter by date range if provided
        if period_start and 'INTERVALSTARTTIME_GMT' in working_df.columns:
            working_df = working_df[working_df['INTERVALSTARTTIME_GMT'] >= period_start]
        if period_end and 'INTERVALSTARTTIME_GMT' in working_df.columns:
            working_df = working_df[working_df['INTERVALSTARTTIME_GMT'] <= period_end]
        
        # Calculate average price per node
        node_avg_prices = working_df.groupby('NODE')['MW'].agg(['mean', 'std', 'count']).reset_index()
        node_avg_prices.columns = ['NODE', 'avg_price', 'price_std', 'data_points']
        
        # Calculate percentile threshold
        threshold = np.percentile(node_avg_prices['avg_price'], percentile)
        
        # Filter nodes below threshold
        low_price_nodes = node_avg_prices[node_avg_prices['avg_price'] <= threshold]
        low_price_nodes = low_price_nodes.sort_values('avg_price')
        
        return low_price_nodes.round(2)
    
    def get_peak_vs_offpeak_analysis(self, df):
        """Analyze peak vs off-peak pricing patterns"""
        if df is None or df.empty or 'HOUR' not in df.columns:
            return pd.DataFrame()
        
        # Define peak hours (typically 7 AM to 10 PM)
        df_copy = df.copy()
        df_copy['PERIOD'] = df_copy['HOUR'].apply(
            lambda x: 'Peak' if 7 <= x <= 22 else 'Off-Peak'
        )
        
        # Calculate statistics by period
        period_stats = df_copy.groupby(['NODE', 'PERIOD'])['MW'].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ]).reset_index()
        
        # Pivot to compare peak vs off-peak
        pivot_stats = period_stats.pivot(index='NODE', columns='PERIOD', values='mean').reset_index()
        
        if 'Peak' in pivot_stats.columns and 'Off-Peak' in pivot_stats.columns:
            pivot_stats['Peak_Premium'] = pivot_stats['Peak'] - pivot_stats['Off-Peak']
            pivot_stats['Peak_Premium_Pct'] = (pivot_stats['Peak_Premium'] / pivot_stats['Off-Peak']) * 100
        
        return pivot_stats.round(2)
    
    def get_price_statistics(self, df):
        """Get comprehensive price statistics"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        stats = df.groupby('NODE')['MW'].agg([
            'count', 'mean', 'median', 'std', 'min', 'max',
            lambda x: np.percentile(x, 25),
            lambda x: np.percentile(x, 75),
            lambda x: np.percentile(x, 95)
        ]).reset_index()
        
        stats.columns = ['NODE', 'count', 'mean', 'median', 'std', 'min', 'max', 'p25', 'p75', 'p95']
        
        return stats.round(2)
    
    def get_hourly_averages(self, df):
        """Get average prices by hour of day"""
        if df is None or df.empty or 'HOUR' not in df.columns:
            return pd.DataFrame()
        
        hourly_avg = df.groupby('HOUR')['MW'].agg(['mean', 'median', 'std']).reset_index()
        hourly_avg.columns = ['HOUR', 'MW', 'median', 'std']
        
        return hourly_avg.round(2)
    
    def get_node_summary(self, df):
        """Get summary statistics for all nodes"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        summary = df.groupby('NODE').agg({
            'MW': ['count', 'mean', 'min', 'max'],
            'INTERVALSTARTTIME_GMT': ['min', 'max'] if 'INTERVALSTARTTIME_GMT' in df.columns else ['count', 'count']
        }).round(2)
        
        # Flatten column names
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.reset_index()
        
        return summary
    
    def detect_price_spikes(self, df, threshold_std=3):
        """Detect price spikes using standard deviation threshold"""
        if df is None or df.empty:
            return pd.DataFrame()
        
        df_copy = df.copy()
        
        # Calculate rolling statistics for each node
        df_copy['rolling_mean'] = df_copy.groupby('NODE')['MW'].transform(
            lambda x: x.rolling(window=24, min_periods=1).mean()
        )
        df_copy['rolling_std'] = df_copy.groupby('NODE')['MW'].transform(
            lambda x: x.rolling(window=24, min_periods=1).std()
        )
        
        # Identify spikes
        df_copy['is_spike'] = (
            df_copy['MW'] > df_copy['rolling_mean'] + threshold_std * df_copy['rolling_std']
        )
        
        spikes = df_copy[df_copy['is_spike']]
        
        if spikes.empty:
            return pd.DataFrame()
        
        return spikes[['INTERVALSTARTTIME_GMT', 'NODE', 'MW', 'rolling_mean']].round(2)
    
    def get_congestion_analysis(self, df):
        """Analyze congestion patterns if MCC data is available"""
        if df is None or df.empty or 'MCC' not in df.columns:
            return pd.DataFrame()
        
        congestion_stats = df.groupby('NODE')['MCC'].agg([
            'mean', 'median', 'std', 'min', 'max',
            lambda x: (x > 0).sum(),  # Hours with positive congestion
            'count'
        ]).reset_index()
        
        congestion_stats.columns = [
            'NODE', 'avg_congestion', 'median_congestion', 'congestion_std',
            'min_congestion', 'max_congestion', 'positive_congestion_hours', 'total_hours'
        ]
        
        congestion_stats['congestion_frequency'] = (
            congestion_stats['positive_congestion_hours'] / congestion_stats['total_hours']
        ) * 100
        
        return congestion_stats.round(2)
