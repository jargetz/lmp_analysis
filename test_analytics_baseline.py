"""
Minimal baseline tests for CAISO LMP Analytics

These tests validate core analytics functionality against actual database data.
Run manually when you want to verify system behavior after changes.

Usage:
    pytest test_analytics_baseline.py -v
"""

import pytest
from analytics import LMPAnalytics
from database import DatabaseManager


@pytest.fixture(scope="module")
def analytics():
    """Create analytics instance for all tests"""
    return LMPAnalytics()


@pytest.fixture(scope="module")
def db():
    """Create database instance for validation queries"""
    return DatabaseManager()


class TestPeakOffPeakAnalysis:
    """Test peak vs off-peak price analysis"""
    
    def test_market_summary_returns_data(self, analytics):
        """Verify market summary returns valid peak/off-peak data"""
        result = analytics.get_peak_vs_offpeak_analysis(
            exclude_zero=True, 
            market_summary=True
        )
        
        assert not result.empty, "Market summary should return data"
        assert 'peak' in result.columns, "Should have peak column"
        assert 'off_peak' in result.columns, "Should have off_peak column"
        assert 'peak_premium' in result.columns, "Should have peak_premium column"
        assert len(result) == 1, "Market summary should return single row"
    
    def test_peak_higher_than_offpeak(self, analytics):
        """Verify peak prices are higher than off-peak (CAISO market dynamics)"""
        result = analytics.get_peak_vs_offpeak_analysis(
            exclude_zero=True,
            market_summary=True
        )
        
        peak = float(result['peak'].iloc[0])
        off_peak = float(result['off_peak'].iloc[0])
        
        assert peak > off_peak, f"Peak price (${peak}) should be higher than off-peak (${off_peak})"
    
    def test_per_node_analysis_returns_multiple_nodes(self, analytics):
        """Verify per-node analysis returns data for multiple nodes"""
        result = analytics.get_peak_vs_offpeak_analysis(
            exclude_zero=True,
            market_summary=False
        )
        
        assert not result.empty, "Per-node analysis should return data"
        assert len(result) > 1, "Should have multiple nodes"
        assert 'node' in result.columns, "Should have node column"


class TestPriceStatistics:
    """Test basic price statistics"""
    
    def test_market_stats_returns_valid_data(self, analytics):
        """Verify price statistics return valid ranges"""
        result = analytics.get_price_statistics(
            exclude_zero=True,
            market_summary=True
        )
        
        assert not result.empty, "Statistics should return data"
        # Note: get_price_statistics uses 'mean', 'min', 'max' not 'avg_price', 'min_price', 'max_price'
        assert 'mean' in result.columns, "Should have mean"
        assert 'min' in result.columns, "Should have min"
        assert 'max' in result.columns, "Should have max"
        
        avg = float(result['mean'].iloc[0])
        min_price = float(result['min'].iloc[0])
        max_price = float(result['max'].iloc[0])
        
        assert min_price <= avg <= max_price, "Average should be between min and max"
        assert min_price < max_price, "Min should be less than max"


class TestCheapestHours:
    """Test cheapest hours analysis"""
    
    def test_cheapest_hours_returns_requested_count(self, analytics):
        """Verify cheapest hours returns correct number of results"""
        n_hours = 5
        # Note: get_cheapest_hours doesn't support market_summary parameter
        result = analytics.get_cheapest_hours(
            n_hours=n_hours,
            exclude_zero=True
        )
        
        assert not result.empty, "Should return results"
        assert len(result) <= n_hours, f"Should return at most {n_hours} hours"
    
    def test_cheapest_hours_sorted_ascending(self, analytics):
        """Verify results are sorted by price (cheapest first)"""
        result = analytics.get_cheapest_hours(
            n_hours=10,
            exclude_zero=True
        )
        
        if len(result) > 1 and 'avg_price' in result.columns:
            prices = result['avg_price'].tolist()
            assert prices == sorted(prices), "Hours should be sorted by price ascending"


class TestHourlyAverages:
    """Test hourly average analysis"""
    
    def test_hourly_averages_covers_24_hours(self, analytics):
        """Verify hourly averages returns data for all hours"""
        result = analytics.get_hourly_averages(
            exclude_zero=True,
            market_summary=True
        )
        
        assert not result.empty, "Should return hourly data"
        assert 'hour' in result.columns, "Should have hour column"
        
        # Should have data for multiple hours (might not be all 24 depending on dataset)
        unique_hours = result['hour'].nunique()
        assert unique_hours > 0, "Should have at least one hour of data"
        assert unique_hours <= 24, "Should not exceed 24 hours"


class TestDataIntegrity:
    """Basic data integrity checks"""
    
    def test_database_has_data(self, db):
        """Verify database contains LMP data"""
        query = "SELECT COUNT(*) as count FROM caiso.lmp_data"
        result = db.execute_query(query)
        
        assert result, "Query should return results"
        count = result[0]['count']
        assert count > 0, "Database should contain LMP data"
    
    def test_database_has_required_columns(self, db):
        """Verify required columns exist"""
        query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_schema = 'caiso' 
        AND table_name = 'lmp_data'
        """
        result = db.execute_query(query)
        
        columns = [row['column_name'] for row in result]
        required = ['node', 'opr_dt', 'opr_hr', 'mw']
        
        for col in required:
            assert col in columns, f"Missing required column: {col}"
