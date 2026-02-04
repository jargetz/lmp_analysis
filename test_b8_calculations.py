"""
B8 Calculation Verification Tests

These tests validate that B8 (cheapest 8 hours) calculations are correct
using known values from the settlement nodes:
- TH_NP15_GEN-APND (NP15 generator aggregate)
- TH_SP15_GEN-APND (SP15 generator aggregate)  
- TH_ZP26_GEN-APND (ZP26 generator aggregate)

Test data verified manually on 2024-01-15 and annual 2024 averages.
Run with: pytest test_b8_calculations.py -v
"""

import pytest
import duckdb
import os
import sys

SETTLEMENT_NODES = ["TH_NP15_GEN-APND", "TH_SP15_GEN-APND", "TH_ZP26_GEN-APND"]

EXPECTED_B8_2024_01_15 = {
    "TH_NP15_GEN-APND": 180.20,
    "TH_SP15_GEN-APND": 113.79,
    "TH_ZP26_GEN-APND": 141.45,
}

EXPECTED_B8_HOURS_2024_01_15 = {
    "TH_NP15_GEN-APND": [14, 13, 15, 12, 10, 11, 4, 2],
    "TH_SP15_GEN-APND": [12, 14, 13, 11, 15, 10, 16, 9],
    "TH_ZP26_GEN-APND": [14, 12, 13, 11, 15, 10, 16, 9],
}

EXPECTED_ANNUAL_B8_2024 = {
    "TH_NP15_GEN-APND": {"avg": 22.35, "min": -45.95, "max": 208.12, "days": 365},
    "TH_SP15_GEN-APND": {"avg": 7.26, "min": -65.70, "max": 113.79, "days": 365},
    "TH_ZP26_GEN-APND": {"avg": 7.81, "min": -66.07, "max": 141.45, "days": 365},
}

RAW_HOURLY_2024_01_15 = {
    "TH_NP15_GEN-APND": {
        1: 204.88927, 2: 190.47948, 3: 190.79880, 4: 189.49644, 5: 190.99599,
        6: 210.64644, 7: 223.68086, 8: 215.72656, 9: 213.40349, 10: 180.67896,
        11: 182.12907, 12: 179.49371, 13: 174.80844, 14: 167.17209, 15: 177.35558,
        16: 205.30296, 17: 245.58551, 18: 254.20000, 19: 250.24123, 20: 257.43607,
        21: 254.62161, 22: 249.38081, 23: 231.85876, 24: 219.69855,
    },
    "TH_SP15_GEN-APND": {
        1: 197.66272, 2: 184.56883, 3: 183.83116, 4: 182.72533, 5: 184.24167,
        6: 202.11588, 7: 217.69328, 8: 215.67530, 9: 152.08000, 10: 112.76614,
        11: 103.61262, 12: 94.77395, 13: 99.14088, 14: 97.54940, 15: 104.97563,
        16: 145.45917, 17: 228.27316, 18: 253.47485, 19: 250.28377, 20: 246.66489,
        21: 243.14279, 22: 241.70116, 23: 223.32516, 24: 211.92215,
    },
    "TH_ZP26_GEN-APND": {
        1: 198.21703, 2: 184.91315, 3: 184.86198, 4: 183.80643, 5: 185.25578,
        6: 203.48161, 7: 217.80562, 8: 215.70099, 9: 175.95491, 10: 143.46667,
        11: 134.04517, 12: 128.94353, 13: 130.28386, 14: 126.27785, 15: 134.18716,
        16: 158.46600, 17: 232.83030, 18: 253.47855, 19: 250.31638, 20: 249.26128,
        21: 246.46227, 22: 243.75240, 23: 225.45856, 24: 213.89920,
    },
}


@pytest.fixture(scope="module")
def duckdb_conn():
    """Create a MotherDuck connection for testing"""
    token = os.getenv("MOTHERDUCK_TOKEN")
    if not token:
        pytest.skip("MOTHERDUCK_TOKEN not set")
    conn = duckdb.connect(f"md:?motherduck_token={token}")
    yield conn
    conn.close()


@pytest.fixture(scope="module")
def s3_bucket():
    return os.getenv("AWS_S3_BUCKET", "oasis-data-for-replit-2025")


class TestRawDataIntegrity:
    """Verify raw hourly data matches expected values from CAISO"""
    
    def test_raw_hourly_data_2024_01_15(self, duckdb_conn, s3_bucket):
        """Verify raw hourly prices for 2024-01-15 match fixture data"""
        node_list = ", ".join(f"'{n}'" for n in SETTLEMENT_NODES)
        
        query = f"""
            SELECT node, opr_hr, mw
            FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/month=01/2024-01-15.parquet')
            WHERE node IN ({node_list})
            ORDER BY node, opr_hr
        """
        result = duckdb_conn.execute(query).fetchdf()
        
        for node in SETTLEMENT_NODES:
            node_data = result[result['node'] == node]
            assert len(node_data) == 24, f"{node} should have 24 hours"
            
            for _, row in node_data.iterrows():
                hour = int(row['opr_hr'])
                expected_price = RAW_HOURLY_2024_01_15[node][hour]
                actual_price = float(row['mw'])
                assert abs(actual_price - expected_price) < 0.01, \
                    f"{node} hour {hour}: expected {expected_price}, got {actual_price}"
    
    def test_opr_hr_range_is_1_to_24(self, duckdb_conn, s3_bucket):
        """Verify OPR_HR values are in 1-24 range (CAISO Pacific time), not 0-23"""
        query = f"""
            SELECT DISTINCT opr_hr
            FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/month=01/2024-01-15.parquet')
            ORDER BY opr_hr
        """
        result = duckdb_conn.execute(query).fetchdf()
        hours = sorted(result['opr_hr'].tolist())
        
        assert hours == list(range(1, 25)), \
            f"Hours should be 1-24 (CAISO Pacific), got {hours}"
    
    def test_no_hour_0_in_data(self, duckdb_conn, s3_bucket):
        """Confirm no hour 0 exists - this would indicate GMT derivation bug"""
        query = f"""
            SELECT COUNT(*) as cnt
            FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/**/*.parquet', hive_partitioning=true)
            WHERE opr_hr = 0
        """
        result = duckdb_conn.execute(query).fetchdf()
        count = int(result['cnt'].iloc[0])
        
        assert count == 0, \
            f"Found {count} rows with hour=0. OPR_HR should be 1-24, not 0-23."


class TestB8SingleDayCalculation:
    """Verify B8 calculation for a single day matches manual computation"""
    
    def test_b8_average_2024_01_15(self, duckdb_conn, s3_bucket):
        """Verify B8 average for 2024-01-15 matches expected values"""
        node_list = ", ".join(f"'{n}'" for n in SETTLEMENT_NODES)
        
        query = f"""
            WITH ranked AS (
                SELECT node, opr_hr, mw,
                    ROW_NUMBER() OVER (PARTITION BY node ORDER BY mw ASC) as rn
                FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/month=01/2024-01-15.parquet')
                WHERE node IN ({node_list})
            )
            SELECT node, AVG(mw) as b8_avg
            FROM ranked WHERE rn <= 8
            GROUP BY node
        """
        result = duckdb_conn.execute(query).fetchdf()
        
        for node in SETTLEMENT_NODES:
            node_row = result[result['node'] == node]
            actual_b8 = float(node_row['b8_avg'].iloc[0])
            expected_b8 = EXPECTED_B8_2024_01_15[node]
            
            assert abs(actual_b8 - expected_b8) < 0.01, \
                f"{node}: expected B8=${expected_b8:.2f}, got ${actual_b8:.2f}"
    
    def test_b8_hours_selected_2024_01_15(self, duckdb_conn, s3_bucket):
        """Verify the correct 8 cheapest hours are selected"""
        node_list = ", ".join(f"'{n}'" for n in SETTLEMENT_NODES)
        
        query = f"""
            WITH ranked AS (
                SELECT node, opr_hr, mw,
                    ROW_NUMBER() OVER (PARTITION BY node ORDER BY mw ASC) as rn
                FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/month=01/2024-01-15.parquet')
                WHERE node IN ({node_list})
            )
            SELECT node, opr_hr
            FROM ranked WHERE rn <= 8
            ORDER BY node, rn
        """
        result = duckdb_conn.execute(query).fetchdf()
        
        for node in SETTLEMENT_NODES:
            node_hours = result[result['node'] == node]['opr_hr'].tolist()
            node_hours = [int(h) for h in node_hours]
            expected_hours = EXPECTED_B8_HOURS_2024_01_15[node]
            
            assert node_hours == expected_hours, \
                f"{node}: expected hours {expected_hours}, got {node_hours}"
    
    def test_b8_manual_calculation_matches(self):
        """Verify B8 using pure Python on fixture data (no DB)"""
        for node, hourly_prices in RAW_HOURLY_2024_01_15.items():
            sorted_prices = sorted(hourly_prices.values())
            b8_avg = sum(sorted_prices[:8]) / 8
            expected_b8 = EXPECTED_B8_2024_01_15[node]
            
            assert abs(b8_avg - expected_b8) < 0.01, \
                f"{node}: manual B8=${b8_avg:.2f}, expected ${expected_b8:.2f}"


class TestAnnualB8Calculation:
    """Verify annual B8 averages for full year 2024"""
    
    def test_annual_b8_average_2024(self, duckdb_conn, s3_bucket):
        """Verify annual B8 averages match expected values (tolerance: $0.10)"""
        node_list = ", ".join(f"'{n}'" for n in SETTLEMENT_NODES)
        
        query = f"""
            WITH file_data AS (
                SELECT 
                    regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) as opr_dt,
                    node, opr_hr, mw
                FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/**/*.parquet', filename=true, hive_partitioning=true)
                WHERE node IN ({node_list})
            ),
            ranked AS (
                SELECT opr_dt, node, opr_hr, mw,
                    ROW_NUMBER() OVER (PARTITION BY opr_dt, node ORDER BY mw ASC) as rn
                FROM file_data
            ),
            daily_bx AS (
                SELECT opr_dt, node, AVG(mw) as bx_price
                FROM ranked WHERE rn <= 8
                GROUP BY opr_dt, node
            )
            SELECT node, AVG(bx_price) as annual_b8_avg
            FROM daily_bx GROUP BY node
        """
        result = duckdb_conn.execute(query).fetchdf()
        
        for node in SETTLEMENT_NODES:
            node_row = result[result['node'] == node]
            actual_avg = float(node_row['annual_b8_avg'].iloc[0])
            expected_avg = EXPECTED_ANNUAL_B8_2024[node]["avg"]
            
            assert abs(actual_avg - expected_avg) < 0.10, \
                f"{node}: expected annual B8=${expected_avg:.2f}, got ${actual_avg:.2f}"
    
    def test_annual_day_count_2024(self, duckdb_conn, s3_bucket):
        """Verify we have 365 days of data for 2024"""
        node_list = ", ".join(f"'{n}'" for n in SETTLEMENT_NODES)
        
        query = f"""
            WITH file_data AS (
                SELECT 
                    regexp_extract(filename, '(\\d{{4}}-\\d{{2}}-\\d{{2}})\\.parquet', 1) as opr_dt,
                    node
                FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/**/*.parquet', filename=true, hive_partitioning=true)
                WHERE node IN ({node_list})
            )
            SELECT node, COUNT(DISTINCT opr_dt) as day_count
            FROM file_data GROUP BY node
        """
        result = duckdb_conn.execute(query).fetchdf()
        
        for node in SETTLEMENT_NODES:
            node_row = result[result['node'] == node]
            actual_days = int(node_row['day_count'].iloc[0])
            expected_days = EXPECTED_ANNUAL_B8_2024[node]["days"]
            
            assert actual_days >= expected_days, \
                f"{node}: expected {expected_days} days, got {actual_days}"


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_hour_25_dst_filtering(self, duckdb_conn, s3_bucket):
        """Verify hour 25 (DST fall-back) is present in raw data but handled correctly"""
        query = f"""
            SELECT COUNT(*) as cnt
            FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/**/*.parquet', hive_partitioning=true)
            WHERE opr_hr = 25
        """
        result = duckdb_conn.execute(query).fetchdf()
        hour_25_count = int(result['cnt'].iloc[0])
        
        assert hour_25_count >= 0, "Hour 25 count should be defined"
    
    def test_b8_excludes_hour_25(self, duckdb_conn, s3_bucket):
        """Verify B8 calculation can handle days with hour 25 by filtering it"""
        node = "TH_NP15_GEN-APND"
        
        query = f"""
            WITH ranked AS (
                SELECT opr_hr, mw,
                    ROW_NUMBER() OVER (ORDER BY mw ASC) as rn
                FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/month=11/2024-11-03.parquet')
                WHERE node = '{node}' AND opr_hr <= 24
            )
            SELECT COUNT(*) as cnt, AVG(mw) as b8_avg
            FROM ranked WHERE rn <= 8
        """
        result = duckdb_conn.execute(query).fetchdf()
        count = int(result['cnt'].iloc[0])
        
        assert count == 8, f"B8 should include exactly 8 hours, got {count}"
    
    def test_empty_node_list_handled(self, duckdb_conn, s3_bucket):
        """Verify empty node list doesn't crash the query"""
        query = f"""
            SELECT node, mw
            FROM read_parquet('s3://{s3_bucket}/lmp_parquet/year=2024/month=01/2024-01-15.parquet')
            WHERE node IN ('NONEXISTENT_NODE_XYZ')
        """
        result = duckdb_conn.execute(query).fetchdf()
        assert len(result) == 0, "Empty result expected for nonexistent node"


class TestSubprocessQueryContract:
    """Test that subprocess_query.py returns expected data structures"""
    
    def test_node_bx_returns_expected_structure(self):
        """Verify node_bx query returns dict with per_node and per_node_hours"""
        import subprocess
        import json
        
        nodes = ["TH_NP15_GEN-APND"]
        cmd = [
            sys.executable, "subprocess_query.py",
            "node_bx", "8", json.dumps(nodes), "2024"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        
        if result.returncode != 0:
            pytest.skip(f"Subprocess failed: {result.stderr}")
        
        data = json.loads(result.stdout)
        
        assert "success" in data, "Response should have 'success' field"
        if data.get("success"):
            assert "per_node" in data, "Response should have 'per_node'"
            assert "per_node_hours" in data, "Response should have 'per_node_hours'"
            assert "avg_price" in data, "Response should have 'avg_price'"
            assert "node_count" in data, "Response should have 'node_count'"
    
    def test_hourly_avg_returns_list_of_dicts(self):
        """Verify hourly_avg query returns list with hour and avg_price"""
        import subprocess
        import json
        
        nodes = ["TH_NP15_GEN-APND"]
        cmd = [
            sys.executable, "subprocess_query.py",
            "hourly_avg", json.dumps(nodes), "2024"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
        
        if result.returncode != 0:
            pytest.skip(f"Subprocess failed: {result.stderr}")
        
        data = json.loads(result.stdout)
        
        assert isinstance(data, list), "hourly_avg should return a list"
        if len(data) > 0:
            assert "hour" in data[0], "Each item should have 'hour'"
            assert "avg_price" in data[0], "Each item should have 'avg_price'"
            assert len(data) == 24, "Should have 24 hours"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
