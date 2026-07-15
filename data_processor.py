import pandas as pd
import numpy as np
from datetime import datetime, date
import io
import csv
import logging
import sys
from typing import Dict, Any, List, Tuple

class CAISODataProcessor:
    """Handles processing and cleaning of CAISO LMP data"""
    
    def __init__(self):
        self.required_columns = ['NODE', 'MW', 'OPR_DT', 'OPR_HR']
        self.optional_columns = ['MCC', 'MLC', 'POS']
        self.logger = logging.getLogger(__name__)
        
    def process_csv_content_to_db_fast(self, csv_content: str, source_file: str = "") -> Dict[str, Any]:
        """Legacy method - no longer stores to database. Use parse_csv_to_records instead."""
        self.logger.warning("process_csv_content_to_db_fast is deprecated. Use parse_csv_to_records() + parquet storage.")
        return {'records_inserted': 0, 'error': 'Deprecated: raw data now stored as parquet in S3'}

    def parse_csv_to_records(self, csv_content: str) -> Tuple[date, List[Dict]]:
        """Parse CSV content and return (operating_date, list of records).
        
        Returns:
            Tuple of (opr_date, records) where records is list of dicts with node, mw, opr_hr
        """
        lines = csv_content.strip().split('\n')
        if len(lines) < 2:
            return None, []
        
        reader = csv.reader(lines)
        header = [col.upper() for col in next(reader)]
        
        opr_dt_idx = next((i for i, c in enumerate(header) if c == 'OPR_DT'), None)
        opr_hr_idx = next((i for i, c in enumerate(header) if c == 'OPR_HR'), None)
        node_idx = next((i for i, c in enumerate(header) if c == 'NODE' or 'PNODE' in c), None)
        mw_idx = next((i for i, c in enumerate(header) if c == 'MW'), None)
        
        if opr_dt_idx is None or opr_hr_idx is None or node_idx is None or mw_idx is None:
            return None, []
        
        records = []
        opr_date = None
        
        for row in reader:
            try:
                if len(row) <= max(opr_dt_idx, opr_hr_idx, node_idx, mw_idx):
                    continue
                
                opr_dt_str = row[opr_dt_idx].strip()
                if not opr_dt_str:
                    continue
                
                if opr_date is None:
                    opr_date = datetime.strptime(opr_dt_str, '%Y-%m-%d').date()
                
                try:
                    opr_hr = int(row[opr_hr_idx].strip())
                except:
                    continue
                
                node = row[node_idx].strip()
                mw_str = row[mw_idx].strip()
                if not mw_str or not node:
                    continue
                try:
                    mw = float(mw_str)
                except:
                    continue
                
                records.append({
                    'node': node,
                    'mw': mw,
                    'opr_hr': opr_hr
                })
            except Exception:
                continue
        
        return opr_date, records

    def process_csv_content_to_db(self, csv_content: str, source_file: str = "") -> Dict[str, Any]:
        """Legacy method - no longer stores to database. Use parse_csv_to_records instead."""
        self.logger.warning("process_csv_content_to_db is deprecated. Use parse_csv_to_records() + parquet storage.")
        return {'records_inserted': 0, 'error': 'Deprecated: raw data now stored as parquet in S3'}
    
    def process_csv_content(self, csv_content):
        """Process CSV content from string - legacy method for backward compatibility"""
        try:
            # Read CSV from string
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Basic validation
            if df.empty:
                return None
                
            # Check if this looks like CAISO LMP data
            if not self._validate_caiso_format(df):
                return None
                
            # Process the data
            df = self._standardize_columns(df)
            df = self._parse_datetime(df)
            df = self._clean_numeric_columns(df)
            
            return df
            
        except Exception as e:
            logging.error(f"Error processing CSV content: {str(e)}")
            return None
    
    def _validate_caiso_format(self, df):
        """Validate if DataFrame contains CAISO LMP data"""
        # Check for key columns that should exist in CAISO LMP data
        expected_patterns = ['INTERVALSTARTTIME', 'NODE', 'MW']
        
        for pattern in expected_patterns:
            if not any(pattern in col.upper() for col in df.columns):
                return False
        return True
    
    def _standardize_columns(self, df):
        """Standardize column names"""
        # Create mapping for common CAISO column variations
        column_mapping = {}
        
        for col in df.columns:
            col_upper = col.upper()
            if 'INTERVALSTARTTIME' in col_upper and 'GMT' in col_upper:
                column_mapping[col] = 'INTERVALSTARTTIME_GMT'
            elif col_upper == 'NODE':  # Exact match for NODE
                column_mapping[col] = 'NODE'
            elif 'NODE' in col_upper and 'ID' in col_upper:
                column_mapping[col] = 'NODE_ID'  # Map NODE_ID separately
            elif col_upper == 'MW' or 'PRICE' in col_upper:
                column_mapping[col] = 'MW'
            elif 'MCC' in col_upper or 'CONGESTION' in col_upper:
                column_mapping[col] = 'MCC'
            elif 'MLC' in col_upper or 'LOSS' in col_upper:
                column_mapping[col] = 'MLC'
            elif 'POS' in col_upper:
                column_mapping[col] = 'POS'
            elif col_upper == 'OPR_HR':  # Preserve original operational hour
                column_mapping[col] = 'opr_hr'
            elif col_upper == 'OPR_DT':  # Preserve original operational date
                column_mapping[col] = 'opr_dt'
                
        # Apply column mapping
        df_renamed = df.rename(columns=column_mapping)
        
        # Handle NODE vs NODE_ID conflict: keep only NODE
        if 'NODE' in df_renamed.columns and 'NODE_ID' in df_renamed.columns:
            df_renamed = df_renamed.drop(columns=['NODE_ID'])
            
        return df_renamed
    
    def _parse_datetime(self, df):
        """Parse datetime columns and create operational date/hour columns.
        
        CRITICAL: OPR_HR and OPR_DT should come from CAISO's columns directly.
        NEVER derive from INTERVALSTARTTIME_GMT (it's UTC, causes 8-hour offset).
        Only use timestamp as fallback if OPR_HR/OPR_DT columns are missing.
        """
        if 'opr_hr' not in df.columns or 'opr_dt' not in df.columns:
            if 'INTERVALSTARTTIME_GMT' in df.columns:
                try:
                    timestamp_col = pd.to_datetime(df['INTERVALSTARTTIME_GMT'])
                    
                    if 'opr_dt' not in df.columns:
                        logging.warning("OPR_DT missing, falling back to GMT timestamp (may have offset)")
                        df['opr_dt'] = timestamp_col.dt.date
                    if 'opr_hr' not in df.columns:
                        logging.warning("OPR_HR missing, falling back to GMT timestamp (may have offset)")
                        df['opr_hr'] = timestamp_col.dt.hour
                    
                    df['DAY_OF_WEEK'] = timestamp_col.dt.day_name()
                    df['INTERVALSTARTTIME_GMT'] = timestamp_col
                    
                except Exception as e:
                    logging.warning(f"Error parsing datetime: {str(e)}")
        
        return df
    
    def _clean_numeric_columns(self, df):
        """Clean and validate numeric price columns"""
        numeric_columns = ['MW', 'MCC', 'MLC', 'POS']
        
        for col in numeric_columns:
            if col in df.columns:
                # Convert to numeric, replacing non-numeric values with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove obviously erroneous values (e.g., prices > $10,000/MWh or < -$1,000/MWh)
                if col == 'MW':
                    df.loc[(df[col] > 10000) | (df[col] < -1000), col] = np.nan
        
        return df
    
    def clean_and_validate(self, df):
        """Final cleaning and validation of the complete dataset"""
        if df is None or df.empty:
            return df
        
        # Remove rows with missing critical data
        df = df.dropna(subset=['opr_dt', 'opr_hr', 'NODE', 'MW'])
        
        # Remove duplicate records
        df = df.drop_duplicates(subset=['opr_dt', 'opr_hr', 'NODE'])
        
        # Sort by operational date, hour, and node
        df = df.sort_values(['opr_dt', 'opr_hr', 'NODE'])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def get_data_summary_from_db(self) -> Dict[str, Any]:
        """Get data summary from MotherDuck bx_daily_summary table via subprocess"""
        try:
            import subprocess
            import json
            cmd = [sys.executable, 'subprocess_query.py', 'data_summary']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode == 0:
                data = json.loads(result.stdout.strip())
                if 'error' in data:
                    self.logger.error(f"Query error: {data['error']}")
                    return {}
                return data
            return {}
        except Exception as e:
            self.logger.error(f"Error getting data summary: {str(e)}")
            return {}
