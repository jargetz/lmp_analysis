import pandas as pd
import numpy as np
from datetime import datetime, date
import io
import csv
import logging
from typing import Dict, Any, List, Tuple
from database import DatabaseManager

class CAISODataProcessor:
    """Handles processing and cleaning of CAISO LMP data with PostgreSQL storage"""
    
    def __init__(self):
        self.required_columns = ['NODE', 'MW']  # We'll create opr_dt and opr_hr from INTERVALSTARTTIME_GMT
        self.optional_columns = ['MCC', 'MLC', 'POS']  # Congestion, Loss, Position components
        self.db = DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
    def process_csv_content_to_db_fast(self, csv_content: str, source_file: str = "") -> Dict[str, Any]:
        """Fast CSV processing - lean schema (node, mw, opr_dt, opr_hr, source_file only)"""
        try:
            lines = csv_content.strip().split('\n')
            if len(lines) < 2:
                return {'records_inserted': 0, 'error': 'Empty or invalid CSV'}
            
            reader = csv.reader(lines)
            header = [col.upper() for col in next(reader)]
            
            ts_idx = next((i for i, c in enumerate(header) if 'INTERVALSTARTTIME' in c and 'GMT' in c), None)
            node_idx = next((i for i, c in enumerate(header) if c == 'NODE' or 'PNODE' in c), None)
            mw_idx = next((i for i, c in enumerate(header) if c == 'MW'), None)
            
            if ts_idx is None or node_idx is None or mw_idx is None:
                return {'records_inserted': 0, 'error': 'Missing required columns'}
            
            output = io.StringIO()
            writer = csv.writer(output)
            row_count = 0
            
            for row in reader:
                try:
                    if len(row) <= max(ts_idx, node_idx, mw_idx):
                        continue
                    
                    ts_str = row[ts_idx].strip()
                    if not ts_str:
                        continue
                    
                    try:
                        if 'T' in ts_str:
                            dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00').split('+')[0])
                        else:
                            dt = datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')
                    except:
                        dt = datetime.strptime(ts_str[:19], '%Y-%m-%dT%H:%M:%S')
                    
                    opr_dt = dt.date().isoformat()
                    opr_hr = dt.hour
                    
                    node = row[node_idx].strip()
                    mw = row[mw_idx].strip()
                    if not mw or not node:
                        continue
                    try:
                        float(mw)
                    except:
                        continue
                    
                    # Lean schema: only 5 columns
                    writer.writerow([node, mw, opr_dt, opr_hr, source_file])
                    row_count += 1
                except Exception:
                    continue
            
            if row_count == 0:
                return {'records_inserted': 0, 'error': 'No valid rows after processing'}
            
            output.seek(0)
            records_inserted = self.db.bulk_insert_lmp_data_raw(output, row_count)
            
            return {
                'records_inserted': records_inserted,
                'source_file': source_file,
                'total_rows_processed': row_count
            }
            
        except Exception as e:
            self.logger.error(f"Error in fast CSV processing from {source_file}: {str(e)}")
            return {'records_inserted': 0, 'error': str(e)}

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
        """Process CSV content and store directly in database"""
        try:
            # Read CSV from string
            df = pd.read_csv(io.StringIO(csv_content))
            
            # Basic validation
            if df.empty:
                return {'records_inserted': 0, 'error': 'Empty CSV file'}
                
            # Check if this looks like CAISO LMP data
            if not self._validate_caiso_format(df):
                return {'records_inserted': 0, 'error': 'Invalid CAISO LMP format'}
                
            # Process the data
            df = self._standardize_columns(df)
            df = self._parse_datetime(df)
            df = self._clean_numeric_columns(df)
            
            if df.empty:
                return {'records_inserted': 0, 'error': 'No valid data after cleaning'}
            
            # Add source file tracking
            df['source_file'] = source_file
            
            # Store in database
            records_inserted = self.db.bulk_insert_lmp_data(df)
            
            return {
                'records_inserted': records_inserted,
                'source_file': source_file,
                'total_rows_processed': len(df)
            }
            
        except Exception as e:
            self.logger.error(f"Error processing CSV content from {source_file}: {str(e)}")
            return {'records_inserted': 0, 'error': str(e)}
    
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
        """Parse datetime columns and create operational date/hour columns"""
        if 'INTERVALSTARTTIME_GMT' in df.columns:
            try:
                # Parse the timestamp
                timestamp_col = pd.to_datetime(df['INTERVALSTARTTIME_GMT'])
                
                # Create operational date and hour columns (required by new schema)
                df['opr_dt'] = timestamp_col.dt.date
                df['opr_hr'] = timestamp_col.dt.hour
                
                # Extract additional time-based features (optional)
                df['DAY_OF_WEEK'] = timestamp_col.dt.day_name()
                
                # Keep the timestamp column (database requires it with NOT NULL constraint)
                # Just ensure it's properly parsed
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
        """Get data summary from database"""
        try:
            return self.db.get_data_summary()
        except Exception as e:
            self.logger.error(f"Error getting data summary from database: {str(e)}")
            return {}
    
    def get_data_quality_report_from_db(self) -> Dict[str, Any]:
        """Generate a data quality report from database"""
        try:
            summary = self.db.get_data_summary()
            
            if not summary:
                return {"error": "No data available"}
            
            # Handle both dict and list responses
            if isinstance(summary, list) and len(summary) > 0:
                summary = summary[0]  # type: ignore
            
            if not isinstance(summary, dict):
                return {"error": "Invalid data format from database"}
            
            report = {
                "total_records": summary.get('total_records', 0),
                "date_range": {
                    "start": summary.get('earliest_date'),
                    "end": summary.get('latest_date')
                },
                "unique_nodes": summary.get('unique_nodes', 0),
                "price_statistics": {
                    "min": summary.get('min_price'),
                    "max": summary.get('max_price'),
                    "mean": summary.get('avg_price')
                },
                "missing_data": {
                    "mcc": summary.get('missing_mcc', 0) if isinstance(summary, dict) else 0,
                    "mlc": summary.get('missing_mlc', 0) if isinstance(summary, dict) else 0,
                    "pos": summary.get('missing_pos', 0) if isinstance(summary, dict) else 0
                }
            }
            
            return report
        except Exception as e:
            self.logger.error(f"Error generating data quality report: {str(e)}")
            return {"error": str(e)}