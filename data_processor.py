import pandas as pd
import numpy as np
from datetime import datetime, date
import io
import logging
from database import DatabaseManager

class CAISODataProcessor:
    """Handles processing and cleaning of CAISO LMP data with PostgreSQL storage"""
    
    def __init__(self):
        self.required_columns = ['INTERVALSTARTTIME_GMT', 'NODE', 'MW']
        self.optional_columns = ['MCC', 'MLC', 'POS']  # Congestion, Loss, Position components
        self.db = DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
    def process_csv_content(self, csv_content):
        """Process CSV content from string"""
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
            elif 'NODE' in col_upper and 'ID' in col_upper:
                column_mapping[col] = 'NODE'
            elif col_upper == 'MW' or 'PRICE' in col_upper:
                column_mapping[col] = 'MW'
            elif 'MCC' in col_upper or 'CONGESTION' in col_upper:
                column_mapping[col] = 'MCC'
            elif 'MLC' in col_upper or 'LOSS' in col_upper:
                column_mapping[col] = 'MLC'
            elif 'POS' in col_upper:
                column_mapping[col] = 'POS'
        
        return df.rename(columns=column_mapping)
    
    def _parse_datetime(self, df):
        """Parse datetime columns"""
        if 'INTERVALSTARTTIME_GMT' in df.columns:
            try:
                df['INTERVALSTARTTIME_GMT'] = pd.to_datetime(df['INTERVALSTARTTIME_GMT'])
                
                # Extract additional time-based features
                df['HOUR'] = df['INTERVALSTARTTIME_GMT'].dt.hour
                df['DAY_OF_WEEK'] = df['INTERVALSTARTTIME_GMT'].dt.day_name()
                df['DATE'] = df['INTERVALSTARTTIME_GMT'].dt.date
                
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
        df = df.dropna(subset=['INTERVALSTARTTIME_GMT', 'NODE', 'MW'])
        
        # Remove duplicate records
        df = df.drop_duplicates(subset=['INTERVALSTARTTIME_GMT', 'NODE'])
        
        # Sort by time and node
        df = df.sort_values(['INTERVALSTARTTIME_GMT', 'NODE'])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def process_and_store_csv_content(self, csv_content, update_cache=True):
        """Process CSV content and store directly in database"""
        try:
            # Process the data as before
            df = self.process_csv_content(csv_content)
            if df is None or df.empty:
                return {"success": False, "message": "No valid data found"}
            
            # Clean and validate
            df = self.clean_and_validate(df)
            if df.empty:
                return {"success": False, "message": "No data remained after cleaning"}
            
            # Store in database
            records_inserted = self.db.bulk_insert_lmp_data(df)
            
            # Update pre-computed analytics cache if requested
            if update_cache and 'DATE' in df.columns:
                unique_dates = df['DATE'].unique()
                for date_val in unique_dates:
                    if isinstance(date_val, date):
                        self.db.update_cheapest_hours_cache(date_val)
            
            return {
                "success": True,
                "records_inserted": records_inserted,
                "date_range": {
                    "start": df['INTERVALSTARTTIME_GMT'].min(),
                    "end": df['INTERVALSTARTTIME_GMT'].max()
                },
                "unique_nodes": df['NODE'].nunique()
            }
            
        except Exception as e:
            self.logger.error(f"Error processing and storing CSV content: {str(e)}")
            return {"success": False, "message": str(e)}
    
    def process_multiple_zip_files(self, zip_files, progress_callback=None):
        """Process multiple zip files and store all data in database"""
        total_records = 0
        processed_files = 0
        errors = []
        
        for i, zip_file in enumerate(zip_files):
            if progress_callback:
                progress_callback(i, len(zip_files), f"Processing {zip_file.name}")
            
            try:
                import zipfile
                with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                    for file_name in zip_ref.namelist():
                        if file_name.endswith('.csv'):
                            with zip_ref.open(file_name) as csv_file:
                                content = csv_file.read().decode('utf-8')
                                result = self.process_and_store_csv_content(content)
                                
                                if result['success']:
                                    total_records += result['records_inserted']
                                    processed_files += 1
                                else:
                                    errors.append(f"{file_name}: {result['message']}")
            
            except Exception as e:
                errors.append(f"{zip_file.name}: {str(e)}")
        
        return {
            "total_records_inserted": total_records,
            "processed_files": processed_files,
            "errors": errors
        }
    
    def get_data_quality_report_from_db(self):
        """Generate a data quality report from database"""
        try:
            # Get overall summary
            summary = self.db.get_data_summary()
            
            if not summary:
                return {"error": "No data available in database"}
            
            # Get additional statistics
            query_missing = """
            SELECT 
                COUNT(CASE WHEN mcc IS NULL THEN 1 END) as missing_mcc,
                COUNT(CASE WHEN mlc IS NULL THEN 1 END) as missing_mlc,
                COUNT(CASE WHEN pos IS NULL THEN 1 END) as missing_pos,
                COUNT(*) as total_records
            FROM caiso.lmp_data
            """
            
            missing_data_result = self.db.execute_query(query_missing, fetch_all=False)
            
            report = {
                "total_records": summary.get('total_records', 0),
                "date_range": {
                    "start": summary.get('earliest_date'),
                    "end": summary.get('latest_date')
                },
                "unique_nodes": summary.get('unique_nodes', 0),
                "price_statistics": {
                    "min": float(summary.get('min_price', 0)) if summary.get('min_price') else None,
                    "max": float(summary.get('max_price', 0)) if summary.get('max_price') else None,
                    "mean": float(summary.get('avg_price', 0)) if summary.get('avg_price') else None
                },
                "missing_data": {
                    "MCC": missing_data_result.get('missing_mcc', 0) if missing_data_result and isinstance(missing_data_result, dict) else 0,
                    "MLC": missing_data_result.get('missing_mlc', 0) if missing_data_result and isinstance(missing_data_result, dict) else 0,
                    "POS": missing_data_result.get('missing_pos', 0) if missing_data_result and isinstance(missing_data_result, dict) else 0
                } if missing_data_result and isinstance(missing_data_result, dict) else {}
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating data quality report: {str(e)}")
            return {"error": f"Error generating report: {str(e)}"}
    
    def get_data_summary_from_db(self):
        """Get basic data summary from database for UI display"""
        try:
            return self.db.get_data_summary()
        except Exception as e:
            self.logger.error(f"Error getting data summary: {str(e)}")
            return {}
    
    def clear_data_by_date_range(self, start_date, end_date):
        """Clear data for a specific date range (administrative function)"""
        try:
            deleted_count = self.db.clear_data(start_date, end_date)
            return {"success": True, "deleted_records": deleted_count}
        except Exception as e:
            self.logger.error(f"Error clearing data: {str(e)}")
            return {"success": False, "message": str(e)}
