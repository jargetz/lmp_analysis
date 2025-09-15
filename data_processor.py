import pandas as pd
import numpy as np
from datetime import datetime
import io
import logging

class CAISODataProcessor:
    """Handles processing and cleaning of CAISO LMP data"""
    
    def __init__(self):
        self.required_columns = ['INTERVALSTARTTIME_GMT', 'NODE', 'MW']
        self.optional_columns = ['MCC', 'MLC', 'POS']  # Congestion, Loss, Position components
        
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
    
    def get_data_quality_report(self, df):
        """Generate a data quality report"""
        if df is None or df.empty:
            return {"error": "No data available"}
        
        report = {
            "total_records": len(df),
            "date_range": {
                "start": df['INTERVALSTARTTIME_GMT'].min() if 'INTERVALSTARTTIME_GMT' in df.columns else None,
                "end": df['INTERVALSTARTTIME_GMT'].max() if 'INTERVALSTARTTIME_GMT' in df.columns else None
            },
            "unique_nodes": df['NODE'].nunique() if 'NODE' in df.columns else 0,
            "price_statistics": {
                "min": df['MW'].min() if 'MW' in df.columns else None,
                "max": df['MW'].max() if 'MW' in df.columns else None,
                "mean": df['MW'].mean() if 'MW' in df.columns else None,
                "median": df['MW'].median() if 'MW' in df.columns else None
            },
            "missing_data": {
                col: df[col].isnull().sum() for col in df.columns if col in ['MW', 'MCC', 'MLC', 'POS']
            }
        }
        
        return report
