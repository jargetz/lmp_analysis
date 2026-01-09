import boto3
import os
import logging
import io
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import date
from typing import List, Dict, Any, Optional

class ParquetStorage:
    """Handles storing raw LMP data as Parquet files in S3"""
    
    def __init__(self):
        self.bucket_name = os.getenv('AWS_S3_BUCKET')
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.logger = logging.getLogger(__name__)
        self.parquet_prefix = "lmp_parquet/"
    
    def _get_parquet_key(self, opr_date: date) -> str:
        """Generate S3 key for parquet file based on date (organized by year/month)"""
        return f"{self.parquet_prefix}year={opr_date.year}/month={opr_date.month:02d}/{opr_date.isoformat()}.parquet"
    
    def write_day_to_parquet(self, data: List[Dict], opr_date: date) -> Dict[str, Any]:
        """Write a day's worth of LMP data to S3 as parquet.
        
        Args:
            data: List of dicts with keys: node, mw, opr_hr
            opr_date: The operating date for this data
            
        Returns:
            Dict with success status and details
        """
        if not data:
            return {'success': False, 'error': 'No data provided'}
        
        try:
            schema = pa.schema([
                ('node', pa.string()),
                ('mw', pa.float64()),
                ('opr_hr', pa.int32()),
            ])
            
            nodes = [d['node'] for d in data]
            mws = [float(d['mw']) for d in data]
            hours = [int(d['opr_hr']) for d in data]
            
            table = pa.table({
                'node': nodes,
                'mw': mws,
                'opr_hr': hours,
            }, schema=schema)
            
            buffer = io.BytesIO()
            pq.write_table(table, buffer, compression='snappy')
            buffer.seek(0)
            
            s3_key = self._get_parquet_key(opr_date)
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer.getvalue()
            )
            
            self.logger.info(f"Wrote {len(data)} records to {s3_key}")
            return {
                'success': True,
                'records': len(data),
                's3_key': s3_key,
                'size_bytes': buffer.tell()
            }
            
        except Exception as e:
            self.logger.error(f"Error writing parquet for {opr_date}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def read_day_from_parquet(self, opr_date: date) -> Optional[pa.Table]:
        """Read a day's LMP data from parquet in S3.
        
        Returns:
            PyArrow Table or None if not found
        """
        try:
            s3_key = self._get_parquet_key(opr_date)
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            content = response['Body'].read()
            
            buffer = io.BytesIO(content)
            table = pq.read_table(buffer)
            return table
            
        except self.s3_client.exceptions.NoSuchKey:
            return None
        except Exception as e:
            self.logger.error(f"Error reading parquet for {opr_date}: {str(e)}")
            return None
    
    def read_day_as_dicts(self, opr_date: date) -> List[Dict]:
        """Read a day's LMP data and return as list of dicts."""
        table = self.read_day_from_parquet(opr_date)
        if table is None:
            return []
        return table.to_pylist()
    
    def list_available_dates(self, year: int = None, month: int = None) -> List[date]:
        """List all dates that have parquet data available.
        
        Args:
            year: Optional filter by year
            month: Optional filter by month
        """
        try:
            prefix = self.parquet_prefix
            if year:
                prefix += f"year={year}/"
                if month:
                    prefix += f"month={month:02d}/"
            
            paginator = self.s3_client.get_paginator('list_objects_v2')
            dates = []
            
            for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
                if 'Contents' in page:
                    for obj in page['Contents']:
                        key = obj['Key']
                        if key.endswith('.parquet'):
                            filename = key.split('/')[-1]
                            date_str = filename.replace('.parquet', '')
                            try:
                                d = date.fromisoformat(date_str)
                                dates.append(d)
                            except ValueError:
                                pass
            
            return sorted(dates)
            
        except Exception as e:
            self.logger.error(f"Error listing parquet dates: {str(e)}")
            return []
    
    def check_date_exists(self, opr_date: date) -> bool:
        """Check if parquet file exists for a given date."""
        try:
            s3_key = self._get_parquet_key(opr_date)
            self.s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except:
            return False
    
    def get_node_data_for_date_range(
        self, 
        nodes: List[str], 
        start_date: date, 
        end_date: date
    ) -> List[Dict]:
        """Read data for specific nodes across a date range.
        
        This is used for detailed drill-down queries from the chatbot.
        """
        import pandas as pd
        
        all_data = []
        current = start_date
        
        while current <= end_date:
            table = self.read_day_from_parquet(current)
            if table is not None:
                df = table.to_pandas()
                filtered = df[df['node'].isin(nodes)]
                for _, row in filtered.iterrows():
                    all_data.append({
                        'opr_dt': current,
                        'node': row['node'],
                        'mw': row['mw'],
                        'opr_hr': row['opr_hr']
                    })
            current = date(current.year, current.month, current.day + 1) if current.day < 28 else self._next_day(current)
        
        return all_data
    
    def _next_day(self, d: date) -> date:
        """Get next day handling month/year boundaries."""
        from datetime import timedelta
        return d + timedelta(days=1)
