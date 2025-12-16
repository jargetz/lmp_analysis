import boto3
import os
import logging
import zipfile
import io
import re
from datetime import date
from typing import List, Dict, Any, Optional
from data_processor import CAISODataProcessor
from preprocessing import CAISOPreprocessor
from bx_calculator import BXCalculator

class S3DataLoader:
    """Handles loading CAISO LMP data from S3 bucket"""
    
    def __init__(self):
        self.bucket_name = os.getenv('AWS_S3_BUCKET')
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        self.processor = CAISODataProcessor()
        self.preprocessor = CAISOPreprocessor()
        self.bx_calculator = BXCalculator()
    
    def _extract_date_from_filename(self, filename: str) -> Optional[date]:
        """Extract date from CAISO filename like 20240101_20240101_DAM_LMP..."""
        match = re.match(r'(\d{4})(\d{2})(\d{2})_', filename)
        if match:
            year, month, day = int(match.group(1)), int(match.group(2)), int(match.group(3))
            return date(year, month, day)
        return None
        
    def list_caiso_files(self) -> List[str]:
        """List all CAISO LMP zip files in the S3 bucket"""
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.bucket_name)
            files = []
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Filter for CAISO LMP files
                    if key.endswith('.zip') and 'DAM_LMP' in key:
                        files.append(key)
            
            return sorted(files)
        except Exception as e:
            logging.error(f"Error listing S3 files: {str(e)}")
            return []
    
    def check_file_already_processed(self, s3_key: str) -> bool:
        """Check if S3 file has already been processed"""
        try:
            # Simple check - see if we have any data from this source file
            query = "SELECT COUNT(*) as count FROM caiso.lmp_data WHERE source_file = %s LIMIT 1"
            result = self.processor.db.execute_query(query, (s3_key,))
            if result and len(result) > 0 and isinstance(result[0], dict):
                count_value = result[0].get('count', 0)
                return count_value > 0
            return False
        except:
            return False
    
    def download_and_process_file(self, s3_key: str, calculate_bx: bool = False) -> Dict[str, Any]:
        """Download a single zip file from S3 and process it.
        
        Args:
            s3_key: The S3 key for the zip file
            calculate_bx: If True, calculate BX for the day after loading
        """
        try:
            # Skip if already processed (prevent duplicates)
            if self.check_file_already_processed(s3_key):
                return {
                    'success': True,
                    'records_inserted': 0,
                    'file': s3_key,
                    'skipped': True,
                    'reason': 'Already processed'
                }
            
            # Download file from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            zip_content = response['Body'].read()
            
            # Process the zip file
            with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
                for file_name in zip_ref.namelist():
                    # Only process the main LMP CSV file (PRC_LMP_DAM_LMP)
                    # Skip MCC, MCE, MCL files as they contain component data
                    if 'PRC_LMP_DAM_LMP' in file_name and file_name.endswith('.csv'):
                        logging.info(f"Processing {file_name} from {s3_key}")
                        with zip_ref.open(file_name) as csv_file:
                            content = csv_file.read().decode('utf-8')
                            
                            # Process and store in database
                            result = self.processor.process_csv_content_to_db(content, s3_key)
                            records_inserted = result.get('records_inserted', 0)
                            
                            # Calculate BX for this day if requested
                            bx_result = None
                            if calculate_bx and records_inserted > 0:
                                file_date = self._extract_date_from_filename(s3_key)
                                if file_date:
                                    logging.info(f"Calculating BX for {file_date}")
                                    bx_result = self.bx_calculator.calculate_all_bx_for_date(file_date)
                            
                            return {
                                'success': True,
                                'records_inserted': records_inserted,
                                'file': s3_key,
                                'bx_calculated': bx_result
                            }
            
            return {'success': False, 'error': f'No PRC_LMP_DAM_LMP CSV found in {s3_key}'}
            
        except Exception as e:
            logging.error(f"Error processing S3 file {s3_key}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def load_all_data(
        self, 
        progress_callback=None, 
        batch_size=None, 
        calculate_bx: bool = True
    ) -> Dict[str, Any]:
        """Load all CAISO data from S3 into the database.
        
        Args:
            progress_callback: Callback function for progress updates
            batch_size: If set, only process this many unprocessed files per run (for resumability)
            calculate_bx: If True (default), calculate BX for each day after loading
        """
        # Phase 1: Download and process raw data
        if progress_callback:
            progress_callback(0, 100, "Starting S3 data download and processing...")
        
        files = self.list_caiso_files()
        
        if not files:
            return {'success': False, 'error': 'No CAISO files found in S3 bucket'}
        
        # Filter to unprocessed files only for resumable loading
        if batch_size:
            unprocessed_files = [f for f in files if not self.check_file_already_processed(f)]
            if not unprocessed_files:
                return {'success': True, 'message': 'All files already processed!', 
                       'total_files': len(files), 'processed_files': 0, 'skipped_files': len(files)}
            files = unprocessed_files[:batch_size]
        
        total_files = len(files)
        processed_files = 0
        skipped_files = 0
        total_records = 0
        bx_calculated_days = 0
        errors = []
        
        # Process files with optional BX calculation per day
        for i, file_key in enumerate(files):
            progress_percent = int((i / total_files) * 100)
            bx_status = " + BX" if calculate_bx else ""
            if progress_callback:
                progress_callback(progress_percent, 100, f"Processing{bx_status}... ({i+1}/{total_files})")
            
            result = self.download_and_process_file(file_key, calculate_bx=calculate_bx)
            
            if result['success']:
                if result.get('skipped'):
                    skipped_files += 1
                else:
                    processed_files += 1
                    total_records += result.get('records_inserted', 0)
                    if result.get('bx_calculated'):
                        bx_calculated_days += 1
            else:
                errors.append(f"{file_key}: {result.get('error', 'Unknown error')}")
        
        # Combine results
        final_result = {
            'success': True,
            'total_files': total_files,
            'processed_files': processed_files,
            'skipped_files': skipped_files,
            'total_records': total_records,
            'bx_calculated_days': bx_calculated_days,
            'errors': errors
        }
        
        if progress_callback:
            progress_callback(100, 100, f"Done! {processed_files} files, {total_records:,} records")
        
        return final_result
    
    def check_data_freshness(self) -> Dict[str, Any]:
        """Check if database has recent data or needs refresh from S3"""
        try:
            summary = self.processor.get_data_summary_from_db()
            s3_files = self.list_caiso_files()
            
            return {
                'db_has_data': summary and summary.get('total_records', 0) > 0,
                'db_records': summary.get('total_records', 0) if summary else 0,
                's3_files_available': len(s3_files),
                'latest_db_date': summary.get('latest_date') if summary else None,
                's3_files': s3_files[:5]  # Show first 5 files as sample
            }
        except Exception as e:
            logging.error(f"Error checking data freshness: {str(e)}")
            return {'error': str(e)}