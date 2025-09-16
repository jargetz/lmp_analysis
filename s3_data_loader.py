import boto3
import os
import logging
import zipfile
import io
from typing import List, Dict, Any
from data_processor import CAISODataProcessor
from preprocessing import CAISOPreprocessor

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
    
    def download_and_process_file(self, s3_key: str) -> Dict[str, Any]:
        """Download a single zip file from S3 and process it"""
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
                    # Only process the LMP CSV files we care about
                    if file_name.endswith('LMP_DAM_LMP_v12.csv'):
                        with zip_ref.open(file_name) as csv_file:
                            content = csv_file.read().decode('utf-8')
                            
                            # Process and store in database
                            result = self.processor.process_csv_content_to_db(content, s3_key)
                            return {
                                'success': True,
                                'records_inserted': result.get('records_inserted', 0),
                                'file': s3_key
                            }
            
            return {'success': False, 'error': f'No LMP_DAM_LMP_v12.csv found in {s3_key}'}
            
        except Exception as e:
            logging.error(f"Error processing S3 file {s3_key}: {str(e)}")
            return {'success': False, 'error': str(e)}
    
    def load_all_data(self, progress_callback=None) -> Dict[str, Any]:
        """Load all CAISO data from S3 into the database and run preprocessing"""
        # Phase 1: Download and process raw data
        if progress_callback:
            progress_callback(0, 100, "Starting S3 data download and processing...")
        
        files = self.list_caiso_files()
        
        if not files:
            return {'success': False, 'error': 'No CAISO files found in S3 bucket'}
        
        # TEMPORARY: Limit to 1 file for testing to prevent credit burn
        files = files[:1]
        total_files = len(files)
        processed_files = 0
        skipped_files = 0
        total_records = 0
        errors = []
        
        # Process files (70% of progress) - simplified progress display
        for i, file_key in enumerate(files):
            progress_percent = int((i / total_files) * 70)
            if progress_callback:
                progress_callback(progress_percent, 100, f"Processing CAISO files... ({i+1}/{total_files})")
            
            result = self.download_and_process_file(file_key)
            
            if result['success']:
                if result.get('skipped'):
                    skipped_files += 1
                else:
                    processed_files += 1
                    total_records += result.get('records_inserted', 0)
            else:
                errors.append(f"{file_key}: {result.get('error', 'Unknown error')}")
        
        # Phase 2: Run preprocessing (B6/B8 calculations)
        if progress_callback:
            progress_callback(70, 100, "Starting B6/B8 preprocessing...")
        
        def preprocessing_progress(current, total, message):
            # Map preprocessing progress to 70-100% range
            preprocessing_percent = 70 + int((current / total) * 30)
            if progress_callback:
                progress_callback(preprocessing_percent, 100, f"Preprocessing: {message}")
        
        preprocessing_result = self.preprocessor.run_full_preprocessing(preprocessing_progress)
        
        # Combine results
        final_result = {
            'success': True,
            'total_files': total_files,
            'processed_files': processed_files,
            'skipped_files': skipped_files,
            'total_records': total_records,
            'errors': errors,
            'preprocessing': preprocessing_result
        }
        
        if progress_callback:
            if preprocessing_result.get('success'):
                progress_callback(100, 100, "✅ Data loading and preprocessing completed successfully!")
            else:
                progress_callback(100, 100, "⚠️ Data loaded but preprocessing had issues")
                final_result['errors'].append(f"Preprocessing error: {preprocessing_result.get('error', 'Unknown preprocessing error')}")
        
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