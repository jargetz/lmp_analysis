# CAISO LMP Analysis Tool

## Overview

This is a Streamlit-based web application for analyzing CAISO (California Independent System Operator) Day Ahead Locational Marginal Price (LMP) data. The tool provides data processing capabilities, interactive visualizations, and an AI-powered chatbot for natural language querying of electricity pricing data. Users can upload ZIP files containing CAISO CSV data, perform various analytics operations, and get insights through both traditional data analysis and conversational AI interface.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Framework**: Streamlit for web interface
- **Visualization**: Plotly (Express and Graph Objects) for interactive charts and graphs
- **Layout**: Wide layout configuration with sidebar for file uploads and main area for analysis results
- **State Management**: Streamlit session state for maintaining data, processor instances, analytics, chatbot, and chat history across user interactions

### Backend Architecture
- **Data Processing Layer**: `CAISODataProcessor` class handles CSV parsing, validation, and data cleaning
- **Analytics Layer**: `LMPAnalytics` class provides core analytical functions like finding cheapest hours, congestion analysis, and price statistics
- **AI Layer**: `LMPChatbot` class integrates with OpenAI's GPT models for natural language processing and query interpretation
- **Modular Design**: Separate modules for data processing, analytics, and chatbot functionality to maintain separation of concerns

### Data Processing Pipeline
- **Input Validation**: Validates CAISO data format by checking for required columns (INTERVALSTARTTIME_GMT, NODE, MW)
- **Data Standardization**: Standardizes column names and formats across different CAISO file versions
- **Datetime Parsing**: Converts timestamp strings to datetime objects for time-based analysis
- **Numeric Cleaning**: Processes MW price data and optional components (MCC, MLC, POS for congestion, loss, and position)
- **Error Handling**: Comprehensive error handling with logging for data processing failures

### AI Integration
- **OpenAI API**: Uses GPT-5 model for natural language understanding and response generation
- **Intent Analysis**: Analyzes user questions to determine appropriate analysis type (cheapest_hours, price_percentile, congestion_analysis, etc.)
- **Context Awareness**: Incorporates data context and available columns when processing queries
- **Structured Responses**: Returns JSON-formatted analysis instructions that are executed by the analytics engine

## External Dependencies

### AI Services
- **OpenAI API**: GPT-5 model for natural language processing and chatbot functionality
- **API Key Management**: Environment variable-based configuration for OpenAI API key

### Data Processing Libraries
- **Pandas**: Core data manipulation and analysis
- **NumPy**: Numerical computing and array operations
- **Plotly**: Interactive visualization components (Express and Graph Objects)

### Web Framework
- **Streamlit**: Web application framework for the user interface
- **File Handling**: Built-in support for ZIP file uploads and CSV processing

### Development Tools
- **Logging**: Python logging module for error tracking and debugging
- **IO Operations**: String and file I/O operations for data processing
- **DateTime**: Date and time manipulation for temporal analysis

### Data Format Support
- **CSV Processing**: Handles CAISO-specific CSV formats with various column naming conventions
- **ZIP File Support**: Processes multiple ZIP files containing CSV data
- **Time Zone Handling**: GMT timestamp processing for electricity market data

### Testing
- **Framework**: pytest for baseline testing
- **Test Coverage**: Core analytics methods (peak/off-peak, price statistics, cheapest hours, hourly averages)
- **Test Strategy**: Lightweight baseline tests against real database data, designed for manual runs during development
- **Run Tests**: `pytest test_analytics_baseline.py -v`
- **Philosophy**: Minimal but useful - catches breaking changes without slowing iteration

## Data Loading

- **Source**: 312 ZIP files stored in AWS S3 bucket (full year of CAISO Day Ahead LMP data)
- **Loading Strategy**: Resumable batch processing to handle platform execution time limits
- **Batch Size**: Configurable (default 20 files per run)
- **Progress Tracking**: Automatic duplicate detection via `source_file` column
- **How to Load**: Run `python3 load_full_year.py` multiple times until complete
- **Full Guide**: See `DATALOAD_GUIDE.md` for detailed instructions

## Recent Changes

### January 7, 2026
- **Pre-computed Summary Tables**: Added `bx_monthly_summary` and `bx_annual_summary` tables for fast dashboard queries
  - Annual summaries: ~1.3M rows/year vs 35M daily rows (90% reduction)
  - Post-import aggregation runs automatically after S3 data load completes
  - Methods: `aggregate_monthly_summaries()`, `aggregate_annual_summaries()`, `run_post_import_aggregation()`
- **Dashboard Time Period Selector**: Replaced date range with Annual/Monthly selector
  - Annual view uses pre-computed `get_annual_bx_average()` for fast queries
  - Monthly view uses daily aggregation with date range
  - Dynamic year dropdown via `get_available_years()` method
- **Node Selection Mode**: Added toggle between "By Zone" and "By Node Selection" analysis modes
  - Zone mode: Filter by NP15, SP15, ZP26 zones (existing functionality)
  - Node mode: Search-based node selection with autocomplete (new)
- **BX Calculator Optimization**: Simplified to only store daily summaries (~112k rows/day vs ~1.1M)
  - Removed `bx_hours` table usage for better performance
  - Queries now support filtering by both zones and specific node lists
- **Node Search**: Added `search_nodes()` method for efficient server-side search
  - Handles 16k+ nodes without loading them all at once
  - Returns matching nodes with ILIKE pattern matching

### December 10, 2025
- **Dashboard-First UI**: Restructured app.py into two tabs: Dashboard (primary) and AI Assistant
- **Node-to-Zone Mapping**: Created `node_zone_mapping.py` module to map PNODE_ID to zones (NP15, SP15, ZP26)
  - Loaded 5,593 node mappings (1,694 mapped to zones, 3,899 unmapped preserved for visibility)
  - Supports custom file paths for refreshing mappings
- **BX Calculator**: Created `bx_calculator.py` with support for B4-B10 (cheapest X hours analysis)
  - Unified table approach with `bx_type` column instead of separate B6/B8 tables
  - Efficient single-query-per-date design
  - Query methods: `get_bx_average()`, `get_bx_trend()` with zone and node filtering
- **Dashboard Features**: Zone filter, BX selector, summary statistics cards, hourly price chart

### October 25, 2025  
- Fixed peak hour definition: Changed from 16-21 (4-9 PM) to 0-6 (midnight-6 AM) based on actual CAISO market data showing higher prices during early morning hours
- Added baseline testing suite with pytest covering critical analytics methods
- Tests validate core functionality against actual database data for regression detection
- Implemented resumable batch loading system for processing 312 ZIP files from S3
- Fixed timestamp column handling to maintain `interval_start_time_gmt` for database NOT NULL constraint
- Created comprehensive data loading guide with batch processing instructions