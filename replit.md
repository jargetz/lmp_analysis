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