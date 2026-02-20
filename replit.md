# CAISO LMP Analysis Tool

## Overview

This Streamlit-based web application analyzes CAISO Day Ahead Locational Marginal Price (LMP) data. It offers interactive visualizations, data processing, and an AI-powered chatbot for natural language queries on electricity pricing. Users can upload CAISO CSV data to gain insights through both traditional analysis and conversational AI. The project aims to provide efficient and accurate electricity pricing analysis, leveraging cloud-native solutions for scalability and performance.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
- **Framework**: Streamlit for the user interface.
- **Visualization**: Plotly for interactive charts.
- **Layout**: Wide layout with a sidebar for uploads and a main area for results.
- **State Management**: Streamlit session state maintains data and application state across interactions.

### Backend
- **Data Processing**: `CAISODataProcessor` handles CSV parsing, validation, and cleaning, ensuring correct `OPR_HR` usage to avoid timezone errors.
- **Analytics**: `LMPAnalytics` provides core functions like cheapest hour identification, congestion analysis, and price statistics.
- **AI Integration**: `LMPChatbot` uses OpenAI's GPT models for natural language understanding, intent analysis, and generating structured analysis instructions.
- **Modular Design**: Separation of concerns across data processing, analytics, and chatbot modules.

### Data Processing Pipeline
- **Validation**: Checks for required columns and standardizes data formats.
- **Cleaning**: Parses datetimes and processes numeric price data.
- **Error Handling**: Robust logging for processing failures.

### Core Architectural Decisions
- **Cloud-Native Data Storage**: Utilizes MotherDuck (DuckDB cloud) for all analytics queries, with raw LMP data stored as Parquet files in AWS S3. This provides 10x faster queries, native S3 parquet support, and a unified SQL interface.
- **Hybrid Storage**: Raw LMP data (Parquet in S3) and aggregated summaries (MotherDuck) optimize for storage limits and query performance.
- **Pre-computed Aggregates**: Daily, monthly, and annual summary tables in MotherDuck accelerate dashboard queries.
- **Security**: Implemented input sanitization for node names, zone names, S3 bucket names, and parameterized queries to prevent SQL injection.
- **Dynamic Node Selection**: Supports both zone-based and search-based node analysis with autocomplete.

## External Dependencies

### AI Services
- **OpenAI API**: GPT-5 model for chatbot functionality and natural language processing.

### Data Processing Libraries
- **Pandas**: Core for data manipulation and analysis.
- **NumPy**: For numerical operations.
- **Plotly**: For interactive visualizations.

### Web Framework
- **Streamlit**: For building the web application.

### Data Storage
- **MotherDuck**: Cloud-native analytical database (DuckDB cloud).
- **AWS S3**: For storing raw Parquet data files.