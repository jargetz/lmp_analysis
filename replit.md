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
- **MotherDuck-First Storage**: All data lives in MotherDuck (DuckDB cloud). Node-level hourly data (296M rows, `node_hourly_lmp` table), zone-level hourly data (`zone_hourly_lmp`), pre-computed BX summaries (`bx_daily_summary`, `generator_bx_summary`), and node-zone mappings are all in MotherDuck.
- **S3 Parquet as Archive**: Raw LMP data still exists as Parquet files in S3 (1 GB, 741 files) but is no longer queried at runtime. All queries go through MotherDuck tables.
- **Migration Script**: `migrate_parquet_to_motherduck.py` handles bulk loading S3 parquet files into `node_hourly_lmp`. Supports incremental loading by year.
- **Pre-computed Aggregates**: Daily, monthly, and annual summary tables in MotherDuck accelerate dashboard queries.
- **Monthly Weighting**: Annual averages use calendar-day weighting: `sum(month_avg × calendar_days) / total_calendar_days`.
- **Three Averaging Methods**: (1) EIA load-weighted zone averages from `zone_hourly_lmp`, (2) Generator settlement from `generator_bx_summary` (TH_*_GEN-APND nodes), (3) Unweighted node averages from `bx_daily_summary`.
- **Node-Zone Mapping**: Uses CAISO AS Region Map files to assign nodes to zones. Logic: AS_NP15 ∩ AS_NP26 → NP15, AS_SP15 ∩ AS_NP26 → ZP26, AS_SP15 only → SP15. Stored in `node_zone_mapping` table (6,394 nodes: 3,206 NP15, 2,627 SP15, 561 ZP26).
- **APNode Mapping**: `node_apnode_mapping` table maps component nodes to aggregated pricing nodes (TH_NP15_GEN-APND: 698, TH_SP15_GEN-APND: 1,076, TH_ZP26_GEN-APND: 178).
- **CA Substations**: `ca_substations` table (3,261 rows) from DataBasin/HIFLD CA electric substations dataset (https://databasin.org/datasets/20502139197843f7b1b2751a427d9f68/). Fields: substation_id (primary key), substation_name (NOT unique — 214 duplicates), owner, status (3,199 Operational; 36 Low kV, 17 Proposed, 9 Closed), highest_kv (normalized — one source typo '33kV to 92Kv' corrected to '33kV to 92kV'), lat, lon. Loaded by `load_substations.py`.
- **Node-Substation Mapping**: `node_substation_mapping` table (11,978 rows) maps every pnode to its geographically nearest CA substation. Fields: pnode_id, node_type, substation_id, substation_name, owner, status, highest_kv, dist_km. GEN nodes (power plants) often match far-away substations — this is expected. LOAD nodes in CA typically match within a few km. Non-operational substations are flagged with ⚠ in the UI.
- **AB 617 Communities**: `ab617_communities` table (65 rows) loaded from CARB's live ArcGIS FeatureServer API (consistently nominated communities, 2023 list). Fields: community_name, lat, lon. Note: CARB's API has fields named backwards — `centroid_longtitude` holds lat values, `centroid_latitude` holds lon values. Used in Node Finder tab for filtering emitters by proximity (≤30 km).
- **Node Finder Feature**: `node_finder` subprocess query type finds (A) top-N cheapest B-hour nodes from `node_bx_monthly_summary` + `pnode_coordinates`, and (B) nodes geographically nearest to top-M GHG emitters from `facility_emissions`. Optionally filters emitters to those within 30 km of an AB 617 community. Returns cheapest[], nearest[], emitters[], summary{}, ab617_communities[]. Distance = Euclidean × 111 km/degree with cos(lat) correction on lon.
- **Node BX Monthly Summary**: `node_bx_monthly_summary` table (418K rows) pre-computes B4–B10 averages per node per month in a single scan. Used by the Node Selection tool and Node Map tab for fast lookups. Annual mode uses days_count-weighted average of monthly values. The `subprocess_query.py` `get_node_bx()` function reads from this table.
- **Node Map Tab**: Geographic PNODE price map using Plotly scatter_mapbox (carto-positron, no API key). Data joined from `node_bx_monthly_summary` + `pnode_coordinates` + `node_zone_mapping`. Color by zone (NP15=blue, SP15=orange, ZP26=green) or by price (RdYlGn diverging scale). Price distribution histogram shown above the map.
- **PNODE Coordinates**: `pnode_coordinates` table (11,978 rows) from CAISO pricemap markers CSV. Covers full Western Interconnect; 2,602 CA nodes overlap with LMP data and zone mappings.
- **Data Quality**: Dec 10, 2024 data in `node_hourly_lmp` was corrupted during ETL (all nodes at ~-$2,594/MWh). Corrected by replacing with data from the official CAISO CSV (DAM_LMP, v12). Always validate node data against `generator_bx_summary` TH apnode values as ground truth.
- **Security**: Input sanitization for node names, zone names, S3 bucket names, and parameterized queries to prevent SQL injection.
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