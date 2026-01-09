-- Lean schema for CAISO LMP data
-- Run this after recreating the database

CREATE SCHEMA IF NOT EXISTS caiso;

-- Main LMP data table - lean version with only essential columns
CREATE TABLE IF NOT EXISTS caiso.lmp_data (
    id SERIAL PRIMARY KEY,
    node VARCHAR(100) NOT NULL,
    mw NUMERIC(12,4) NOT NULL,
    opr_dt DATE NOT NULL,
    opr_hr INTEGER NOT NULL,
    source_file VARCHAR(255)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_lmp_node ON caiso.lmp_data(node);
CREATE INDEX IF NOT EXISTS idx_lmp_opr_dt ON caiso.lmp_data(opr_dt);
CREATE INDEX IF NOT EXISTS idx_lmp_opr_hr ON caiso.lmp_data(opr_hr);
CREATE INDEX IF NOT EXISTS idx_lmp_opr_dt_hr ON caiso.lmp_data(opr_dt, opr_hr);
CREATE INDEX IF NOT EXISTS idx_lmp_source ON caiso.lmp_data(source_file);

-- Unique constraint to prevent duplicates
CREATE UNIQUE INDEX IF NOT EXISTS idx_lmp_unique 
ON caiso.lmp_data(node, opr_dt, opr_hr);

-- Node zone mapping table
CREATE TABLE IF NOT EXISTS caiso.node_zone_mapping (
    id SERIAL PRIMARY KEY,
    pnode_id VARCHAR(100) NOT NULL UNIQUE,
    zone VARCHAR(20)
);

CREATE INDEX IF NOT EXISTS idx_node_zone_pnode ON caiso.node_zone_mapping(pnode_id);
CREATE INDEX IF NOT EXISTS idx_node_zone_zone ON caiso.node_zone_mapping(zone);

-- BX daily summary table
CREATE TABLE IF NOT EXISTS caiso.bx_daily_summary (
    id SERIAL PRIMARY KEY,
    opr_dt DATE NOT NULL,
    node VARCHAR(100) NOT NULL,
    bx_type INTEGER NOT NULL,
    avg_price NUMERIC(12,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(opr_dt, node, bx_type)
);

CREATE INDEX IF NOT EXISTS idx_bx_daily_dt ON caiso.bx_daily_summary(opr_dt);
CREATE INDEX IF NOT EXISTS idx_bx_daily_node ON caiso.bx_daily_summary(node);
CREATE INDEX IF NOT EXISTS idx_bx_daily_type ON caiso.bx_daily_summary(bx_type);

-- BX monthly summary table (for fast dashboard queries)
CREATE TABLE IF NOT EXISTS caiso.bx_monthly_summary (
    id SERIAL PRIMARY KEY,
    year_month VARCHAR(7) NOT NULL,
    zone VARCHAR(20),
    bx_type INTEGER NOT NULL,
    avg_price NUMERIC(12,4),
    node_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(year_month, zone, bx_type)
);

-- BX annual summary table (for fast dashboard queries)
CREATE TABLE IF NOT EXISTS caiso.bx_annual_summary (
    id SERIAL PRIMARY KEY,
    year INTEGER NOT NULL,
    zone VARCHAR(20),
    bx_type INTEGER NOT NULL,
    avg_price NUMERIC(12,4),
    node_count INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(year, zone, bx_type)
);
