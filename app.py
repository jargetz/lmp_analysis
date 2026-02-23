"""
CAISO LMP Analysis Tool - Main Application

This is the main Streamlit application with two primary views:
1. Dashboard - Summary statistics, BX analysis, and zone filtering
2. AI Assistant - Natural language queries about the data

The dashboard is the primary interface for exploring LMP data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta, date
import os

from data_processor import CAISODataProcessor
from analytics import LMPAnalytics, get_registered_analytics
from chatbot import LMPChatbot
from node_zone_mapping import NodeZoneMapper, VALID_ZONES
from bx_calculator import BXCalculator, SUPPORTED_BX_VALUES
from charts import (
    create_hourly_price_chart,
    create_bx_trend_chart,
    create_zone_comparison_bar,
    create_top_nodes_bar,
    create_empty_chart,
    create_zone_hourly_chart,
    create_node_hourly_chart,
    create_zone_bx_trend_chart,
    create_node_bx_trend_chart,
    create_node_box_plot,
    create_month_hour_heatmap,
    create_node_hourly_lines_chart,
    create_node_month_hour_heatmap,
    create_8760_heatmap
)

def main():
    st.set_page_config(
        page_title="CAISO LMP Analysis Tool",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ CAISO LMP Analysis Tool")
    st.markdown("Analyze electricity pricing with AI-powered insights using comprehensive CAISO Day Ahead LMP data.")
    
    # Initialize session state (database-backed)
    if 'processor' not in st.session_state:
        st.session_state.processor = CAISODataProcessor()
    if 'analytics' not in st.session_state:
        st.session_state.analytics = LMPAnalytics()
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = LMPChatbot()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'init_data' not in st.session_state:
        import subprocess
        import json as json_mod
        try:
            r = subprocess.run(['python3', 'subprocess_query.py', 'init_dashboard'],
                               capture_output=True, text=True, timeout=30)
            if r.returncode == 0 and r.stdout.strip():
                init_data = json_mod.loads(r.stdout.strip())
                st.session_state.db_summary = init_data.get('data_summary', {})
                st.session_state.init_years = init_data.get('available_years', [2024])
                st.session_state.init_nodes = init_data.get('all_nodes', [])
                st.session_state.individual_nodes = init_data.get('individual_nodes', [])
                st.session_state.zone_years = init_data.get('zone_years', [2024])
            else:
                st.session_state.db_summary = {}
                st.session_state.init_years = [2024]
                st.session_state.init_nodes = []
                st.session_state.individual_nodes = []
                st.session_state.zone_years = [2024]
        except Exception:
            st.session_state.db_summary = {}
            st.session_state.init_years = [2024]
            st.session_state.init_nodes = []
            st.session_state.individual_nodes = []
            st.session_state.zone_years = [2024]
        st.session_state.init_data = True
    
    with st.sidebar:
        st.header("Data Status")
        
        summary = st.session_state.db_summary
        if summary and summary.get('total_records', 0) > 0:
            st.success("Data loaded and ready")
            days_loaded = summary.get('total_records', 0) // 28
            st.metric("Days Loaded", f"{days_loaded}")
            if summary.get('earliest_date') and summary.get('latest_date'):
                earliest = summary['earliest_date']
                latest = summary['latest_date']
                if hasattr(earliest, 'strftime'):
                    earliest = earliest.strftime('%Y-%m-%d')
                if hasattr(latest, 'strftime'):
                    latest = latest.strftime('%Y-%m-%d')
                st.caption(f"{earliest} to {latest}")
            st.caption("Zone aggregates in MotherDuck, raw data in S3 Parquet")
            st.session_state.data_loaded = True
            
            st.subheader("Data Details")
            st.metric("Unique Zones", summary.get('unique_nodes', 0))
            if summary.get('earliest_date') and summary.get('latest_date'):
                st.markdown("**Date Range**")
                st.markdown(f"Start: {summary['earliest_date']}")
                st.markdown(f"End: {summary['latest_date']}")
        else:
            st.warning("No data available")
            st.caption("MotherDuck database may be loading...")
            st.session_state.data_loaded = False
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("Connecting to MotherDuck database... If data doesn't appear, please refresh the page.")
        
        st.header("Sample Questions")
        st.markdown("""
        Once data loads, you can ask the AI Assistant questions like:
        - What are the 10 cheapest hours at node SLAP_PGE2?
        - Show me the nodes with the lowest 10% of prices (B10)
        - What are the average prices by hour of day?
        - Show me the B6 and B8 hours (cheapest 6 and 8 hours) for each node
        """)
        
    else:
        tab_dashboard, tab_methodology, tab_ai = st.tabs(["📊 Dashboard", "📋 Methodology & Data", "💬 AI Assistant"])
        
        # =====================================================================
        # DASHBOARD TAB - Primary interface for BX analysis with zone filtering
        # =====================================================================
        with tab_dashboard:
            render_dashboard_tab()
        
        # =====================================================================
        # METHODOLOGY & DATA TAB
        # =====================================================================
        with tab_methodology:
            render_methodology_tab()
        
        # =====================================================================
        # AI ASSISTANT TAB - Natural language queries (existing chatbot)
        # =====================================================================
        with tab_ai:
            render_ai_assistant_tab()


def render_dashboard_tab():
    """
    Render the main dashboard with BX analysis and zone filtering.
    
    This is the primary interface for exploring LMP data.
    """
    if 'dashboard_initialized' not in st.session_state:
        bx_calc_init = BXCalculator()
        st.session_state.bx_calc = bx_calc_init
        st.session_state.available_years = st.session_state.get('init_years', [2024])
        st.session_state.parquet_years = st.session_state.get('init_years', [2024])
        st.session_state.all_nodes = st.session_state.get('individual_nodes', [])
        st.session_state.dashboard_initialized = True
    
    st.header("LMP Dashboard")
    st.markdown("Analyze electricity pricing by zone or specific nodes")
    
    # Filter Panel
    st.subheader("Filters")
    
    # Analysis mode toggle
    analysis_mode = st.radio(
        "Analysis Mode",
        options=["By Zone", "By Node Selection"],
        horizontal=True,
        help="Choose to analyze by zone or select specific nodes"
    )
    
    # Initialize filter variables
    selected_zone = None
    selected_nodes = []
    
    if analysis_mode == "By Zone":
        filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
        
        with filter_col1:
            st.markdown("**Zone Comparison**")
            st.caption("Showing NP15, SP15, ZP26, and Overall averages")
    else:
        # Node mode: wider layout for node selection
        node_col, options_col = st.columns([3, 1])
        
        with node_col:
            # Initialize selected nodes from session state
            if 'selected_nodes_list' not in st.session_state:
                st.session_state.selected_nodes_list = []
            
            # Quick add by prefix
            prefix_col, add_col = st.columns([4, 1])
            with prefix_col:
                prefix = st.text_input(
                    "Add nodes by prefix",
                    placeholder="e.g., PGE, SCE, SLAP",
                    help="Type a prefix and click Add to select all matching nodes",
                    key="node_prefix"
                )
            with add_col:
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Add All", key="add_prefix"):
                    if prefix and len(prefix) >= 2:
                        matching = [n for n in st.session_state.all_nodes if n.upper().startswith(prefix.upper())]
                        if matching:
                            current = st.session_state.get('selected_nodes_list', [])
                            updated = list(set(current + matching))
                            st.session_state.selected_nodes_list = updated
                            st.rerun()
            
            # Multiselect with wider display
            selected_nodes = st.multiselect(
                "Selected Nodes",
                options=st.session_state.all_nodes,
                default=st.session_state.selected_nodes_list,
                placeholder="Type to search nodes...",
                help="Select individual nodes or use prefix above to add many at once",
                key="node_multiselect"
            )
            
            # Sync selection back to session state (ensure unique)
            st.session_state.selected_nodes_list = list(dict.fromkeys(selected_nodes))
            
            # Show count and clear button
            if selected_nodes:
                st.caption(f"{len(selected_nodes)} nodes selected")
                if st.button("Clear All", key="clear_nodes"):
                    st.session_state.selected_nodes_list = []
                    st.rerun()
        
        # Smaller options column for BX/Year selectors
        filter_col1, filter_col2, filter_col3, filter_col4 = options_col, options_col, options_col, options_col
    
    # Common filters (BX, Time Period, Year)
    if analysis_mode == "By Zone":
        with filter_col2:
            selected_bx = st.selectbox(
                "BX Hours",
                options=SUPPORTED_BX_VALUES,
                index=4,
                format_func=lambda x: f"B{x} (Cheapest {x} hours)",
                key="zone_bx",
                help="Number of cheapest hours to analyze"
            )
        
        with filter_col3:
            time_period = st.selectbox(
                "Time Period",
                options=["Annual", "Monthly"],
                key="zone_time_period",
                help="Choose annual or monthly view"
            )
        
        with filter_col4:
            zone_years = [y for y in st.session_state.get('zone_years', [2024]) if y <= 2024]
            if not zone_years:
                zone_years = [2024]
            
            if time_period == "Annual":
                selected_year = st.selectbox(
                    "Year",
                    options=zone_years,
                    key="zone_annual_year",
                    help="Select year"
                )
                selected_month = None
            else:
                selected_year = st.selectbox(
                    "Year",
                    options=zone_years,
                    key="monthly_year",
                    help="Select year"
                )
                month_options = ["January", "February", "March", "April", "May", "June", 
                               "July", "August", "September", "October", "November", "December"]
                selected_month_name = st.selectbox(
                    "Month",
                    options=month_options,
                    help="Select month"
                )
                selected_month = month_options.index(selected_month_name) + 1
    else:
        # Node mode: options in the side column
        # Use parquet years (only years with node data)
        parquet_years = st.session_state.get('parquet_years', [2024])
        
        with options_col:
            selected_bx = st.selectbox(
                "BX Hours",
                options=SUPPORTED_BX_VALUES,
                index=4,
                format_func=lambda x: f"B{x}",
                key="node_bx",
                help="Number of cheapest hours to analyze"
            )
            
            time_period = st.selectbox(
                "Time Period",
                options=["Annual", "Monthly"],
                key="node_time_period",
                help="Choose annual or monthly view"
            )
            
            if time_period == "Annual":
                # Preserve year selection when nodes change
                default_year_idx = 0
                if 'last_node_year' in st.session_state and st.session_state.last_node_year in parquet_years:
                    default_year_idx = parquet_years.index(st.session_state.last_node_year)
                
                selected_year = st.selectbox(
                    "Year",
                    options=parquet_years,
                    index=default_year_idx,
                    key="node_annual_year",
                    help="Select year (only years with node data)"
                )
                st.session_state.last_node_year = selected_year
                selected_month = None
            else:
                selected_year = st.selectbox(
                    "Year",
                    options=parquet_years,
                    key="monthly_year_node",
                    help="Select year"
                )
                month_options = ["January", "February", "March", "April", "May", "June", 
                               "July", "August", "September", "October", "November", "December"]
                selected_month_name = st.selectbox(
                    "Month",
                    options=month_options,
                    key="monthly_month_node",
                    help="Select month"
                )
                selected_month = month_options.index(selected_month_name) + 1
    
    st.divider()
    
    # BX Price Summary - title reflects the selection
    if time_period == "Annual":
        period_label = str(selected_year)
    else:
        month_names = ["January", "February", "March", "April", "May", "June", 
                       "July", "August", "September", "October", "November", "December"]
        period_label = f"{month_names[selected_month-1]} {selected_year}"
    st.subheader(f"B{selected_bx} Price Summary ({period_label})")
    
    try:
        # Use cached BXCalculator (preloaded at startup)
        bx_calc = st.session_state.bx_calc
        
        if analysis_mode == "By Zone":
            load_cache_key = f"load_stats_{selected_bx}_{selected_year}_{time_period}_{selected_month}"
            if load_cache_key not in st.session_state:
                st.session_state[load_cache_key] = bx_calc.get_all_zones_load_weighted_bx(
                    bx=selected_bx,
                    year=selected_year,
                    time_period=time_period,
                    month=selected_month
                )
            load_stats = st.session_state[load_cache_key]
            
            st.markdown("**EIA Load-Weighted Zone Average** (monthly weighted)")
            zone_cols = st.columns(4)
            zone_order = ['NP15', 'SP15', 'ZP26', 'Overall']
            
            for col, zone_name in zip(zone_cols, zone_order):
                with col:
                    stats = load_stats.get(zone_name, {})
                    if stats.get('success') and stats.get('avg_price') is not None:
                        st.metric(
                            f"{zone_name}",
                            f"${stats['avg_price']:.2f}/MWh",
                            help=f"BX of CAISO EIA zone price, monthly weighted. Days with data: {stats.get('day_count', 0)}"
                        )
                    else:
                        st.metric(zone_name, "N/A")
            
            gen_cache_key = f"gen_stats_{selected_bx}_{selected_year}"
            if gen_cache_key not in st.session_state:
                st.session_state[gen_cache_key] = bx_calc.get_generator_bx_average(
                    bx=selected_bx,
                    year=selected_year
                )
            gen_stats = st.session_state[gen_cache_key]
            
            if gen_stats.get('success') and gen_stats.get('zones'):
                st.markdown("**Generator Settlement Node Average** (unweighted, monthly weighted)")
                gen_cols = st.columns(3)
                gen_zone_order = ['NP15', 'SP15', 'ZP26']
                
                for col, zone_name in zip(gen_cols, gen_zone_order):
                    with col:
                        zone_data = gen_stats['zones'].get(zone_name, {})
                        if zone_data.get('avg_price') is not None:
                            st.metric(
                                f"{zone_name}",
                                f"${zone_data['avg_price']:.2f}/MWh",
                                help=f"TH_{zone_name}_GEN-APND (unweighted). Days: {zone_data.get('day_count', 0)}"
                            )
                        else:
                            st.metric(f"{zone_name}", "N/A")
            
            node_cache_key = f"node_avg_stats_{selected_bx}_{selected_year}_{time_period}_{selected_month}"
            if node_cache_key not in st.session_state:
                st.session_state[node_cache_key] = bx_calc.get_all_zones_bx_average(
                    bx=selected_bx,
                    year=selected_year,
                    time_period=time_period,
                    month=selected_month
                )
            node_stats = st.session_state[node_cache_key]
            
            with st.expander("Node Average (unweighted)", expanded=False):
                st.caption("Simple average of individual node BX values within each zone — not load-weighted")
                node_cols = st.columns(4)
                for col, zone_name in zip(node_cols, zone_order):
                    with col:
                        stats = node_stats.get(zone_name, {})
                        if stats.get('success') and stats.get('avg_price') is not None:
                            st.metric(
                                f"{zone_name}",
                                f"${stats['avg_price']:.2f}/MWh",
                                help=f"Unweighted avg of node BX values. Days: {stats.get('day_count', 0)}"
                            )
                        else:
                            st.metric(zone_name, "N/A")
            
            st.subheader("Averages - Day Ahead LMP")
            heatmap_zones = ['Overall', 'NP15', 'SP15', 'ZP26']
            heatmap_tabs = st.tabs(heatmap_zones)
            
            all_heatmap_key = f"all_heatmaps_{selected_year}"
            if all_heatmap_key not in st.session_state:
                st.session_state[all_heatmap_key] = bx_calc.get_all_zones_month_hour(year=selected_year)
            all_heatmaps = st.session_state[all_heatmap_key]
            
            for tab, zone_name in zip(heatmap_tabs, heatmap_zones):
                with tab:
                    heatmap_data = all_heatmaps.get(zone_name, [])
                    if heatmap_data:
                        fig = create_month_hour_heatmap(heatmap_data, zone=zone_name)
                        st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'zone_heatmap_{zone_name}_{selected_year}'}})
                    else:
                        st.info("No heatmap data available yet.")
            
            # BX trend chart by zone (cached)
            bx_trend_cache_key = f"bx_trend_zone_{selected_bx}_{selected_year}"
            if bx_trend_cache_key not in st.session_state:
                st.session_state[bx_trend_cache_key] = bx_calc.get_bx_trend_by_zone(
                    bx=selected_bx,
                    year=selected_year,
                    aggregation='monthly'
                )
            zone_trend_data = st.session_state[bx_trend_cache_key]
            
            if any(zone_trend_data.get(z) for z in ['NP15', 'SP15', 'ZP26', 'Overall']):
                fig = create_zone_bx_trend_chart(zone_trend_data, bx_type=selected_bx)
                st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'zone_B{selected_bx}_trend_{selected_year}'}})
        
        elif analysis_mode == "By Node Selection":
            # Node selection mode - show stats for selected nodes from parquet
            
            # Simple diagnostic test using subprocess to avoid Streamlit threading issues
            with st.expander("Database Diagnostic", expanded=False):
                if st.button("Test Simple Query"):
                    import subprocess
                    import time
                    st.write("Testing MotherDuck via subprocess...")
                    try:
                        start = time.time()
                        result = subprocess.run(
                            ['python3', '-c', '''
import os, duckdb
token = os.getenv('MOTHERDUCK_TOKEN')
conn = duckdb.connect(f'md:?motherduck_token={token}')
conn.execute("SET enable_progress_bar = false")
result = conn.execute("SELECT 1 as test").fetchall()
print(f"Basic: {result}")
conn.close()
'''],
                            capture_output=True, text=True, timeout=10
                        )
                        if result.returncode == 0:
                            st.success(f"Basic query: {result.stdout.strip()} ({time.time()-start:.2f}s)")
                        else:
                            st.error(f"Failed: {result.stderr}")
                    except subprocess.TimeoutExpired:
                        st.error("Query timed out after 10s")
                    except Exception as e:
                        st.error(f"Error: {e}")
            
            if not selected_nodes:
                st.info("Select one or more nodes above to see BX statistics.")
            else:
                import subprocess
                import json
                
                def run_subprocess_query(query_type, *args, timeout=60):
                    """Run MotherDuck query in subprocess to avoid Streamlit blocking"""
                    cmd = ['python3', 'subprocess_query.py', query_type] + [str(a) if not isinstance(a, list) else json.dumps(a) for a in args]
                    try:
                        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
                        if result.returncode == 0:
                            return json.loads(result.stdout)
                        return {'success': False, 'error': result.stderr}
                    except subprocess.TimeoutExpired:
                        return {'success': False, 'error': f'Query timed out after {timeout}s'}
                    except Exception as e:
                        return {'success': False, 'error': str(e)}
                
                # Fetch BX stats using subprocess
                with st.spinner(f"Computing B{selected_bx} for {len(selected_nodes)} node(s)... (this may take 10-30 seconds)"):
                    bx_stats = run_subprocess_query('node_bx', selected_bx, selected_nodes, selected_year, timeout=90)
                
                if bx_stats.get('error'):
                    st.error(f"Query error: {bx_stats.get('error')}")
                elif bx_stats.get('success') and bx_stats.get('avg_price'):
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric(f"B{selected_bx} Overall Avg", f"${bx_stats['avg_price']:.2f}/MWh")
                    with stat_col2:
                        st.metric("Min", f"${bx_stats['min_price']:.2f}/MWh" if bx_stats.get('min_price') else "N/A")
                    with stat_col3:
                        st.metric("Max", f"${bx_stats['max_price']:.2f}/MWh" if bx_stats.get('max_price') else "N/A")
                    with stat_col4:
                        st.metric("Nodes", f"{bx_stats.get('node_count', 0):,}")
                    
                    # Per-node BX stats table
                    per_node = bx_stats.get('per_node', {})
                    per_node_hours = bx_stats.get('per_node_hours', {})
                    if per_node and isinstance(per_node, dict):
                        st.subheader(f"B{selected_bx} by Node")
                        node_stats_df = pd.DataFrame([
                            {
                                'Node': node, 
                                f'B{selected_bx} Avg ($/MWh)': round(price, 2),
                                f'Most Common B{selected_bx} Hours': ', '.join(map(str, per_node_hours.get(node, []))) if isinstance(per_node_hours, dict) else ''
                            }
                            for node, price in sorted(per_node.items(), key=lambda x: x[1])
                        ])
                        st.dataframe(node_stats_df, use_container_width=True, hide_index=True)
                    
                    # Note about DST hour 25
                    st.caption("Note: Hour 25 (DST transition) is excluded from charts")
                    
                    # Node price heatmap (month x hour) - using subprocess
                    with st.spinner("Loading price heatmap..."):
                        heatmap_result = run_subprocess_query('heatmap', selected_nodes, selected_year, timeout=90)
                    
                    if isinstance(heatmap_result, list) and heatmap_result:
                        fig, clipping_info = create_node_month_hour_heatmap(heatmap_result, title=f'Price Heatmap ({len(selected_nodes)} nodes, {selected_year})')
                        st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'node_heatmap_{selected_year}'}})
                        if clipping_info:
                            parts = []
                            if clipping_info['clipped_below']:
                                parts.append(f"values below ${clipping_info['zmin']:.0f} (actual min: ${clipping_info['actual_min']:.0f})")
                            if clipping_info['clipped_above']:
                                parts.append(f"values above ${clipping_info['zmax']:.0f} (actual max: ${clipping_info['actual_max']:.0f})")
                            st.caption(f"Color scale clipped: {'; '.join(parts)}. Hover for exact values.")
                    elif isinstance(heatmap_result, dict) and heatmap_result.get('error'):
                        st.warning(f"Heatmap unavailable: {heatmap_result.get('error')}")
                    
                    # Hourly price chart - AVERAGE across all nodes - using subprocess
                    with st.spinner("Loading average hourly prices..."):
                        hourly_result = run_subprocess_query('hourly_avg', selected_nodes, selected_year, timeout=90)
                    
                    if isinstance(hourly_result, list) and hourly_result:
                        fig = create_node_hourly_chart(hourly_result, title=f'Hourly Price Average ({len(selected_nodes)} nodes, {selected_year})')
                        st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'node_hourly_avg_{selected_year}'}})
                    elif isinstance(hourly_result, dict) and hourly_result.get('error'):
                        st.warning(f"Hourly chart unavailable: {hourly_result.get('error')}")
                    else:
                        st.info(f"Per-node hourly chart available for 25 or fewer nodes (currently {len(selected_nodes)} selected)")
                    
                    # BX trend chart - using subprocess (limit to 10 nodes for performance)
                    if len(selected_nodes) <= 10:
                        with st.spinner("Loading BX trend per node... (this may take 30-60 seconds)"):
                            trend_result = run_subprocess_query('bx_trend', selected_bx, selected_nodes, selected_year, timeout=120)
                        
                        if isinstance(trend_result, list) and trend_result:
                            fig = create_node_bx_trend_chart(trend_result, bx_type=selected_bx)
                            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'node_B{selected_bx}_trend_{selected_year}'}})
                        elif isinstance(trend_result, dict) and trend_result.get('error'):
                            st.warning(f"BX trend unavailable: {trend_result.get('error')}")
                    else:
                        st.info(f"BX trend chart available for 10 or fewer nodes (currently {len(selected_nodes)} selected)")
                    
                    # Box plot for node comparison (outlier detection)
                    if len(selected_nodes) > 1:
                        with st.spinner("Loading price distribution..."):
                            box_result = run_subprocess_query('box_stats', selected_bx, selected_nodes, selected_year, timeout=90)
                        
                        if isinstance(box_result, list) and box_result:
                            fig = create_node_box_plot(box_result, title=f'B{selected_bx} Price Distribution by Node ({selected_year})')
                            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'node_B{selected_bx}_distribution_{selected_year}'}})
                        elif isinstance(box_result, dict) and box_result.get('error'):
                            st.warning(f"Distribution chart unavailable: {box_result.get('error')}")
                    
                    # Full year 8760-hour heatmap (daily granularity) - only for Annual time period
                    if time_period == "Annual":
                        st.subheader("8760 Full Year Price Heatmap (All Hours)")
                        with st.spinner("Loading 8760 heatmap... (this may take 30-60 seconds)"):
                            full_year_result = run_subprocess_query('full_year_8760', selected_nodes, selected_year, timeout=120)
                        
                        if isinstance(full_year_result, list) and full_year_result:
                            fig = create_8760_heatmap(full_year_result, title=f'All Hourly Prices ({len(selected_nodes)} nodes, {selected_year})', year=selected_year)
                            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'node_8760_heatmap_{selected_year}'}})
                        elif isinstance(full_year_result, dict) and full_year_result.get('error'):
                            st.warning(f"8760 heatmap unavailable: {full_year_result.get('error')}")
                else:
                    error_msg = bx_stats.get('error', 'No data found')
                    st.warning(f"Could not compute statistics: {error_msg}")
                    st.info(f"Available years for node analysis: {st.session_state.get('parquet_years', [])}")
    
    except Exception as e:
        st.warning(f"Could not load BX statistics: {str(e)}")
        st.info("Make sure LMP data is loaded and BX calculations have been run.")


def render_methodology_tab():
    """Render the Methodology & Data tab with calculation explanations, missing data report, and daily BX tables."""
    import subprocess
    import json as json_mod

    st.header("Methodology")

    with st.expander("BX Calculation Methodology", expanded=True):
        st.markdown("""
**BX (Cheapest X Hours) Calculation**

For each operating day, the BX value is computed as follows:

1. **Source Data**: Hourly LMP prices from CAISO Day Ahead market
2. **Operating Hour**: Uses CAISO's `OPR_HR` column (1-24, Pacific Prevailing Time). 
   This is **never** derived from `INTERVALSTARTTIME_GMT` (which is UTC/GMT and would cause a 7-8 hour offset)
3. **Hour 25 (DST)**: On fall-back DST days, CAISO reports a 25th hour. This tool **filters out hour 25** from BX calculations to keep a consistent 24-hour basis
4. **Ranking**: All 24 hours for a given day and zone/node are sorted by LMP price ascending
5. **Selection**: The cheapest X hours are selected (e.g., B8 = cheapest 8 hours)
6. **Daily BX Price**: Simple average of the selected X cheapest hours' LMP values
7. **Monthly Average**: Simple average of daily BX values within each month (using only days that have data)
8. **Annual Average**: Each month's average is weighted by that month's share of the year's calendar days.
   - Formula: `(Jan_avg × 31 + Feb_avg × days_in_feb + ... + Dec_avg × 31) / total_days_in_year`
   - **Leap year example (2024, 366 days)**: Feb gets weight 29/366 ≈ 7.9%, Jan gets 31/366 ≈ 8.5%, etc.
   - **Non-leap year (2023, 365 days)**: Feb gets weight 28/365 ≈ 7.7%
   - The denominator is always the total calendar days in the year (365 or 366), not the number of days with data

**Why this weighting?** If data is missing disproportionately in some months (e.g., only 12 of 29 February days loaded in 2024), a simple average of all daily values would under-count February. By first averaging within each month, then weighting months by their calendar-day share, each month contributes proportionally regardless of how many days of data are available.

**Example**: If B8 for SP15 on Jan 1 selects hours with prices [$5, $8, $10, $12, $15, $18, $20, $22], 
the B8 price for that day = ($5 + $8 + $10 + $12 + $15 + $18 + $20 + $22) / 8 = $13.75/MWh
""")

    with st.expander("EIA Zone Averaging", expanded=True):
        st.markdown("""
**Zone Price Source**

The zone-level hourly prices (NP15, SP15, ZP26) come from CAISO's published **EIA zone aggregate** files 
(PRC_LMP dataset, `LMP_TYPE = LMP`). These are CAISO's own load-weighted average prices for each zone, 
not computed by this tool.

- **NP15**: Northern California (Pacific Gas & Electric territory)
- **SP15**: Southern California (Southern California Edison territory)  
- **ZP26**: Central California (Zone P26, between NP15 and SP15)

**Generator Settlement Nodes** (TH_NP15_GEN-APND, TH_SP15_GEN-APND, TH_ZP26_GEN-APND) are separate 
CAISO-published aggregate prices representing generation-weighted averages for each zone. These may 
differ from the EIA zone averages because they weight by generation output rather than load.

**Missing Data Handling**: When EIA data is missing for certain days, each month's average is computed 
using only the days that have data. The annual average then weights each month by its share of the 
year's calendar days (e.g., January = 31/366 in a leap year), matching the BX methodology above.
This tool does **not** interpolate or fill in missing days.
""")

    with st.expander("Data Sources & Storage", expanded=False):
        st.markdown("""
**Data Pipeline**

- **Source**: CAISO OASIS Day Ahead LMP files (ZIP format, one per day)
- **Archive**: Raw Parquet files in AWS S3 (741 files, ~1 GB) — retained for backup, not queried at runtime
- **Analytics Database**: MotherDuck (DuckDB cloud) — all queries run here, including node-level and zone-level data

**MotherDuck Tables**
- `zone_hourly_lmp`: Hourly LMP by zone (NP15, SP15, ZP26) with congestion/energy/loss components
- `node_hourly_lmp`: Raw node-level hourly LMP data (~296M rows, ~17,500 nodes, 2020–2025)
- `bx_daily_summary`: Pre-computed daily BX values by zone (from zone_hourly_lmp)
- `generator_bx_summary`: BX values for generator settlement nodes (TH_*_GEN-APND)
- `node_zone_mapping`: EIA mapping of individual pricing nodes to zones
""")

    st.divider()

    st.header("Data Coverage Report")

    available_years = st.session_state.get('init_years', [2024])
    selected_year = st.selectbox("Select Year", available_years, key="methodology_year")

    try:
        r = subprocess.run(
            ['python3', 'subprocess_query.py', 'missing_days', str(selected_year)],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0 and r.stdout.strip():
            missing_data = json_mod.loads(r.stdout.strip())
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Days Expected", missing_data['total_expected'])
            with col2:
                st.metric("Days Loaded", missing_data['total_loaded'])
            with col3:
                st.metric("Days Missing", missing_data['missing_count'])

            if missing_data['missing_count'] > 0:
                st.warning(f"{missing_data['missing_count']} days missing from zone hourly data for {selected_year}")
                missing_df = pd.DataFrame({'Missing Date': missing_data['missing_dates']})
                missing_df['Month'] = pd.to_datetime(missing_df['Missing Date']).dt.strftime('%B')
                month_counts = missing_df['Month'].value_counts().sort_index()
                st.markdown("**Missing days by month:**")
                for month, count in month_counts.items():
                    dates_in_month = missing_df[missing_df['Month'] == month]['Missing Date'].tolist()
                    st.markdown(f"- **{month}**: {count} days ({', '.join(dates_in_month)})")
            else:
                st.success(f"All {missing_data['total_expected']} days loaded for {selected_year}")
        else:
            st.error("Could not retrieve data coverage information")
    except Exception as e:
        st.error(f"Error checking data coverage: {str(e)}")

    st.divider()

    st.header("Daily BX Values by Zone")
    st.markdown("Daily BX prices for EIA zones, computed from `zone_hourly_lmp`. Download as CSV to compare with your own calculations.")

    bx_select = st.selectbox("BX Type", [4, 5, 6, 7, 8, 9, 10], index=4, key="methodology_bx",
                             format_func=lambda x: f"B{x} (Cheapest {x} Hours)")

    if st.button("Load Daily BX Data", type="primary"):
        with st.spinner(f"Computing daily B{bx_select} for all zones in {selected_year}..."):
            try:
                r = subprocess.run(
                    ['python3', 'subprocess_query.py', 'zone_daily_bx', str(bx_select), str(selected_year)],
                    capture_output=True, text=True, timeout=60
                )
                if r.returncode == 0 and r.stdout.strip():
                    rows = json_mod.loads(r.stdout.strip())
                    if isinstance(rows, list) and len(rows) > 0:
                        df = pd.DataFrame(rows)
                        pivot = df.pivot(index='opr_dt', columns='zone', values='bx_price')
                        pivot = pivot.sort_index()
                        pivot.index.name = 'Date'

                        st.subheader(f"B{bx_select} Annual Averages — Monthly Weighted ({selected_year})")
                        from calendar import monthrange, isleap
                        total_cal_days = 366 if isleap(selected_year) else 365
                        df['opr_dt_parsed'] = pd.to_datetime(df['opr_dt'])
                        df['month'] = df['opr_dt_parsed'].dt.month
                        avg_cols = st.columns(len(pivot.columns))
                        for i, zone in enumerate(sorted(pivot.columns)):
                            with avg_cols[i]:
                                zone_df = df[df['zone'] == zone]
                                monthly_avgs = zone_df.groupby('month')['bx_price'].mean()
                                weighted_sum = 0
                                for m, avg in monthly_avgs.items():
                                    _, cal_days = monthrange(selected_year, int(m))
                                    weighted_sum += avg * cal_days
                                monthly_weighted = weighted_sum / total_cal_days
                                simple_avg = zone_df['bx_price'].mean()
                                st.metric(zone, f"${monthly_weighted:.2f}/MWh",
                                          help=f"Monthly weighted (by calendar days). Simple avg: ${simple_avg:.2f}/MWh. Days with data: {len(zone_df)}")

                        st.subheader(f"Daily B{bx_select} Values")
                        display_df = pivot.copy()
                        for col in display_df.columns:
                            display_df[col] = display_df[col].apply(lambda x: f"${x:.2f}" if pd.notna(x) else "N/A")
                        st.dataframe(display_df, use_container_width=True, height=400)

                        csv = pivot.to_csv()
                        st.download_button(
                            label=f"Download B{bx_select} Daily Data as CSV",
                            data=csv,
                            file_name=f"B{bx_select}_daily_{selected_year}.csv",
                            mime="text/csv"
                        )
                    else:
                        st.warning(f"No B{bx_select} data found for {selected_year}")
                else:
                    error_msg = r.stderr if r.stderr else "Unknown error"
                    st.error(f"Error computing BX data: {error_msg}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


    st.divider()

    st.header("Monthly BX Spot-Check")
    st.markdown("Monthly breakdown comparing all three averaging methods. Use this to verify annual averages and spot-check individual months.")

    sc_col1, sc_col2, sc_col3 = st.columns(3)
    with sc_col1:
        sc_bx = st.selectbox("BX Type", [4, 5, 6, 7, 8, 9, 10], index=4, key="spotcheck_bx",
                              format_func=lambda x: f"B{x}")
    with sc_col2:
        sc_zone = st.selectbox("Zone", ['SP15', 'NP15', 'ZP26'], key="spotcheck_zone")
    with sc_col3:
        sc_year = st.selectbox("Year", available_years, key="spotcheck_year")

    if st.button("Load Monthly Spot-Check", type="primary", key="load_spotcheck"):
        with st.spinner(f"Loading monthly B{sc_bx} for {sc_zone} ({sc_year})..."):
            try:
                r = subprocess.run(
                    ['python3', 'subprocess_query.py', 'monthly_bx_spotcheck',
                     str(sc_bx), str(sc_year), sc_zone],
                    capture_output=True, text=True, timeout=60
                )
                if r.returncode == 0 and r.stdout.strip():
                    sc_data = json_mod.loads(r.stdout.strip())
                    st.session_state['spotcheck_data'] = sc_data
                else:
                    st.error("Could not load spot-check data")
            except Exception as e:
                st.error(f"Error: {str(e)}")

    if 'spotcheck_data' in st.session_state:
        sc_data = st.session_state['spotcheck_data']
        rows = []
        for m in sc_data['months']:
            rows.append({
                'Month': m['month'],
                'Cal Days': m['cal_days'],
                'Load-Weighted ($/MWh)': f"${m['load_weighted']:.2f}" if m['load_weighted'] is not None else "N/A",
                'LW Days': m['load_weighted_days'],
                'Generator ($/MWh)': f"${m['generator']:.2f}" if m['generator'] is not None else "N/A",
                'Gen Days': m['generator_days'],
                'Node Avg ($/MWh)': f"${m['node_avg']:.2f}" if m['node_avg'] is not None else "N/A",
                'Node Days': m['node_avg_days'],
            })
        ann = sc_data['annual']
        rows.append({
            'Month': 'Annual',
            'Cal Days': '',
            'Load-Weighted ($/MWh)': f"${ann['load_weighted']:.2f}" if ann['load_weighted'] else "N/A",
            'LW Days': '',
            'Generator ($/MWh)': f"${ann['generator']:.2f}" if ann['generator'] else "N/A",
            'Gen Days': '',
            'Node Avg ($/MWh)': f"${ann['node_avg']:.2f}" if ann['node_avg'] else "N/A",
            'Node Days': '',
        })
        sc_df = pd.DataFrame(rows)
        st.dataframe(sc_df, use_container_width=True, hide_index=True)
        st.caption("Annual = sum(month_avg × calendar_days_in_month) / total_days_in_year (365 or 366 for leap years)")


def render_ai_assistant_tab():
    """
    Render the AI Assistant tab with the chatbot interface.
    
    This contains the existing chatbot functionality.
    """
    st.header("AI-Powered Analysis")
    st.markdown("Ask natural language questions about your CAISO LMP data")
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        with st.container():
            st.markdown(f"**Question {i+1}:** {question}")
            st.markdown(f"**Answer:** {answer}")
            st.divider()
    
    # Chat input
    user_question = st.text_input(
        "Ask a question about your LMP data:",
        placeholder="e.g., What are the 5 cheapest hours at each node?"
    )
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("Ask AI", type="primary"):
            if user_question:
                with st.spinner("Analyzing your question..."):
                    try:
                        answer = st.session_state.chatbot.process_question(user_question)
                        st.session_state.chat_history.append((user_question, answer))
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")
    
    with col2:
        if st.button("Quick Answer", type="secondary"):
            if user_question and user_question.strip():
                # Direct data answers without AI
                answer = "Quick Analysis Results:\n\n"
                try:
                    if any(word in user_question.lower() for word in ['cheapest', 'lowest']) and any(word in user_question.lower() for word in ['hour', 'operational hour']):
                        hourly_data = st.session_state.analytics.get_hourly_averages()
                        if not hourly_data.empty:
                            cheapest_hour = hourly_data.loc[hourly_data['mw'].idxmin()]
                            answer += f"**Cheapest Hour:** Hour {cheapest_hour['hour']} with average price ${cheapest_hour['mw']:.2f}/MWh\n\n"
                            answer += "All hourly averages:\n"
                            for _, row in hourly_data.head(10).iterrows():
                                answer += f"Hour {row['hour']}: ${row['mw']:.2f}/MWh\n"
                        else:
                            answer += "No hourly data available"
                    else:
                        # Default to cheapest hours
                        cheapest = st.session_state.analytics.get_cheapest_hours(10)
                        if not cheapest.empty:
                            answer += f"**10 Cheapest Individual Hours:**\n"
                            for _, row in cheapest.head(10).iterrows():
                                answer += f"{row['operational_date']} Hour {row['operational_hour']}: ${row['mw']:.2f}/MWh at {row['node']}\n"
                        else:
                            answer += "No data available"
                except Exception as e:
                    answer += f"Error: {str(e)}"
                
                st.session_state.chat_history.append((user_question, answer))
                st.rerun()
    
    with col3:
        if st.button("Clear Chat"):
            st.session_state.chat_history = []
            st.rerun()

    # Available Analytics Tools Section
    st.divider()
    st.subheader("Available Analytics Tools")
    st.markdown("Below are all the analytics methods available. You can ask questions that relate to any of these capabilities:")
    
    # Get registered analytics methods
    try:
        registered_methods = get_registered_analytics()
        
        # Display in expandable sections for better organization
        num_cols = 2
        method_items = list(registered_methods.items())
        
        for i in range(0, len(method_items), num_cols):
            cols = st.columns(num_cols)
            
            for j, col in enumerate(cols):
                if i + j < len(method_items):
                    method_name, method_info = method_items[i + j]
                    
                    with col:
                        with st.expander(f"📊 {method_info['description'][:50]}...", expanded=False):
                            st.markdown(f"**Method:** `{method_name}`")
                            st.markdown(f"**Description:** {method_info['description']}")
                            
                            if method_info['parameters']:
                                params_str = ", ".join(method_info['parameters'])
                                st.markdown(f"**Parameters:** `{params_str}`")
                            
                            if method_info['example_questions']:
                                st.markdown("**Example Questions:**")
                                for question in method_info['example_questions']:
                                    st.markdown(f"• _{question}_")
                                    
    except Exception as e:
        st.error(f"Error loading analytics tools: {str(e)}")


if __name__ == "__main__":
    main()
