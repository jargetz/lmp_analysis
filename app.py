"""
BX CAISO Nodal Analysis Tool - Main Application

Primary views:
1. Dashboard - BX analysis and zone/node filtering
2. Node Map - Geographic PNODE price map with facility overlay
3. Methodology & Data - Calculation explanations and spot-check tables
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
    create_8760_heatmap,
    create_pnode_map,
    create_pnode_price_histogram,
    create_node_finder_map,
)

def main():
    st.set_page_config(
        page_title="BX CAISO Nodal Analysis Tool",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ BX CAISO Nodal Analysis Tool")
    st.markdown("Analyze historical data for the cheapest 4–10 hours of nodal prices across CAISO.")
    
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
                st.session_state.node_years = init_data.get('node_years', [2024])
            else:
                st.session_state.db_summary = {}
                st.session_state.init_years = [2024]
                st.session_state.node_years = [2024]
                st.session_state.init_nodes = []
                st.session_state.individual_nodes = []
                st.session_state.zone_years = [2024]
        except Exception:
            st.session_state.db_summary = {}
            st.session_state.init_years = [2024]
            st.session_state.node_years = [2024]
            st.session_state.init_nodes = []
            st.session_state.individual_nodes = []
            st.session_state.zone_years = [2024]
        st.session_state.init_data = True
    
    with st.sidebar:
        st.header("Data Status")
        
        summary = st.session_state.db_summary
        if summary and summary.get('total_records', 0) > 0:
            st.success("Data loaded and ready")
            st.caption("All data in MotherDuck (DuckDB cloud)")
            st.session_state.data_loaded = True
            
            st.subheader("Data Details")
            st.markdown("**Zones (zonal data)**")
            st.markdown("NP15, SP15, ZP26")
            
            zone_earliest = summary.get('earliest_date', '')
            zone_latest = summary.get('latest_date', '')
            if hasattr(zone_earliest, 'strftime'):
                zone_earliest = zone_earliest.strftime('%Y-%m-%d')
            if hasattr(zone_latest, 'strftime'):
                zone_latest = zone_latest.strftime('%Y-%m-%d')
            zone_earliest = str(zone_earliest).split(' ')[0] if zone_earliest else ''
            zone_latest = str(zone_latest).split(' ')[0] if zone_latest else ''
            
            if zone_earliest and zone_latest:
                st.markdown(f"Zone data: {zone_earliest} to {zone_latest}")
            
            available_years = st.session_state.get('init_years', [2024])
            if available_years:
                st.markdown(f"Node data years: {min(available_years)}–{max(available_years)}")
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
        tab_dashboard, tab_node_map, tab_node_finder, tab_methodology = st.tabs([
            "📊 Dashboard", "🗺️ Node Map", "🔍 Node Finder", "📋 Methodology & Data"
        ])
        
        # =====================================================================
        # DASHBOARD TAB - Primary interface for BX analysis with zone filtering
        # =====================================================================
        with tab_dashboard:
            render_dashboard_tab()
        
        # =====================================================================
        # NODE MAP TAB - Geographic PNODE price map with facility overlay
        # =====================================================================
        with tab_node_map:
            render_node_map_tab()
        
        # =====================================================================
        # NODE FINDER TAB - Cheapest nodes ∪ nodes near top GHG emitters
        # =====================================================================
        with tab_node_finder:
            render_node_finder_tab()
        
        # =====================================================================
        # METHODOLOGY & DATA TAB
        # =====================================================================
        with tab_methodology:
            render_methodology_tab()


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
        # Use node_years (only years present in node_hourly_lmp)
        parquet_years = st.session_state.get('node_years', st.session_state.get('parquet_years', [2024]))
        
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
            
            st.markdown("**EIA Load-Weighted Zone Average**")
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
            
            gen_cache_key = f"gen_stats_{selected_bx}_{selected_year}_{time_period}_{selected_month}"
            if gen_cache_key not in st.session_state:
                st.session_state[gen_cache_key] = bx_calc.get_generator_bx_average(
                    bx=selected_bx,
                    year=selected_year,
                    time_period=time_period,
                    month=selected_month
                )
            gen_stats = st.session_state[gen_cache_key]
            
            if gen_stats.get('success') and gen_stats.get('zones'):
                st.markdown("**Generator Settlement Prices** (gen-weighted by CAISO)")
                gen_cols = st.columns(3)
                gen_zone_order = ['NP15', 'SP15', 'ZP26']
                
                for col, zone_name in zip(gen_cols, gen_zone_order):
                    with col:
                        zone_data = gen_stats['zones'].get(zone_name, {})
                        if zone_data.get('avg_price') is not None:
                            st.metric(
                                f"{zone_name}",
                                f"${zone_data['avg_price']:.2f}/MWh",
                                help=f"TH_{zone_name}_GEN-APND (gen-weighted by CAISO). Days: {zone_data.get('day_count', 0)}"
                            )
                        else:
                            st.metric(f"{zone_name}", "N/A")
            
            st.subheader("Averages - Day Ahead LMP")
            heatmap_tab_labels = ['Overall (not weighted by zone)', 'NP15', 'SP15', 'ZP26']
            heatmap_data_keys = ['Overall', 'NP15', 'SP15', 'ZP26']
            heatmap_tabs = st.tabs(heatmap_tab_labels)
            
            all_heatmap_key = f"all_heatmaps_{selected_year}"
            if all_heatmap_key not in st.session_state:
                st.session_state[all_heatmap_key] = bx_calc.get_all_zones_month_hour(year=selected_year)
            all_heatmaps = st.session_state[all_heatmap_key]
            
            for tab, data_key, tab_label in zip(heatmap_tabs, heatmap_data_keys, heatmap_tab_labels):
                with tab:
                    heatmap_data = all_heatmaps.get(data_key, [])
                    if heatmap_data:
                        fig = create_month_hour_heatmap(heatmap_data, zone=tab_label)
                        st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'zone_heatmap_{data_key}_{selected_year}'}})
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
                    stat_col1, stat_col2 = st.columns(2)
                    with stat_col1:
                        st.metric(f"B{selected_bx} Overall Avg", f"${bx_stats['avg_price']:.2f}/MWh")
                    with stat_col2:
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
                        st.caption("Hours shown are the most frequently selected across all days — actual hours vary each day based on that day's prices.")
                    
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
                            fig, trend_clipping = create_node_bx_trend_chart(trend_result, bx_type=selected_bx)
                            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'node_B{selected_bx}_trend_{selected_year}'}})
                            if trend_clipping:
                                parts = []
                                if trend_clipping['clipped_below']:
                                    parts.append(f"values below ${trend_clipping['ymin']:.0f} (actual min: ${trend_clipping['actual_min']:.0f})")
                                if trend_clipping['clipped_above']:
                                    parts.append(f"values above ${trend_clipping['ymax']:.0f} (actual max: ${trend_clipping['actual_max']:.0f})")
                                st.caption(f"Y-axis clipped to 2nd–98th percentile: {'; '.join(parts)}. Hover for exact values.")
                        elif isinstance(trend_result, dict) and trend_result.get('error'):
                            st.warning(f"BX trend unavailable: {trend_result.get('error')}")
                    else:
                        st.info(f"BX trend chart available for 10 or fewer nodes (currently {len(selected_nodes)} selected)")
                    
                    # Box plot for node comparison (outlier detection)
                    if len(selected_nodes) > 1:
                        with st.spinner("Loading price distribution..."):
                            box_result = run_subprocess_query('box_stats', selected_bx, selected_nodes, selected_year, timeout=90)
                        
                        if isinstance(box_result, list) and box_result:
                            fig, box_clipping = create_node_box_plot(box_result, title=f'B{selected_bx} Price Distribution by Node ({selected_year})')
                            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'node_B{selected_bx}_distribution_{selected_year}'}})
                            if box_clipping:
                                st.caption(f"Minimum values capped at ${box_clipping['floor']:.0f}/MWh for {box_clipping['clipped_count']} node(s) — actual worst min: ${box_clipping['worst_original_min']:.0f}/MWh.")
                        elif isinstance(box_result, dict) and box_result.get('error'):
                            st.warning(f"Distribution chart unavailable: {box_result.get('error')}")
                    
                    # Full year 8760-hour heatmap (daily granularity) - only for Annual time period
                    if time_period == "Annual":
                        st.subheader("8760 Full Year Price Heatmap (All Hours)")
                        with st.spinner("Loading 8760 heatmap... (this may take 30-60 seconds)"):
                            full_year_result = run_subprocess_query('full_year_8760', selected_nodes, selected_year, timeout=120)
                        
                        if isinstance(full_year_result, list) and full_year_result:
                            fig, clipping_info_8760 = create_8760_heatmap(full_year_result, title=f'All Hourly Prices ({len(selected_nodes)} nodes, {selected_year})', year=selected_year)
                            st.plotly_chart(fig, use_container_width=True, config={'toImageButtonOptions': {'filename': f'node_8760_heatmap_{selected_year}'}})
                            if clipping_info_8760:
                                parts = []
                                if clipping_info_8760['clipped_below']:
                                    parts.append(f"values below ${clipping_info_8760['zmin']:.0f} (actual min: ${clipping_info_8760['actual_min']:.0f})")
                                if clipping_info_8760['clipped_above']:
                                    parts.append(f"values above ${clipping_info_8760['zmax']:.0f} (actual max: ${clipping_info_8760['actual_max']:.0f})")
                                st.caption(f"Color scale clipped to 2nd–98th percentile: {'; '.join(parts)}. Hover for exact values.")
                        elif isinstance(full_year_result, dict) and full_year_result.get('error'):
                            st.warning(f"8760 heatmap unavailable: {full_year_result.get('error')}")
                else:
                    error_msg = bx_stats.get('error', 'No data found')
                    st.warning(f"Could not compute statistics: {error_msg}")
                    st.info(f"Available years for node analysis: {st.session_state.get('parquet_years', [])}")
    
    except Exception as e:
        st.warning(f"Could not load BX statistics: {str(e)}")
        st.info("Make sure LMP data is loaded and BX calculations have been run.")


def render_node_map_tab():
    """Render the geographic PNODE price map tab."""
    import subprocess
    import json

    st.header("Node Map")
    st.markdown("Geographic view of PNODE B*X* average prices. Joined from the pre-computed monthly summary table and coordinate data.")

    # ── Filters ──────────────────────────────────────────────────────────────
    map_col1, map_col2, map_col3, map_col4, map_col5 = st.columns([1, 1, 1, 1, 1])

    with map_col1:
        map_bx = st.selectbox(
            "BX Hours",
            options=SUPPORTED_BX_VALUES,
            index=4,
            format_func=lambda x: f"B{x} (Cheapest {x} hours)",
            key="map_bx",
        )

    with map_col2:
        map_time_period = st.selectbox(
            "Time Period",
            options=["Annual", "Monthly"],
            key="map_time_period",
        )

    month_options = ["January", "February", "March", "April", "May", "June",
                     "July", "August", "September", "October", "November", "December"]

    available_years = st.session_state.get('parquet_years', [2024, 2025])
    map_years = [y for y in available_years]
    if not map_years:
        map_years = [2024]
    default_year_idx = map_years.index(2025) if 2025 in map_years else 0

    with map_col3:
        map_year = st.selectbox(
            "Year",
            options=map_years,
            index=default_year_idx,
            key="map_year",
        )

    map_month = None
    if map_time_period == "Monthly":
        with map_col4:
            map_month_name = st.selectbox("Month", options=month_options, key="map_month")
            map_month = month_options.index(map_month_name) + 1

    with map_col5:
        color_by = st.radio("Color by", options=["Zone", "Price"], key="map_color_by", horizontal=True)

    # ── Facility controls ─────────────────────────────────────────────────────
    fac_col1, fac_col2 = st.columns([1, 2])
    with fac_col1:
        show_facilities = st.checkbox("Show CARB facilities (2023 data)", value=True, key="map_show_facilities")
    with fac_col2:
        if show_facilities:
            fac_filter = st.radio(
                "Filter",
                options=["All facilities", "Covered entities only"],
                key="map_facility_filter",
                horizontal=True,
            )
        else:
            fac_filter = "All facilities"

    st.divider()

    # ── Load facility data (cached for the session) ───────────────────────────
    if 'facility_data' not in st.session_state:
        with st.spinner("Loading CARB facility data…"):
            try:
                proc = subprocess.run(
                    ['python3', 'subprocess_query.py', 'facility_emissions'],
                    capture_output=True, text=True, timeout=30
                )
                if proc.returncode == 0:
                    st.session_state['facility_data'] = json.loads(proc.stdout)
                else:
                    st.session_state['facility_data'] = []
            except Exception:
                st.session_state['facility_data'] = []

    facilities_all = st.session_state.get('facility_data', [])
    all_facilities_sorted = sorted(facilities_all, key=lambda f: f['facility'])
    all_facility_names = [f['facility'] for f in all_facilities_sorted]

    facilities_to_show = facilities_all if show_facilities else []
    if fac_filter == "Covered entities only":
        facilities_to_show = [f for f in facilities_to_show if f.get('cap_and_trade') == 'Yes']

    # ── PNODE price data fetch ────────────────────────────────────────────────
    period_label = str(map_year) if map_time_period == "Annual" else f"{month_options[map_month - 1]} {map_year}"
    cache_key = f"node_map_{map_bx}_{map_year}_{map_time_period}_{map_month}"

    if cache_key not in st.session_state:
        with st.spinner(f"Loading B{map_bx} node map for {period_label}…"):
            cmd = [
                'python3', 'subprocess_query.py', 'node_map',
                str(map_bx), str(map_year), map_time_period,
                str(map_month) if map_month else '',
            ]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                if proc.returncode == 0:
                    st.session_state[cache_key] = json.loads(proc.stdout)
                else:
                    st.session_state[cache_key] = {'error': proc.stderr}
            except subprocess.TimeoutExpired:
                st.session_state[cache_key] = {'error': 'Query timed out after 60s'}
            except Exception as exc:
                st.session_state[cache_key] = {'error': str(exc)}

    map_data = st.session_state.get(cache_key, [])

    if isinstance(map_data, dict) and map_data.get('error'):
        st.error(f"Map data error: {map_data['error']}")
        return

    if not map_data:
        st.warning("No coordinate data found for the selected period.")
        return

    # ── Price distribution histogram ──────────────────────────────────────────
    total_nodes = len(map_data)
    fac_note = f" · {len(facilities_to_show)} facilities shown" if facilities_to_show else ""
    st.caption(f"{total_nodes:,} nodes with coordinates plotted{fac_note}")
    hist_fig = create_pnode_price_histogram(map_data, bx_label=f"B{map_bx}")
    st.plotly_chart(hist_fig, use_container_width=True)

    # ── Facility search + nearest-node (above map) ────────────────────────────
    search_col1, search_col2 = st.columns([2, 3])
    with search_col1:
        selected_name = st.selectbox(
            "Search facility",
            options=all_facility_names,
            index=None,
            placeholder="Type to search...",
            key="map_facility_search",
        )

    import math as _math
    import numpy as _np

    if 'ca_substations_df' not in st.session_state:
        try:
            _sub_df = pd.read_csv(
                'attached_assets/CA_Substation_Coordinates_1773167584973.csv',
                dtype=str,
            )
            for _c in _sub_df.columns:
                if _sub_df[_c].dtype == object:
                    _sub_df[_c] = _sub_df[_c].str.strip()
            _sub_df['lat'] = pd.to_numeric(_sub_df['Y'], errors='coerce')
            _sub_df['lon'] = pd.to_numeric(_sub_df['X'], errors='coerce')
            _sub_df['Highest_kV'] = _sub_df['Highest_kV'].replace('33kV to 92Kv', '33kV to 92kV')
            _sub_df = _sub_df.dropna(subset=['lat', 'lon']).reset_index(drop=True)
            st.session_state['ca_substations_df'] = _sub_df
        except Exception:
            st.session_state['ca_substations_df'] = pd.DataFrame()

    sub_df = st.session_state['ca_substations_df']

    selected_facility = None
    nearest_node = None
    nearest_substation = None
    nearest_any_sub = None

    if selected_name:
        selected_facility = next((f for f in all_facilities_sorted if f['facility'] == selected_name), None)

    if selected_facility and map_data:
        flat = selected_facility['lat']
        flon = selected_facility['lon']
        _R = 6371.0
        _phi1 = _math.radians(flat)

        valid_nodes = [n for n in map_data if n.get('lat') is not None and n.get('lon') is not None]
        if valid_nodes:
            _nlats = _np.array([n['lat'] for n in valid_nodes])
            _nlons = _np.array([n['lon'] for n in valid_nodes])
            _dphi = _np.radians(_nlats - flat)
            _dlam = _np.radians(_nlons - flon)
            _a = (_np.sin(_dphi / 2) ** 2
                  + _math.cos(_phi1) * _np.cos(_np.radians(_nlats)) * _np.sin(_dlam / 2) ** 2)
            _node_dists = _R * 2 * _np.arcsin(_np.sqrt(_np.clip(_a, 0, 1)))
            _ni = int(_node_dists.argmin())
            nearest_node = valid_nodes[_ni]
            dist_km = float(_node_dists[_ni])
        else:
            nearest_node = None
            dist_km = 0.0

        if not sub_df.empty:
            _low_kv = {'12kV to 32kV', '33kV to 92kV', '33kV to 92Kv'}
            hv_df = sub_df[~sub_df['Highest_kV'].isin(_low_kv)].reset_index(drop=True)
            _search_df = hv_df if not hv_df.empty else sub_df

            def _nearest_sub(sdf):
                _lats = sdf['lat'].values
                _lons = sdf['lon'].values
                _dp = _np.radians(_lats - flat)
                _dl = _np.radians(_lons - flon)
                _av = (_np.sin(_dp / 2) ** 2
                       + _math.cos(_phi1) * _np.cos(_np.radians(_lats)) * _np.sin(_dl / 2) ** 2)
                _ds = _R * 2 * _np.arcsin(_np.sqrt(_np.clip(_av, 0, 1)))
                _i = int(_ds.argmin())
                _r = sdf.iloc[_i]
                return {
                    'substation_name': str(_r['Substation_Name']),
                    'owner': str(_r['Owner']),
                    'highest_kv': str(_r['Highest_kV']) if pd.notna(_r['Highest_kV']) else None,
                    'status': str(_r['Status']),
                    'lat': float(_r['lat']),
                    'lon': float(_r['lon']),
                    'dist_km': float(_ds[_i]),
                }

            nearest_substation = _nearest_sub(_search_df)
            nearest_any_sub = _nearest_sub(sub_df)


        with search_col2:
            ct_badge = "Cap-and-Trade" if selected_facility['cap_and_trade'] == 'Yes' else "Non-covered"
            node_price = f"${nearest_node['avg_price']:.2f}/MWh" if nearest_node and nearest_node.get('avg_price') is not None else "N/A"
            node_name = nearest_node['pnode_id'] if nearest_node else "—"

            ns_line = ''
            closer_lv_line = ''
            if nearest_substation:
                ns = nearest_substation
                kv_s = f', {ns["highest_kv"]}' if ns.get('highest_kv') else ''
                status_warn = f' ⚠ {ns["status"]}' if ns.get('status') and ns['status'] != 'Operational' else ''
                ns_line = (
                    f'  \n**Nearest Substation (≥110kV):** {ns["substation_name"]} '
                    f'({ns["owner"]}{kv_s}, {ns["dist_km"]:.1f} km){status_warn}'
                )
                if (nearest_any_sub
                        and nearest_any_sub['substation_name'] != ns['substation_name']
                        and nearest_any_sub['dist_km'] < ns['dist_km']):
                    lv = nearest_any_sub
                    lv_kv = f', {lv["highest_kv"]}' if lv.get('highest_kv') else ''
                    closer_lv_line = (
                        f'  \n⚠ Closer lower-voltage substation: {lv["substation_name"]} '
                        f'({lv["owner"]}{lv_kv}, {lv["dist_km"]:.1f} km)'
                    )

            st.info(
                f"**{selected_facility['facility']}** · {ct_badge}  \n"
                f"{selected_facility['primary_sector']} · {selected_facility['county']} Co. · {selected_facility['city']}  \n"
                f"Total GHG: **{selected_facility['total_ghg']:,.0f}** MT CO₂e · "
                f"CO₂: {selected_facility['co2']:,.0f} · "
                f"NOx: {selected_facility['nox']:,.1f} · "
                f"SOx: {selected_facility['sox']:,.1f} · "
                f"PM2.5: {selected_facility['pm25']:,.1f}  \n"
                f"**Nearest Node:** {node_name} ({dist_km:.1f} km) · B{map_bx} avg {node_price}"
                f"{ns_line}"
                f"{closer_lv_line}"
            )

    # ── Map chart ─────────────────────────────────────────────────────────────
    fig = create_pnode_map(
        map_data,
        bx_label=f"B{map_bx}",
        color_by=color_by.lower(),
        facilities=facilities_to_show if facilities_to_show else None,
        selected_facility=selected_facility,
        nearest_node=nearest_node,
        nearest_substation=nearest_substation,
        nearest_lv_substation=(
            nearest_any_sub
            if nearest_any_sub and nearest_substation
            and nearest_any_sub['substation_name'] != nearest_substation['substation_name']
            and nearest_any_sub['dist_km'] < nearest_substation['dist_km']
            else None
        ),
    )
    st.plotly_chart(fig, use_container_width=True, config={'scrollZoom': True})


def render_methodology_tab():
    """Render the Methodology & Data tab with calculation explanations, missing data report, and daily BX tables."""
    import subprocess
    import json as json_mod

    st.header("Methodology")

    st.subheader("BX Calculation Methodology")
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

    st.subheader("EIA Zone Averaging")
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

    st.subheader("Data Sources & Storage")
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

    st.subheader("California Substation Data")
    st.markdown("""
**Source**

[DataBasin / HIFLD — California Electric Substations](https://databasin.org/datasets/20502139197843f7b1b2751a427d9f68/)  
3,261 CA substations with geographic coordinates, owner, voltage class, and operational status.

**Coverage**

| Status | Count |
|---|---|
| Operational | 3,199 |
| Low kV (distribution-only, too small to appear in CAISO) | 36 |
| Proposed (not yet built) | 17 |
| Closed | 9 |

Non-operational substations are flagged with a ⚠ warning wherever shown in this tool.

**Major owners:** PG&E (991), SCE (945), SMUD (258), SDG&E (145), IID (133), LADWP (45), WAPA (33)

**Voltage tier explanation (Highest_kV column)**

The `Highest_kV` field records the *maximum* voltage class operating at that substation — derived 
from six boolean tier columns in the source data (kV_12_TO_32 through kV_500_DC):

| Tier | Typical role |
|---|---|
| 12–32 kV | Local distribution (neighborhood feeders) |
| 33–92 kV | Sub-transmission / distribution |
| 110–161 kV | Regional sub-transmission |
| 220–287 kV | Bulk transmission |
| 345–500 kV | High-voltage interstate backbone |
| 500 kV DC | HVDC long-distance transmission |

Higher kV generally means the substation sits closer to the bulk transmission grid and can 
handle larger power flows — relevant to industrial and data-center loads.

**How nodes are matched to substations**

Each CAISO pricing node (PNODE) is matched to its geographically nearest CA substation using 
Euclidean distance with a cos(lat) correction on longitude. All 11,978 pnodes in the CAISO 
pricemap are matched, but most are outside California (Western Interconnect coverage). 
For non-CA nodes, the nearest substation may be far away — this is expected.

**Node type note (sanity check)**

- **LOAD nodes** (9,076 total) — represent load aggregation points, typically at or near 
  substations. CA load nodes generally match within a few km.
- **GEN nodes** (2,902 total) — represent individual generation units (power plants). 
  Plants are not substations; their nearest substation match is often 5–50+ km away. 
  This is correct behavior, not a data error.
""")

    st.subheader("AB 617 Community Data Source")
    st.markdown("""
**Assembly Bill 617 — Community Air Protection Program**

AB 617 (signed 2017) established CARB's Community Air Protection Program to reduce air pollution 
in California's most burdened communities. CARB maintains a "Consistently Nominated Communities" 
list — areas repeatedly identified by air districts, community-based organizations, or self-nomination 
as priority locations for air quality intervention.

**Data used in this tool**

The Node Finder tab uses the **August 2023 Consistently Nominated Communities list** (65 communities 
across 5 air districts), sourced live from CARB's ArcGIS FeatureServer API:

| Air District | Communities |
|---|---|
| Bay Area AQMD | 20 |
| South Coast AQMD | 19 |
| San Joaquin Valley APCD | 17 |
| Sacramento Metropolitan AQMD | 6 |
| Imperial APCD | 2 |
| Other | 1 |

**Official PDF reference:**  
[2023 Consistently Nominated AB 617 Communities List (CARB, October 2023)](https://ww2.arb.ca.gov/sites/default/files/2023-10/2023%2008%20Consistently%20Nominated%20Communities_10.16.2023.pdf)

**How it's used in Node Finder**

When "AB 617 communities only" is checked, facilities from `facility_emissions` are filtered to 
those within **30 km** of any AB 617 community centroid. This surfaces industrial emitters that 
are already subject to (or near) community air quality programs — and whose nearest CAISO nodes 
may represent decarbonization opportunities.

**Known data note:** CARB's ArcGIS API has its `centroid_latitude` and `centroid_longtitude` 
field names reversed relative to their actual values. This tool applies the correct mapping 
(matching all 65 communities to valid California coordinates). Minor name discrepancies between 
the PDF and the API exist (CARB's own data) but do not affect geographic proximity calculations.
""")

    st.divider()

    st.header("Data Coverage Report")

    available_years = st.session_state.get('init_years', [2024])
    selected_year = st.selectbox("Select Year", available_years, key="methodology_year")

    zone_tab, node_tab = st.tabs(["Zone Data (zone_hourly_lmp)", "Node Data (node_hourly_lmp)"])

    with zone_tab:
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
                st.info(f"No zone data available for {selected_year}")
        except Exception as e:
            st.error(f"Error checking zone data coverage: {str(e)}")

    with node_tab:
        try:
            r = subprocess.run(
                ['python3', 'subprocess_query.py', 'node_coverage', str(selected_year)],
                capture_output=True, text=True, timeout=60
            )
            if r.returncode == 0 and r.stdout.strip():
                node_cov = json_mod.loads(r.stdout.strip())
                if node_cov.get('has_data'):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Days Loaded", node_cov['days_loaded'])
                    with col2:
                        st.metric("Days Expected", node_cov['total_expected'])
                    with col3:
                        st.metric("Unique Nodes", f"{node_cov['node_count']:,}")
                    with col4:
                        st.metric("Total Rows", f"{node_cov['total_rows']:,}")
                    st.markdown(f"**Date range**: {node_cov['earliest_date']} to {node_cov['latest_date']}")
                    missing_node_days = node_cov['total_expected'] - node_cov['days_loaded']
                    if missing_node_days > 0:
                        st.warning(f"{missing_node_days} days missing from node data for {selected_year}")
                    else:
                        st.success(f"All {node_cov['total_expected']} days loaded for {selected_year}")
                else:
                    st.info(f"No node data available for {selected_year}")
            else:
                st.info(f"No node data available for {selected_year}")
        except Exception as e:
            st.error(f"Error checking node data coverage: {str(e)}")

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


def render_node_finder_tab():
    """
    Facility-centric Node Finder: for each CARB GHG emitter, find its
    geographically nearest CAISO node and show that node's B-hour price.
    Facilities are ranked by cheapest nearest-node price.
    """
    import subprocess
    import json as json_mod

    def _run(query_type, *args, timeout=120):
        cmd = ['python3', 'subprocess_query.py', query_type] + [
            str(a) if not isinstance(a, list) else json_mod.dumps(a) for a in args
        ]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
            if r.returncode == 0 and r.stdout.strip():
                return json_mod.loads(r.stdout.strip())
            return {'error': r.stderr or 'Empty response'}
        except subprocess.TimeoutExpired:
            return {'error': f'Query timed out after {timeout}s'}
        except Exception as e:
            return {'error': str(e)}

    st.header("Node Finder")
    st.warning("This tab is a work in progress. Results and methodology may change.")
    st.markdown(
        "For each CARB-reported GHG-emitting facility, this tool finds the "
        "**geographically nearest CAISO pricing node** and shows its B-hour electricity price. "
        "Facilities are ranked by how cheap that nearest node is — giving you a picture of "
        "which industrial sites have access to the cheapest off-peak power."
    )

    node_years = st.session_state.get('node_years', [2025, 2024])

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        nf_year = st.selectbox("Year", options=node_years, index=0, key="nf_year")
    with c2:
        nf_bx = st.selectbox("BX", options=SUPPORTED_BX_VALUES, index=4,
                              format_func=lambda x: f"B{x}", key="nf_bx")
    with c3:
        nf_zone = st.selectbox("Node zone filter",
                               options=["All", "NP15", "SP15", "ZP26"], key="nf_zone",
                               help="Restrict which CAISO nodes are considered when finding nearest node")
    with c4:
        nf_top_m = st.slider("Top N facilities by GHG", 0, 391, 0, step=10,
                              key="nf_top_m",
                              help="0 = all 391 facilities. Otherwise limit to top N by total GHG emissions.")
    with c5:
        nf_ab617 = st.checkbox(
            "AB 617 communities only",
            key="nf_ab617",
            help="Filter facilities to those within 30 km of a CARB AB 617 nominated community"
        )

    top_m_label = "all" if nf_top_m == 0 else str(nf_top_m)
    cache_key = f"nf_{nf_bx}_{nf_year}_{nf_top_m}_{nf_ab617}_{nf_zone}"

    run_col, _ = st.columns([1, 5])
    with run_col:
        run_clicked = st.button("Find Nodes", type="primary", key="nf_run")

    if run_clicked and cache_key not in st.session_state:
        with st.spinner(
            f"Finding nearest node for {top_m_label} facilities ({nf_year} B{nf_bx})…"
        ):
            result = _run(
                'node_finder', nf_bx, nf_year, nf_top_m,
                str(nf_ab617).lower(), nf_zone,
                timeout=120
            )
            st.session_state[cache_key] = result

    if cache_key not in st.session_state:
        st.info("Set your filters above and click **Find Nodes** to run the analysis.")
        return

    result = st.session_state[cache_key]
    if 'error' in result:
        st.error(f"Query error: {result['error']}")
        return

    summary = result.get('summary', {})
    facilities = result.get('facilities', [])
    ab617 = result.get('ab617_communities', [])

    if not facilities:
        st.warning("No facilities returned — try adjusting the filters.")
        return

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Facilities Analyzed", summary.get('n_facilities', 0))
    with m2:
        st.metric("With Negative B Avg", summary.get('n_negative_b', 0),
                  help="Facilities whose nearest node has a negative average price during cheapest hours")
    with m3:
        avg_b = summary.get('avg_b_all')
        st.metric(f"Avg B{nf_bx} at Nearest Node",
                  f"${avg_b:.2f}/MWh" if avg_b is not None else "N/A")
    with m4:
        avg_d = summary.get('avg_dist_km')
        st.metric("Avg Distance to Node",
                  f"{avg_d:.1f} km" if avg_d is not None else "N/A")

    price_col = f'B{nf_bx} Avg ($/MWh)'
    table_rows = []
    for r in facilities:
        sub_name = r.get('substation_name') or '—'
        sub_status = r.get('substation_status')
        if sub_status and sub_status != 'Operational':
            sub_name = f'⚠ {sub_name} ({sub_status})'
        table_rows.append({
            'Facility': r['facility'],
            'County': r['county'],
            'Sector': r['primary_sector'],
            'GHG (MT CO₂e)': r['total_ghg'],
            'Cap & Trade': r['cap_and_trade'],
            'Nearest Node': r['nearest_node'],
            'Zone': r['node_zone'],
            'Dist (km)': round(r['dist_km'], 1),
            price_col: round(r['node_b_avg'], 2),
            'Substation': sub_name,
            'Owner': r.get('substation_owner') or '—',
            'Voltage': r.get('highest_kv') or '—',
        })
    display_df = pd.DataFrame(table_rows)

    st.divider()
    tab_table, tab_map = st.tabs(["📋 Data Table", "🗺️ Map"])

    with tab_table:
        fmt_df = display_df.copy()
        fmt_df['GHG (MT CO₂e)'] = fmt_df['GHG (MT CO₂e)'].apply(lambda x: f"{x:,.0f}")
        st.dataframe(fmt_df, use_container_width=True, hide_index=True, height=520)

        with st.expander("Summary Statistics"):
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                mn = summary.get('min_b')
                mx = summary.get('max_b')
                st.markdown(f"**Cheapest nearest-node B{nf_bx}:** ${mn:.2f}/MWh" if mn is not None else "N/A")
                st.markdown(f"**Most expensive:** ${mx:.2f}/MWh" if mx is not None else "N/A")
            with sc2:
                if facilities:
                    top5 = facilities[:5]
                    st.markdown("**Top 5 facilities with cheapest nearest node:**")
                    for f in top5:
                        st.markdown(f"- **{f['facility'][:40]}** → {f['nearest_node']} (${f['node_b_avg']:.2f}/MWh, {f['dist_km']:.1f} km)")
            with sc3:
                if facilities:
                    worst5 = facilities[-5:][::-1]
                    st.markdown("**Top 5 facilities with most expensive nearest node:**")
                    for f in worst5:
                        st.markdown(f"- **{f['facility'][:40]}** → {f['nearest_node']} (${f['node_b_avg']:.2f}/MWh, {f['dist_km']:.1f} km)")

        st.download_button(
            "Download as CSV",
            data=display_df.to_csv(index=False),
            file_name=f"node_finder_B{nf_bx}_{nf_year}_{nf_zone}.csv",
            mime="text/csv",
        )

    with tab_map:
        fig = create_node_finder_map(facilities, ab617, bx_label=f'B{nf_bx}')
        st.plotly_chart(
            fig,
            use_container_width=True,
            config={'scrollZoom': True,
                    'toImageButtonOptions': {'filename': f'node_finder_B{nf_bx}_{nf_year}'}}
        )
        st.caption(
            "Circles = facilities, colored by their nearest CAISO node's B-hour price "
            "(green = cheap, red = expensive). Size = relative GHG emissions. "
            "Gray dots = matched CAISO nodes. Purple = AB 617 communities."
        )


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
