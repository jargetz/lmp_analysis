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
from s3_data_loader import S3DataLoader
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
    create_month_hour_heatmap
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
    if 's3_loader' not in st.session_state:
        st.session_state.s3_loader = S3DataLoader()
    
    # Sidebar for data management and info
    with st.sidebar:
        st.header("Data Status")
        
        # Cache database status check (runs once per session)
        if 'db_summary' not in st.session_state:
            try:
                st.session_state.db_summary = st.session_state.processor.get_data_summary_from_db()
            except Exception:
                st.session_state.db_summary = None
        
        summary = st.session_state.db_summary
        try:
            if summary and summary.get('total_records', 0) > 0:
                st.success("✅ Data loaded and ready")
                st.metric("Records in Database", f"{summary.get('total_records', 0):,}")
                if summary.get('latest_date'):
                    st.metric("Latest Data", summary['latest_date'].strftime('%Y-%m-%d'))
                st.session_state.data_loaded = True
            else:
                st.warning("⚠️ No data in database")
                st.info("Use the admin button below to load data from S3")
                st.session_state.data_loaded = False
            
            # Admin data refresh option (explicit action)
            st.subheader("🔧 Admin Functions")
            
            # Simple admin protection
            admin_password = st.text_input("Admin Password:", type="password", help="Required for S3 data operations")
            
            if admin_password == os.getenv('ADMIN_PASSWORD', 'admin123'):
                if st.button("🔄 Load Data from S3", help="Admin: Download and process all CAISO files from S3 bucket"):
                        def progress_callback(current, total, message):
                            st.progress(current / total, text=message)
                        
                        result = st.session_state.s3_loader.load_all_data(progress_callback)
                        
                        if result['success']:
                            # Show data loading results
                            success_msg = f"✅ **Data Loading**: Processed {result['processed_files']} files, skipped {result.get('skipped_files', 0)} duplicates"
                            if result['total_records'] > 0:
                                success_msg += f", added {result['total_records']:,} new records"
                            st.success(success_msg)
                            
                            # Show preprocessing results
                            preprocessing = result.get('preprocessing', {})
                            if preprocessing.get('success'):
                                preprocess_msg = f"✅ **B6/B8 Preprocessing**: Processed {preprocessing.get('processed_dates', 0)} days"
                                preprocess_msg += f", created {preprocessing.get('total_b6_records', 0)} B6 and {preprocessing.get('total_b8_records', 0)} B8 records"
                                st.success(preprocess_msg)
                            else:
                                st.warning(f"⚠️ **Preprocessing Issues**: {preprocessing.get('error', 'Unknown preprocessing error')}")
                            
                            # Show any errors
                            if result.get('errors'):
                                st.warning(f"⚠️ **Encountered {len(result['errors'])} errors**:")
                                for error in result['errors'][:3]:  # Show first 3 errors
                                    st.text(f"• {error}")
                                if len(result['errors']) > 3:
                                    st.text(f"... and {len(result['errors']) - 3} more errors")
                            
                            st.session_state.data_loaded = True
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to load data: {result.get('error', 'Unknown error')}")
            else:
                st.button("🔄 Load Data from S3", disabled=True, help="Enter admin password to enable")
                
        except Exception as e:
            st.error(f"Error checking database status: {str(e)}")
            st.session_state.data_loaded = False
        
        # Additional database details (if data exists) - use cached summary
        if st.session_state.data_loaded and summary:
            st.subheader("Database Details")
            st.metric("Unique Nodes", summary.get('unique_nodes', 0))
            
            if summary.get('earliest_date') and summary.get('latest_date'):
                st.markdown("**Date Range**")
                st.markdown(f"📅 Start: {summary['earliest_date']}")
                st.markdown(f"📅 End: {summary['latest_date']}")
    
    # Main content area
    if not st.session_state.data_loaded:
        st.info("👈 Click 'Refresh Data from S3' in the sidebar to load CAISO data")
        
        # Show sample questions that can be asked
        st.header("What You Can Ask")
        st.markdown("""
        Once data is loaded, you can ask questions like:
        - What are the 10 cheapest hours at node SLAP_PGE2?
        - Show me the nodes with the lowest 10% of prices (B10)
        - Which nodes have the lowest congestion component during peak hours?
        - What are the average prices by hour of day?
        - Show me the B6 and B8 hours (cheapest 6 and 8 hours) for each node
        - Find the hours with the highest price volatility
        """)
        
    else:
        # Create tabs: Dashboard first (primary), then AI Assistant
        tab_dashboard, tab_ai = st.tabs(["📊 Dashboard", "💬 AI Assistant"])
        
        # =====================================================================
        # DASHBOARD TAB - Primary interface for BX analysis with zone filtering
        # =====================================================================
        with tab_dashboard:
            render_dashboard_tab()
        
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
    # Preload all cached data at startup (runs once)
    if 'dashboard_initialized' not in st.session_state:
        with st.spinner("Loading dashboard data..."):
            bx_calc_init = BXCalculator()
            st.session_state.bx_calc = bx_calc_init
            st.session_state.all_nodes = bx_calc_init.get_all_nodes()
            st.session_state.available_years = bx_calc_init.get_available_years() or [2024]
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
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    # Initialize filter variables
    selected_zone = None
    selected_nodes = []
    
    with filter_col1:
        if analysis_mode == "By Zone":
            # Zone mode shows all zones - no individual zone filter needed
            st.markdown("**Zone Comparison**")
            st.caption("Showing NP15, SP15, ZP26, and Overall averages")
        else:
            # Quick add by prefix
            prefix_col, add_col = st.columns([3, 1])
            with prefix_col:
                prefix = st.text_input(
                    "Add nodes by prefix",
                    placeholder="e.g., PGE, SCE, SLAP",
                    help="Type a prefix and click Add to select all matching nodes",
                    key="node_prefix"
                )
            with add_col:
                st.markdown("<br>", unsafe_allow_html=True)  # Align button
                if st.button("Add All", key="add_prefix"):
                    if prefix and len(prefix) >= 2:
                        matching = [n for n in st.session_state.all_nodes if n.upper().startswith(prefix.upper())]
                        if matching:
                            current = st.session_state.get('selected_nodes_list', [])
                            updated = list(set(current + matching))
                            st.session_state.selected_nodes_list = updated
                            st.rerun()
            
            # Initialize selected nodes from session state
            if 'selected_nodes_list' not in st.session_state:
                st.session_state.selected_nodes_list = []
            
            # Multiselect with built-in autocomplete (type to filter)
            # Nodes are preloaded at dashboard startup
            selected_nodes = st.multiselect(
                "Selected Nodes",
                options=st.session_state.all_nodes,
                default=st.session_state.selected_nodes_list,
                placeholder="Type to search or use prefix above...",
                help="Select individual nodes or use prefix above to add many at once",
                key="node_multiselect"
            )
            
            # Sync selection back to session state
            st.session_state.selected_nodes_list = selected_nodes
            
            # Show count and clear button
            if selected_nodes:
                st.caption(f"{len(selected_nodes)} nodes selected")
                if st.button("Clear All", key="clear_nodes"):
                    st.session_state.selected_nodes_list = []
                    st.rerun()
    
    with filter_col2:
        # BX selector
        selected_bx = st.selectbox(
            "BX Hours",
            options=SUPPORTED_BX_VALUES,
            index=4,  # Default to B8
            format_func=lambda x: f"B{x} (Cheapest {x} hours)",
            help="Number of cheapest hours to analyze"
        )
    
    with filter_col3:
        # Time period selector (Annual/Monthly)
        time_period = st.selectbox(
            "Time Period",
            options=["Annual", "Monthly"],
            help="Choose annual or monthly view"
        )
    
    with filter_col4:
        # Year/Month selector (years preloaded at startup)
        available_years = st.session_state.available_years
        
        if time_period == "Annual":
            selected_year = st.selectbox(
                "Year",
                options=available_years,
                help="Select year"
            )
            selected_month = None
        else:
            # Year selector for monthly view
            selected_year = st.selectbox(
                "Year",
                options=available_years,
                key="monthly_year",
                help="Select year"
            )
            # Month selector
            month_options = ["January", "February", "March", "April", "May", "June", 
                           "July", "August", "September", "October", "November", "December"]
            selected_month_name = st.selectbox(
                "Month",
                options=month_options,
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
            # Cache zone stats with key based on filters
            cache_key = f"zone_stats_{selected_bx}_{selected_year}_{time_period}_{selected_month}"
            if cache_key not in st.session_state:
                st.session_state[cache_key] = bx_calc.get_all_zones_bx_average(
                    bx=selected_bx,
                    year=selected_year,
                    time_period=time_period,
                    month=selected_month
                )
            zone_stats = st.session_state[cache_key]
            
            # Display zones in columns: NP15, SP15, ZP26, Overall
            zone_cols = st.columns(4)
            zone_order = ['NP15', 'SP15', 'ZP26', 'Overall']
            
            for col, zone_name in zip(zone_cols, zone_order):
                with col:
                    stats = zone_stats.get(zone_name, {})
                    if stats.get('success') and stats.get('avg_price'):
                        st.metric(
                            zone_name,
                            f"${stats['avg_price']:.2f}/MWh",
                            help=f"Nodes: {stats.get('node_count', 0):,}"
                        )
                    else:
                        st.metric(zone_name, "N/A")
            
            # Month x Hour heatmap - requires raw data (stored in S3 parquet)
            st.subheader("Averages - Day Ahead LMP")
            st.caption("Month x Hour heatmap is computed from raw data stored in S3. This feature is coming soon.")
            
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
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_mode == "By Node Selection":
            # Node selection mode - show stats for selected nodes
            if not selected_nodes:
                st.info("Select one or more nodes above to see BX statistics.")
            else:
                if time_period == "Annual":
                    bx_stats = bx_calc.get_annual_bx_average(
                        bx=selected_bx,
                        year=selected_year,
                        nodes=selected_nodes
                    )
                else:
                    from calendar import monthrange
                    start_date = date(selected_year, selected_month, 1)
                    _, last_day = monthrange(selected_year, selected_month)
                    end_date = date(selected_year, selected_month, last_day)
                    bx_stats = bx_calc.get_bx_average(
                        bx=selected_bx,
                        nodes=selected_nodes,
                        start_date=start_date,
                        end_date=end_date
                    )
                
                if bx_stats.get('success') and bx_stats.get('avg_price'):
                    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                    
                    with stat_col1:
                        st.metric("Average", f"${bx_stats['avg_price']:.2f}/MWh")
                    with stat_col2:
                        st.metric("Min", f"${bx_stats['min_price']:.2f}/MWh" if bx_stats.get('min_price') else "N/A")
                    with stat_col3:
                        st.metric("Max", f"${bx_stats['max_price']:.2f}/MWh" if bx_stats.get('max_price') else "N/A")
                    with stat_col4:
                        st.metric("Nodes", f"{bx_stats.get('node_count', 0):,}")
                    
                    # Hourly price chart for selected nodes
                    node_hourly_key = f"hourly_nodes_{hash(tuple(sorted(selected_nodes)))}_{selected_year}"
                    if node_hourly_key not in st.session_state:
                        st.session_state[node_hourly_key] = bx_calc.get_hourly_averages_for_nodes(
                            nodes=selected_nodes,
                            year=selected_year
                        )
                    node_hourly_data = st.session_state[node_hourly_key]
                    
                    if node_hourly_data:
                        fig = create_node_hourly_chart(node_hourly_data, title=f'Hourly Price Average ({len(selected_nodes)} nodes, {selected_year})')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # BX trend chart per node (or average if many nodes)
                    if len(selected_nodes) <= 10:
                        node_trend_key = f"bx_trend_nodes_{hash(tuple(sorted(selected_nodes)))}_{selected_bx}_{selected_year}"
                        if node_trend_key not in st.session_state:
                            st.session_state[node_trend_key] = bx_calc.get_bx_trend_per_node(
                                bx=selected_bx,
                                nodes=selected_nodes,
                                year=selected_year,
                                aggregation='monthly'
                            )
                        node_trend_data = st.session_state[node_trend_key]
                        
                        if node_trend_data:
                            fig = create_node_bx_trend_chart(node_trend_data, bx_type=selected_bx)
                            st.plotly_chart(fig, use_container_width=True)
                    else:
                        # For many nodes, show average trend only
                        avg_trend = bx_calc.get_bx_trend(
                            bx=selected_bx,
                            start_date=date(selected_year, 1, 1),
                            end_date=date(selected_year, 12, 31),
                            nodes=selected_nodes,
                            aggregation='monthly'
                        )
                        if avg_trend:
                            import pandas as pd
                            df = pd.DataFrame(avg_trend)
                            df.rename(columns={'date': 'opr_dt'}, inplace=True)
                            fig = create_bx_trend_chart(df, bx_type=selected_bx, title=f'B{selected_bx} Average Trend ({len(selected_nodes)} nodes)')
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Box plot for node comparison (outlier detection)
                    if len(selected_nodes) > 1:
                        box_key = f"box_nodes_{hash(tuple(sorted(selected_nodes)))}_{selected_bx}_{selected_year}"
                        if box_key not in st.session_state:
                            st.session_state[box_key] = bx_calc.get_node_summary_statistics(
                                bx=selected_bx,
                                nodes=selected_nodes,
                                year=selected_year
                            )
                        box_data = st.session_state[box_key]
                        
                        if box_data:
                            fig = create_node_box_plot(box_data, title=f'B{selected_bx} Price Distribution by Node ({selected_year})')
                            st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No data found for selected nodes.")
    
    except Exception as e:
        st.warning(f"Could not load BX statistics: {str(e)}")
        st.info("Make sure LMP data is loaded and BX calculations have been run.")


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
