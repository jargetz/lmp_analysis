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
        
        # Check database data status only (fast, no S3 checks)
        try:
            summary = st.session_state.processor.get_data_summary_from_db()
            
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
        
        # Additional database details (if data exists)
        if st.session_state.data_loaded:
            try:
                summary = st.session_state.processor.get_data_summary_from_db()
                if summary:
                    st.subheader("Database Details")
                    st.metric("Unique Nodes", summary.get('unique_nodes', 0))
                    
                    if summary.get('earliest_date') and summary.get('latest_date'):
                        st.metric("Date Range", f"{summary['earliest_date']} to {summary['latest_date']}")
                        
            except Exception as e:
                st.error(f"Error loading database details: {str(e)}")
    
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
    st.header("LMP Dashboard")
    st.markdown("Analyze electricity pricing by zone and BX hours")
    
    # Filter Panel
    st.subheader("Filters")
    filter_col1, filter_col2, filter_col3 = st.columns(3)
    
    with filter_col1:
        # Zone selector
        try:
            mapper = NodeZoneMapper()
            available_zones = mapper.get_available_zones()
            if not available_zones:
                available_zones = VALID_ZONES
        except Exception:
            available_zones = VALID_ZONES
        
        selected_zone = st.selectbox(
            "Zone",
            options=["All Zones"] + available_zones,
            help="Filter results by CAISO zone"
        )
        if selected_zone == "All Zones":
            selected_zone = None
    
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
        # Date range (placeholder - will use actual data range)
        st.date_input(
            "Date Range",
            value=(date.today() - timedelta(days=30), date.today()),
            help="Filter by date range"
        )
    
    st.divider()
    
    # Summary Statistics Cards
    st.subheader("Summary Statistics")
    
    try:
        bx_calc = BXCalculator()
        bx_stats = bx_calc.get_bx_average(
            bx=selected_bx,
            zone=selected_zone
        )
        
        if bx_stats.get('success') and bx_stats.get('avg_price'):
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.metric(
                    f"Average B{selected_bx} Price",
                    f"${bx_stats['avg_price']:.2f}/MWh"
                )
            
            with stat_col2:
                st.metric(
                    "Min B{} Price".format(selected_bx),
                    f"${bx_stats['min_price']:.2f}/MWh" if bx_stats.get('min_price') else "N/A"
                )
            
            with stat_col3:
                st.metric(
                    "Max B{} Price".format(selected_bx),
                    f"${bx_stats['max_price']:.2f}/MWh" if bx_stats.get('max_price') else "N/A"
                )
            
            with stat_col4:
                st.metric(
                    "Nodes Analyzed",
                    f"{bx_stats.get('node_count', 0):,}"
                )
        else:
            st.info("No BX data available. Load data and run BX calculations first.")
            
            # Show how to calculate BX data
            with st.expander("How to calculate BX data"):
                st.markdown("""
                BX data needs to be pre-calculated from the raw LMP data.
                
                Once LMP data is loaded, run the BX calculator:
                ```python
                from bx_calculator import BXCalculator
                calc = BXCalculator()
                calc.calculate_bx_for_date_range(start_date, end_date)
                ```
                """)
    
    except Exception as e:
        st.warning(f"Could not load BX statistics: {str(e)}")
        st.info("Make sure LMP data is loaded and BX calculations have been run.")
    
    st.divider()
    
    # Quick Analytics Section
    st.subheader("Quick Analytics")
    
    analytics_col1, analytics_col2 = st.columns(2)
    
    with analytics_col1:
        st.markdown("**Price Statistics**")
        try:
            price_stats = st.session_state.analytics.get_price_statistics()
            if not price_stats.empty:
                st.dataframe(price_stats.head(10), use_container_width=True)
            else:
                st.info("No price statistics available")
        except Exception as e:
            st.error(f"Error loading price statistics: {str(e)}")
    
    with analytics_col2:
        st.markdown("**Hourly Price Patterns**")
        try:
            hourly_avg = st.session_state.analytics.get_hourly_averages()
            if not hourly_avg.empty:
                fig = px.line(
                    hourly_avg, 
                    x='hour', 
                    y='mw',
                    title='Average Price by Hour',
                    labels={'hour': 'Hour of Day', 'mw': 'Price ($/MWh)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No hourly data available")
        except Exception as e:
            st.error(f"Error loading hourly averages: {str(e)}")


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
