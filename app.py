import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os

from data_processor import CAISODataProcessor
from analytics import LMPAnalytics
from chatbot import LMPChatbot
from s3_data_loader import S3DataLoader

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
        
        # Check current data status
        try:
            data_status = st.session_state.s3_loader.check_data_freshness()
            
            if 'error' in data_status:
                st.error(f"Error checking data: {data_status['error']}")
            else:
                if data_status['db_has_data']:
                    st.success("✅ Data loaded and ready")
                    st.metric("Records in Database", f"{data_status['db_records']:,}")
                    if data_status['latest_db_date']:
                        st.metric("Latest Data", data_status['latest_db_date'].strftime('%Y-%m-%d'))
                    st.session_state.data_loaded = True
                else:
                    st.warning("⚠️ No data in database")
                    st.info(f"Found {data_status['s3_files_available']} files in S3")
                
                # Data refresh option
                if st.button("🔄 Refresh Data from S3", help="Load latest data from S3 bucket"):
                    with st.spinner("Loading data from S3..."):
                        def progress_callback(current, total, message):
                            st.progress(current / total, text=message)
                        
                        result = st.session_state.s3_loader.load_all_data(progress_callback)
                        
                        if result['success']:
                            st.success(f"✅ Loaded {result['processed_files']} files with {result['total_records']:,} records")
                            st.session_state.data_loaded = True
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to load data: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            st.error(f"Error checking data status: {str(e)}")
        
        # Data summary from database (if data exists)
        try:
            summary = st.session_state.processor.get_data_summary_from_db()
            if summary and summary.get('total_records', 0) > 0:
                st.subheader("Database Summary")
                st.metric("Total Records", f"{summary.get('total_records', 0):,}")
                st.metric("Unique Nodes", summary.get('unique_nodes', 0))
                
                if summary.get('earliest_date') and summary.get('latest_date'):
                    st.metric("Date Range", f"{summary['earliest_date'].date()} to {summary['latest_date'].date()}")
                
                st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error loading data summary: {str(e)}")
    
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
        # Create tabs for different analysis views
        tab1, tab2, tab3, tab4 = st.tabs(["💬 AI Chat", "📊 Quick Analytics", "📈 Visualizations", "📋 Data Explorer"])
        
        with tab1:
            st.header("AI-Powered Analysis Chat")
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
            
            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("Ask", type="primary"):
                    if user_question:
                        with st.spinner("Analyzing your question..."):
                            try:
                                answer = st.session_state.chatbot.process_question(user_question)
                                st.session_state.chat_history.append((user_question, answer))
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error processing question: {str(e)}")
            
            with col2:
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        
        with tab2:
            st.header("Quick Analytics")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Price Statistics")
                try:
                    price_stats = st.session_state.analytics.get_price_statistics()
                    if not price_stats.empty:
                        st.dataframe(price_stats)
                    else:
                        st.info("No price statistics available")
                except Exception as e:
                    st.error(f"Error loading price statistics: {str(e)}")
                
                st.subheader("Top 10 Lowest Price Hours")
                try:
                    cheapest_hours = st.session_state.analytics.get_cheapest_hours(10)
                    if not cheapest_hours.empty:
                        st.dataframe(cheapest_hours)
                    else:
                        st.info("No cheapest hours data available")
                except Exception as e:
                    st.error(f"Error loading cheapest hours: {str(e)}")
            
            with col2:
                st.subheader("Node Summary")
                try:
                    node_summary = st.session_state.analytics.get_node_summary()
                    if not node_summary.empty:
                        st.dataframe(node_summary)
                    else:
                        st.info("No node summary available")
                except Exception as e:
                    st.error(f"Error loading node summary: {str(e)}")
                
                st.subheader("Hourly Average Prices")
                try:
                    hourly_avg = st.session_state.analytics.get_hourly_averages()
                    if not hourly_avg.empty:
                        st.line_chart(hourly_avg['mw'])
                    else:
                        st.info("No hourly averages available")
                except Exception as e:
                    st.error(f"Error loading hourly averages: {str(e)}")
        
        with tab3:
            st.header("Basic Visualizations")
            st.info("📊 Advanced time series visualizations with node selection will be available after full database integration. For now, please use the Quick Analytics and AI Chat features.")
            
            # Placeholder for future database-driven visualizations
            try:
                # Get sample data for basic chart
                hourly_avg = st.session_state.analytics.get_hourly_averages()
                if not hourly_avg.empty:
                    st.subheader("Hourly Price Patterns")
                    st.line_chart(hourly_avg.set_index('hour')['mw'])
                
                # Simple price statistics chart
                price_stats = st.session_state.analytics.get_price_statistics()
                if not price_stats.empty and len(price_stats) > 0:
                    st.subheader("Average Price by Node (Top 10)")
                    top_nodes = price_stats.head(10)
                    st.bar_chart(top_nodes.set_index('node')['mean'])
                    
            except Exception as e:
                st.error(f"Error loading visualizations: {str(e)}")
        
        with tab4:
            st.header("Data Explorer")
            
            # Basic data exploration with database queries
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Quick Data Samples")
                
                # Sample cheapest hours
                try:
                    sample_data = st.session_state.analytics.get_cheapest_hours(20)
                    if not sample_data.empty:
                        st.write("**20 Cheapest Hours:**")
                        st.dataframe(sample_data, height=300)
                    else:
                        st.info("No data available")
                except Exception as e:
                    st.error(f"Error loading sample data: {str(e)}")
            
            with col2:
                st.subheader("Node Statistics")
                
                try:
                    node_stats = st.session_state.analytics.get_price_statistics()
                    if not node_stats.empty:
                        st.dataframe(node_stats, height=300)
                    else:
                        st.info("No node statistics available")
                except Exception as e:
                    st.error(f"Error loading node statistics: {str(e)}")
            
            st.info("🔧 Advanced filtering and data export features will be available after completing the full database integration. Use the AI Chat feature to query specific data subsets.")

if __name__ == "__main__":
    main()
