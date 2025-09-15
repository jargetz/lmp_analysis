import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import io
import zipfile
import os

from data_processor import CAISODataProcessor
from analytics import LMPAnalytics
from chatbot import LMPChatbot

def main():
    st.set_page_config(
        page_title="CAISO LMP Analysis Tool",
        page_icon="⚡",
        layout="wide"
    )
    
    st.title("⚡ CAISO LMP Analysis Tool")
    st.markdown("Upload CAISO Day Ahead LMP data and analyze electricity pricing with AI-powered insights.")
    
    # Initialize session state
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processor' not in st.session_state:
        st.session_state.processor = CAISODataProcessor()
    if 'analytics' not in st.session_state:
        st.session_state.analytics = LMPAnalytics()
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = LMPChatbot()
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Sidebar for data upload and basic info
    with st.sidebar:
        st.header("Data Upload")
        
        uploaded_files = st.file_uploader(
            "Upload CAISO LMP ZIP files",
            type=['zip'],
            accept_multiple_files=True,
            help="Upload ZIP files containing CAISO Day Ahead LMP CSV data"
        )
        
        if uploaded_files:
            if st.button("Process Files"):
                with st.spinner("Processing CAISO data..."):
                    try:
                        all_data = []
                        for uploaded_file in uploaded_files:
                            # Extract and process each zip file
                            with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                                for file_name in zip_ref.namelist():
                                    if file_name.endswith('.csv'):
                                        with zip_ref.open(file_name) as csv_file:
                                            content = csv_file.read().decode('utf-8')
                                            df = st.session_state.processor.process_csv_content(content)
                                            if df is not None and not df.empty:
                                                all_data.append(df)
                        
                        if all_data:
                            st.session_state.data = pd.concat(all_data, ignore_index=True)
                            st.session_state.data = st.session_state.processor.clean_and_validate(st.session_state.data)
                            st.success(f"Successfully processed {len(all_data)} files with {len(st.session_state.data)} records")
                        else:
                            st.error("No valid CSV data found in uploaded files")
                            
                    except Exception as e:
                        st.error(f"Error processing files: {str(e)}")
        
        # Data summary in sidebar
        if st.session_state.data is not None:
            st.header("Data Summary")
            st.metric("Total Records", len(st.session_state.data))
            unique_nodes = st.session_state.data['NODE'].nunique() if 'NODE' in st.session_state.data.columns else 0
            st.metric("Unique Nodes", int(unique_nodes))
            
            if 'INTERVALSTARTTIME_GMT' in st.session_state.data.columns:
                date_range = st.session_state.data['INTERVALSTARTTIME_GMT'].dt.date
                st.metric("Date Range", f"{date_range.min()} to {date_range.max()}")
    
    # Main content area
    if st.session_state.data is None:
        st.info("👆 Please upload CAISO LMP ZIP files to begin analysis")
        
        # Show sample questions that can be asked
        st.header("Sample Analysis Questions")
        st.markdown("""
        Once you upload data, you can ask questions like:
        - What are the 10 cheapest hours at node SLAP_PGE2?
        - Show me the nodes with the lowest 10% of prices
        - Which nodes have the lowest congestion component during peak hours?
        - What are the average prices by hour of day?
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
                                answer = st.session_state.chatbot.process_question(
                                    user_question, 
                                    st.session_state.data
                                )
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
                if 'MW' in st.session_state.data.columns:
                    price_stats = st.session_state.analytics.get_price_statistics(st.session_state.data)
                    st.dataframe(price_stats)
                
                st.subheader("Top 10 Lowest Price Hours")
                cheapest_hours = st.session_state.analytics.get_cheapest_hours(st.session_state.data, 10)
                if not cheapest_hours.empty:
                    st.dataframe(cheapest_hours)
            
            with col2:
                st.subheader("Node Summary")
                node_summary = st.session_state.analytics.get_node_summary(st.session_state.data)
                st.dataframe(node_summary)
                
                st.subheader("Hourly Average Prices")
                hourly_avg = st.session_state.analytics.get_hourly_averages(st.session_state.data)
                if not hourly_avg.empty:
                    st.line_chart(hourly_avg['MW'])
        
        with tab3:
            st.header("Price Visualizations")
            
            # Time series chart
            if 'INTERVALSTARTTIME_GMT' in st.session_state.data.columns and 'MW' in st.session_state.data.columns:
                st.subheader("Price Time Series")
                
                # Node selection for time series
                available_nodes = st.session_state.data['NODE'].unique()[:10]  # Limit to first 10 nodes for performance
                selected_nodes = st.multiselect(
                    "Select nodes to display:",
                    available_nodes,
                    default=available_nodes[:3] if len(available_nodes) >= 3 else available_nodes
                )
                
                if selected_nodes:
                    filtered_data = st.session_state.data[st.session_state.data['NODE'].isin(selected_nodes)]
                    
                    fig = px.line(
                        filtered_data,
                        x='INTERVALSTARTTIME_GMT',
                        y='MW',
                        color='NODE',
                        title='LMP Prices Over Time by Node'
                    )
                    fig.update_layout(
                        xaxis_title="Time",
                        yaxis_title="Price ($/MWh)",
                        height=500
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Price distribution
                st.subheader("Price Distribution")
                fig_hist = px.histogram(
                    st.session_state.data,
                    x='MW',
                    nbins=50,
                    title='LMP Price Distribution'
                )
                fig_hist.update_layout(
                    xaxis_title="Price ($/MWh)",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig_hist, use_container_width=True)
        
        with tab4:
            st.header("Data Explorer")
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            selected_nodes_filter = []
            date_range = None
            price_range = None
            
            with col1:
                if 'NODE' in st.session_state.data.columns:
                    selected_nodes_filter = st.multiselect(
                        "Filter by Node:",
                        st.session_state.data['NODE'].unique(),
                        default=[]
                    )
            
            with col2:
                if 'INTERVALSTARTTIME_GMT' in st.session_state.data.columns:
                    date_range = st.date_input(
                        "Date Range:",
                        value=(
                            st.session_state.data['INTERVALSTARTTIME_GMT'].dt.date.min(),
                            st.session_state.data['INTERVALSTARTTIME_GMT'].dt.date.max()
                        ),
                        min_value=st.session_state.data['INTERVALSTARTTIME_GMT'].dt.date.min(),
                        max_value=st.session_state.data['INTERVALSTARTTIME_GMT'].dt.date.max()
                    )
            
            with col3:
                if 'MW' in st.session_state.data.columns:
                    price_range = st.slider(
                        "Price Range ($/MWh):",
                        min_value=float(st.session_state.data['MW'].min()),
                        max_value=float(st.session_state.data['MW'].max()),
                        value=(
                            float(st.session_state.data['MW'].min()),
                            float(st.session_state.data['MW'].max())
                        )
                    )
            
            # Apply filters
            filtered_data = st.session_state.data.copy()
            
            if selected_nodes_filter:
                filtered_data = filtered_data[filtered_data['NODE'].isin(selected_nodes_filter)]
            
            if 'INTERVALSTARTTIME_GMT' in filtered_data.columns and date_range is not None and len(date_range) == 2:
                filtered_data = filtered_data[
                    (filtered_data['INTERVALSTARTTIME_GMT'].dt.date >= date_range[0]) &
                    (filtered_data['INTERVALSTARTTIME_GMT'].dt.date <= date_range[1])
                ]
            
            if 'MW' in filtered_data.columns and price_range is not None:
                filtered_data = filtered_data[
                    (filtered_data['MW'] >= price_range[0]) &
                    (filtered_data['MW'] <= price_range[1])
                ]
            
            st.dataframe(
                filtered_data,
                use_container_width=True,
                height=400
            )
            
            # Download filtered data
            csv = filtered_data.to_csv(index=False)
            st.download_button(
                label="Download Filtered Data as CSV",
                data=csv,
                file_name=f"caiso_lmp_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
