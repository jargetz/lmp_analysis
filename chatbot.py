import json
import os
import pandas as pd
from openai import OpenAI
from analytics import LMPAnalytics

class LMPChatbot:
    """AI-powered chatbot for natural language querying of LMP data"""
    
    def __init__(self):
        # the newest OpenAI model is "gpt-5" which was released August 7, 2025.
        # do not change this unless explicitly requested by the user
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", "default_key"))
        self.analytics = LMPAnalytics()
        
    def process_question(self, question, data):
        """Process a natural language question about LMP data"""
        try:
            # First, analyze the question to understand intent
            analysis_result = self._analyze_question_intent(question, data)
            
            # Execute the appropriate analysis
            result = self._execute_analysis(analysis_result, data)
            
            # Generate a natural language response
            response = self._generate_response(question, result, analysis_result)
            
            return response
            
        except Exception as e:
            return f"I encountered an error processing your question: {str(e)}. Please try rephrasing your question or check if the required data is available."
    
    def _analyze_question_intent(self, question, data):
        """Analyze the user's question to determine what analysis to perform"""
        
        # Get basic data info for context
        data_context = self._get_data_context(data)
        
        system_prompt = """You are an expert in electricity market analysis, specifically CAISO LMP (Locational Marginal Price) data.
        
        Analyze the user's question and determine what type of analysis they want. Respond with JSON in this exact format:
        
        {
            "analysis_type": "one of: cheapest_hours, price_percentile, congestion_analysis, peak_analysis, price_statistics, hourly_patterns, price_spikes, node_comparison, general_query",
            "parameters": {
                "n_hours": integer (if asking for top/bottom N),
                "percentile": integer (if asking for percentile analysis),
                "nodes": ["list of specific nodes if mentioned"],
                "time_period": "specific time period if mentioned",
                "comparison_type": "type of comparison requested"
            },
            "confidence": number between 0 and 1
        }
        
        Available analysis types:
        - cheapest_hours: Finding lowest price hours
        - price_percentile: Nodes in certain price percentiles  
        - congestion_analysis: Congestion component analysis
        - peak_analysis: Peak vs off-peak comparisons
        - price_statistics: General price statistics
        - hourly_patterns: Patterns by hour of day
        - price_spikes: Unusual price events
        - node_comparison: Comparing specific nodes
        - general_query: Other analytical questions
        """
        
        user_message = f"""
        Question: {question}
        
        Available data context:
        - Total records: {data_context['total_records']}
        - Unique nodes: {data_context['unique_nodes']}
        - Date range: {data_context['date_range']}
        - Available columns: {data_context['columns']}
        - Sample nodes: {data_context['sample_nodes']}
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            if content:
                return json.loads(content)
            else:
                return self._fallback_intent_analysis(question)
            
        except Exception as e:
            # Fallback to basic analysis if OpenAI fails
            return self._fallback_intent_analysis(question)
    
    def _execute_analysis(self, analysis_result, data):
        """Execute the determined analysis on the data"""
        analysis_type = analysis_result.get('analysis_type', 'general_query')
        params = analysis_result.get('parameters', {})
        
        try:
            if analysis_type == 'cheapest_hours':
                n_hours = params.get('n_hours', 10)
                nodes = params.get('nodes', [])
                
                if len(nodes) == 1:
                    return self.analytics.get_cheapest_hours(data, n_hours, node=nodes[0])
                elif len(nodes) > 1:
                    return self.analytics.get_cheapest_hours(data, n_hours, aggregate_nodes=nodes)
                else:
                    return self.analytics.get_cheapest_hours(data, n_hours)
            
            elif analysis_type == 'price_percentile':
                percentile = params.get('percentile', 10)
                return self.analytics.get_nodes_by_price_percentile(data, percentile)
            
            elif analysis_type == 'congestion_analysis':
                if 'MCC' in data.columns:
                    n_hours = params.get('n_hours', 20)
                    return self.analytics.get_lowest_congestion_hours(data, n_hours)
                else:
                    return pd.DataFrame(), "Congestion data (MCC) not available in this dataset"
            
            elif analysis_type == 'peak_analysis':
                return self.analytics.get_peak_vs_offpeak_analysis(data)
            
            elif analysis_type == 'price_statistics':
                return self.analytics.get_price_statistics(data)
            
            elif analysis_type == 'hourly_patterns':
                return self.analytics.get_hourly_averages(data)
            
            elif analysis_type == 'price_spikes':
                return self.analytics.detect_price_spikes(data)
            
            elif analysis_type == 'node_comparison':
                nodes = params.get('nodes', [])
                if nodes:
                    filtered_data = data[data['NODE'].isin(nodes)]
                    return self.analytics.get_price_statistics(filtered_data)
                else:
                    return self.analytics.get_node_summary(data)
            
            else:
                # General query - return basic statistics
                return self.analytics.get_price_statistics(data)
                
        except Exception as e:
            return pd.DataFrame(), f"Error executing analysis: {str(e)}"
    
    def _generate_response(self, original_question, analysis_result, intent_analysis):
        """Generate a natural language response based on the analysis results"""
        
        if isinstance(analysis_result, tuple):
            result_df, error_msg = analysis_result
        else:
            result_df = analysis_result
            error_msg = None
        
        if error_msg:
            return error_msg
        
        if result_df is None or (isinstance(result_df, pd.DataFrame) and result_df.empty):
            return "I couldn't find any data matching your query. Please check if the requested information is available in your dataset."
        
        # Convert DataFrame to readable format
        if isinstance(result_df, pd.DataFrame):
            # Limit results for readability
            display_df = result_df.head(20)
            
            # Generate summary
            summary = self._generate_summary(display_df, intent_analysis.get('analysis_type', 'general'))
            
            # Format as table
            table_str = display_df.to_string(index=False, float_format='%.2f')
            
            response = f"{summary}\n\nResults:\n```\n{table_str}\n```"
            
            if len(result_df) > 20:
                response += f"\n\n(Showing first 20 results out of {len(result_df)} total results)"
                
        else:
            response = str(result_df)
        
        return response
    
    def _generate_summary(self, df, analysis_type):
        """Generate a summary of the analysis results"""
        if df.empty:
            return "No results found for your query."
        
        summaries = {
            'cheapest_hours': f"Found {len(df)} cheapest hours. The lowest price was ${df['MW'].min():.2f}/MWh at {df.iloc[0]['NODE'] if 'NODE' in df.columns else 'multiple nodes'}.",
            
            'price_percentile': f"Found {len(df)} nodes in the requested price percentile. Average price range: ${df['avg_price'].min():.2f} - ${df['avg_price'].max():.2f}/MWh.",
            
            'congestion_analysis': f"Analyzed {len(df)} hours with congestion data. Lowest congestion was ${df['MCC'].min():.2f}/MWh.",
            
            'peak_analysis': f"Peak vs off-peak analysis for {len(df)} nodes. Average peak premium varies significantly across nodes.",
            
            'price_statistics': f"Price statistics for {len(df)} nodes. Overall price range: ${df['min'].min():.2f} - ${df['max'].max():.2f}/MWh.",
            
            'hourly_patterns': f"Hourly price patterns show average prices ranging from ${df['MW'].min():.2f} to ${df['MW'].max():.2f}/MWh.",
            
            'price_spikes': f"Detected {len(df)} price spike events across the dataset.",
            
            'node_comparison': f"Compared statistics across {len(df)} nodes.",
        }
        
        return summaries.get(analysis_type, f"Analysis completed with {len(df)} results.")
    
    def _get_data_context(self, data):
        """Get context information about the dataset"""
        if data is None or data.empty:
            return {
                'total_records': 0,
                'unique_nodes': 0,
                'date_range': 'No data',
                'columns': [],
                'sample_nodes': []
            }
        
        return {
            'total_records': len(data),
            'unique_nodes': data['NODE'].nunique() if 'NODE' in data.columns else 0,
            'date_range': f"{data['INTERVALSTARTTIME_GMT'].min()} to {data['INTERVALSTARTTIME_GMT'].max()}" if 'INTERVALSTARTTIME_GMT' in data.columns else 'No date data',
            'columns': list(data.columns),
            'sample_nodes': data['NODE'].unique()[:5].tolist() if 'NODE' in data.columns else []
        }
    
    def _fallback_intent_analysis(self, question):
        """Fallback analysis when OpenAI is not available"""
        question_lower = question.lower()
        
        # Simple keyword-based intent detection
        if any(word in question_lower for word in ['cheapest', 'lowest', 'minimum']):
            return {
                'analysis_type': 'cheapest_hours',
                'parameters': {'n_hours': 10},
                'confidence': 0.7
            }
        elif any(word in question_lower for word in ['percentile', '%']):
            return {
                'analysis_type': 'price_percentile',
                'parameters': {'percentile': 10},
                'confidence': 0.7
            }
        elif any(word in question_lower for word in ['congestion', 'mcc']):
            return {
                'analysis_type': 'congestion_analysis',
                'parameters': {'n_hours': 20},
                'confidence': 0.7
            }
        elif any(word in question_lower for word in ['peak', 'hour']):
            return {
                'analysis_type': 'hourly_patterns',
                'parameters': {},
                'confidence': 0.6
            }
        else:
            return {
                'analysis_type': 'general_query',
                'parameters': {},
                'confidence': 0.5
            }
