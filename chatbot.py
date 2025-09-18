import json
import os
import pandas as pd
from openai import OpenAI
from analytics import LMPAnalytics, get_registered_analytics
from database import DatabaseManager
import logging

class LMPChatbot:
    """AI-powered chatbot for natural language querying of LMP data with PostgreSQL backend"""
    
    def __init__(self):
        # Using GPT-4 for reliable natural language processing
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.analytics = LMPAnalytics()
        self.db = DatabaseManager()
        self.logger = logging.getLogger(__name__)
        
    def process_question(self, question):
        """Process a natural language question about LMP data from database"""
        analysis_result = None
        try:
            # First, analyze the question to understand intent
            analysis_result = self._analyze_question_intent(question)
            self.logger.info(f"Question analysis result: {analysis_result}")
            
            # Execute the appropriate analysis
            result = self._execute_analysis(analysis_result)
            
            # Generate a natural language response
            response = self._generate_response(question, result, analysis_result)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing question: {str(e)}")
            self.logger.error(f"Analysis result was: {analysis_result}")
            return f"I encountered an error processing your question: {str(e)}. Please try rephrasing your question or check if the required data is available."
    
    def _analyze_question_intent(self, question):
        """Analyze the user's question to determine what analysis to perform"""
        
        # Get basic data info for context from database
        data_context = self._get_data_context_from_db()
        
        # Build system prompt dynamically from registered analytics
        registered_analytics = get_registered_analytics()
        
        # Create analysis type options and descriptions
        analysis_types = list(registered_analytics.keys())
        analysis_descriptions = []
        
        for method_name, metadata in registered_analytics.items():
            # Format example questions for better AI understanding
            examples = metadata.get('example_questions', [])
            example_text = f" Examples: {', '.join(examples[:2])}" if examples else ""
            analysis_descriptions.append(f"- {method_name}: {metadata['description']}{example_text}")
        
        # Add fallback general_query option
        if 'general_query' not in analysis_types:
            analysis_types.append('general_query')
            analysis_descriptions.append('- general_query: Other analytical questions not covered by specific methods')
        
        analysis_types_str = ', '.join(analysis_types)
        analysis_descriptions_str = '\n        '.join(analysis_descriptions)
        
        system_prompt = f"""You are an expert in electricity market analysis, specifically CAISO LMP (Locational Marginal Price) data.
        
        Analyze the user's question and determine what type of analysis they want. Respond with JSON in this exact format:
        
        {{
            "analysis_type": "one of: {analysis_types_str}",
            "parameters": {{
                "n_hours": integer (if asking for top/bottom N),
                "n_nodes": integer (if asking for top/bottom N nodes),
                "operational_date": "YYYY-MM-DD format if specific date mentioned",
                "operational_hour": integer 0-23 (if specific hour mentioned),
                "percentile": integer (if asking for percentile analysis),
                "nodes": ["list of specific nodes if mentioned"],
                "time_period": "specific time period if mentioned",
                "comparison_type": "type of comparison requested"
            }},
            "confidence": number between 0 and 1
        }}
        
        Available analysis types:
        {analysis_descriptions_str}
        
        IMPORTANT: If user asks for "nodes" with a specific "operational date" AND "operational hour", use "get_cheapest_nodes_by_hour"
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
                model="gpt-4o-mini",
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
    
    def _execute_analysis(self, analysis_result):
        """Execute the determined analysis using database queries"""
        analysis_type = analysis_result.get('analysis_type', 'general_query')
        params = analysis_result.get('parameters', {})
        
        self.logger.info(f"Executing analysis type: {analysis_type} with params: {params}")
        
        try:
            # Get registered analytics to find the appropriate method
            registered_analytics = get_registered_analytics()
            
            # Check if analysis_type is a registered method
            if analysis_type in registered_analytics:
                method_name = analysis_type
                # Get the method from analytics instance
                if hasattr(self.analytics, method_name):
                    method = getattr(self.analytics, method_name)
                    
                    # Prepare method arguments based on registered parameters
                    method_metadata = registered_analytics[method_name]
                    expected_params = method_metadata.get('parameters', [])
                    
                    # Build method arguments dynamically
                    method_args = {}
                    
                    # Map common parameter names
                    param_mapping = {
                        'n_hours': params.get('n_hours', 10),
                        'n_nodes': params.get('n_nodes', 10),
                        'operational_date': params.get('operational_date'),
                        'operational_hour': params.get('operational_hour'),
                        'percentile': params.get('percentile', 10),
                        'start_date': params.get('start_date'),
                        'end_date': params.get('end_date'),
                        'exclude_zero': True,  # Default for most methods
                        'during_cheap_hours': False,
                        'node': None,
                        'aggregate_nodes': None
                    }
                    
                    # Handle special parameter logic for specific methods
                    if method_name == 'get_cheapest_hours':
                        nodes = params.get('nodes', [])
                        if len(nodes) == 1:
                            method_args['node'] = nodes[0]
                        elif len(nodes) > 1:
                            method_args['aggregate_nodes'] = nodes
                        method_args['n_hours'] = params.get('n_hours', 10)
                        
                    elif method_name == 'get_cheapest_nodes_by_hour':
                        # Handle time_period parsing for this method
                        operational_date = params.get('operational_date')
                        operational_hour = params.get('operational_hour')
                        
                        # Parse operational_date if it's a string (from time_period)
                        if not operational_date and 'time_period' in params:
                            time_period = params.get('time_period', '')
                            if isinstance(time_period, str) and len(time_period) >= 10:
                                # Extract date part (YYYY-MM-DD)
                                operational_date = time_period[:10]
                                
                                # Try to extract hour from time_period if not provided
                                if not operational_hour and ':' in time_period:
                                    try:
                                        # Parse something like "2024-01-01 13:00:00"
                                        from datetime import datetime
                                        dt = datetime.fromisoformat(time_period.replace(' ', 'T') if 'T' not in time_period else time_period)
                                        operational_hour = dt.hour
                                    except:
                                        pass
                        
                        method_args = {
                            'n_nodes': params.get('n_nodes', 10),
                            'operational_date': operational_date,
                            'operational_hour': operational_hour,
                            'exclude_zero': True
                        }
                        
                    elif method_name == 'get_cheapest_operational_hours':
                        method_args = {
                            'n_hours': params.get('n_hours', 5),
                            'start_date': params.get('start_date'),
                            'end_date': params.get('end_date')
                        }
                        
                    elif method_name == 'get_lowest_congestion_hours':
                        method_args = {
                            'n_hours': params.get('n_hours', 20),
                            'during_cheap_hours': params.get('during_cheap_hours', False),
                            'start_date': params.get('start_date'),
                            'end_date': params.get('end_date')
                        }
                        
                    else:
                        # For other methods, map only the parameters they expect
                        for param_name in expected_params:
                            if param_name in param_mapping and param_mapping[param_name] is not None:
                                method_args[param_name] = param_mapping[param_name]
                    
                    # Remove None values to let method defaults take effect
                    method_args = {k: v for k, v in method_args.items() if v is not None}
                    
                    self.logger.info(f"Calling {method_name} with args: {method_args}")
                    
                    # Call the method dynamically
                    result = method(**method_args)
                    
                    # Handle special error cases for congestion analysis
                    if method_name == 'get_lowest_congestion_hours' and hasattr(result, 'empty') and result.empty:
                        return pd.DataFrame(), "Congestion data (MCC) not available in this dataset"
                    
                    return result
                    
                else:
                    self.logger.error(f"Method {method_name} not found in analytics class")
                    return pd.DataFrame(), f"Analysis method {method_name} not found"
            
            else:
                # Fallback for general queries or unrecognized analysis types
                if hasattr(self.analytics, 'get_price_statistics'):
                    return self.analytics.get_price_statistics()
                else:
                    return pd.DataFrame(), "No suitable analysis method found"
                
        except Exception as e:
            self.logger.error(f"Error executing analysis: {str(e)}")
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
        
        # Generate interpretation display
        interpretation = self._format_interpretation(intent_analysis)
        
        # Convert DataFrame to readable format
        if isinstance(result_df, pd.DataFrame):
            # Limit results for readability
            display_df = result_df.head(20)
            
            # Generate summary
            summary = self._generate_summary(display_df, intent_analysis.get('analysis_type', 'general'))
            
            # Format as table
            table_str = display_df.to_string(index=False, float_format='%.2f')
            
            response = f"{interpretation}\n\n{summary}\n\nResults:\n```\n{table_str}\n```"
            
            if len(result_df) > 20:
                response += f"\n\n(Showing first 20 results out of {len(result_df)} total results)"
                
        else:
            response = f"{interpretation}\n\n{str(result_df)}"
        
        return response
    
    def _format_interpretation(self, intent_analysis):
        """Format the AI's interpretation of the user's question in a user-friendly way"""
        analysis_type = intent_analysis.get('analysis_type', 'general_query')
        parameters = intent_analysis.get('parameters', {})
        confidence = intent_analysis.get('confidence', 0)
        
        # Get user-friendly names from registered analytics
        registered_analytics = get_registered_analytics()
        
        if analysis_type in registered_analytics:
            # Use the description from the registered analytics
            friendly_name = registered_analytics[analysis_type].get('description', analysis_type.replace('_', ' ').title())
        else:
            # Fallback for unregistered analysis types
            fallback_names = {
                'general_query': 'General Analysis'
            }
            friendly_name = fallback_names.get(analysis_type, analysis_type.replace('_', ' ').title())
        
        # Format parameters in a user-friendly way
        param_display = []
        for key, value in parameters.items():
            if value is not None and value != [] and value != '':
                if key == 'n_hours':
                    param_display.append(f"Number of results: {value}")
                elif key == 'n_nodes':
                    param_display.append(f"Number of nodes: {value}")
                elif key == 'operational_date':
                    param_display.append(f"Date: {value}")
                elif key == 'operational_hour':
                    param_display.append(f"Hour: {value}")
                elif key == 'percentile':
                    param_display.append(f"Percentile: {value}%")
                elif key == 'nodes':
                    if isinstance(value, list) and len(value) > 0:
                        node_str = ', '.join(value[:3])  # Show first 3 nodes
                        if len(value) > 3:
                            node_str += f" (and {len(value) - 3} more)"
                        param_display.append(f"Nodes: {node_str}")
                elif key == 'time_period':
                    param_display.append(f"Time period: {value}")
        
        params_str = " • ".join(param_display) if param_display else "All available data"
        confidence_str = f"{int(confidence * 100)}%" if confidence > 0 else "High"
        
        interpretation = f"""🤖 **How I interpreted your question:**
**Analysis Type:** {friendly_name}
**Parameters:** {params_str}
**Confidence:** {confidence_str}

---"""
        
        return interpretation
    
    def _generate_summary(self, df, analysis_type):
        """Generate a summary of the analysis results"""
        if df.empty:
            return "No results found for your query."
        
        # Use lazy evaluation to avoid KeyErrors from accessing columns that don't exist
        if analysis_type == 'cheapest_hours':
            if 'mw' in df.columns and len(df) > 0:
                return f"Found {len(df)} cheapest hours. The lowest price was ${df['mw'].min():.2f}/MWh at {df.iloc[0]['node']}."
            else:
                return f"Found {len(df)} results for cheapest hours analysis."
                
        elif analysis_type == 'cheapest_operational_hours':
            if len(df) > 0 and 'opr_hr' in df.columns and 'avg_price' in df.columns:
                return f"Found {len(df)} cheapest operational hours. Hour {df.iloc[0]['opr_hr']} is cheapest with average price ${df.iloc[0]['avg_price']:.2f}/MWh across {df.iloc[0]['unique_nodes']} nodes."
            else:
                return f"Found {len(df)} operational hours analysis results."
                
        elif analysis_type == 'cheapest_nodes_by_operational_hour':
            if 'mw' in df.columns and len(df) > 0:
                return f"Found {len(df)} cheapest nodes for the specified date and hour. The lowest price was ${df['mw'].min():.2f}/MWh at {df.iloc[0]['node']} on {df.iloc[0]['operational_date']}."
            else:
                return f"Found {len(df)} nodes for the specified operational hour."
                
        elif analysis_type == 'price_percentile':
            summary = f"Found {len(df)} nodes in the requested price percentile."
            if 'mw' in df.columns:
                summary += f" Price range: ${df['mw'].min():.2f} - ${df['mw'].max():.2f}/MWh."
            return summary
            
        elif analysis_type == 'congestion_analysis':
            summary = f"Analyzed {len(df)} hours with congestion data."
            if 'mcc' in df.columns:
                summary += f" Lowest congestion was ${df['mcc'].min():.2f}/MWh."
            else:
                summary += " (Congestion data not available)"
            return summary
            
        elif analysis_type == 'peak_analysis':
            return f"Peak vs off-peak analysis for {len(df)} nodes. Average peak premium varies significantly across nodes."
            
        elif analysis_type == 'price_statistics':
            summary = f"Price statistics for {len(df)} records."
            if 'mw' in df.columns:
                summary += f" Price range: ${df['mw'].min():.2f} - ${df['mw'].max():.2f}/MWh."
            return summary
            
        elif analysis_type == 'hourly_patterns':
            summary = f"Hourly price patterns show data for {len(df)} time periods."
            if 'mw' in df.columns:
                summary += f" Price range: ${df['mw'].min():.2f} to ${df['mw'].max():.2f}/MWh."
            return summary
            
        elif analysis_type == 'price_spikes':
            return f"Detected {len(df)} price spike events across the dataset."
            
        elif analysis_type == 'node_comparison':
            return f"Compared statistics across {len(df)} nodes."
            
        else:
            return f"Analysis completed with {len(df)} results."
    
    def _get_data_context_from_db(self):
        """Get context information about the dataset from database"""
        try:
            # Get basic summary from database
            summary = self.db.get_data_summary()
            
            if not summary or summary.get('total_records', 0) == 0:
                context = {
                    'total_records': 0,
                    'unique_nodes': 0,
                    'date_range': 'No data',
                    'columns': ['operational_date', 'operational_hour', 'node', 'mw'],
                    'sample_nodes': []
                }
            else:
                # Get sample nodes
                try:
                    nodes = self.db.get_unique_nodes()[:5]
                except Exception:
                    nodes = []
                
                # Format date range
                date_range = 'No date data'
                if summary.get('earliest_date') and summary.get('latest_date'):
                    date_range = f"{summary['earliest_date']} to {summary['latest_date']}"
                
                # Use known columns from database schema
                columns = ['operational_date', 'operational_hour', 'node', 'mw', 'mcc', 'mlc', 'pos']
                
                context = {
                    'total_records': summary.get('total_records', 0),
                    'unique_nodes': summary.get('unique_nodes', 0),
                    'date_range': date_range,
                    'columns': columns,
                    'sample_nodes': nodes
                }
            
            # Add available analytics methods from registration system
            try:
                registered_analytics = get_registered_analytics()
                analytics_info = []
                for method_name, metadata in registered_analytics.items():
                    method_info = {
                        'name': method_name,
                        'description': metadata.get('description', ''),
                        'example_questions': metadata.get('example_questions', [])[:2]  # Include first 2 examples
                    }
                    analytics_info.append(method_info)
                
                context['available_analytics'] = analytics_info
                context['analytics_count'] = len(analytics_info)
                
            except Exception as e:
                self.logger.error(f"Error getting analytics methods: {str(e)}")
                context['available_analytics'] = []
                context['analytics_count'] = 0
            
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting data context: {str(e)}")
            return {
                'total_records': 0,
                'unique_nodes': 0,
                'date_range': 'Error retrieving data',
                'columns': ['operational_date', 'operational_hour', 'node', 'mw'],
                'sample_nodes': [],
                'available_analytics': [],
                'analytics_count': 0
            }
    
    def _fallback_intent_analysis(self, question):
        """Fallback analysis when OpenAI is not available"""
        question_lower = question.lower()
        
        # Simple keyword-based intent detection with better defaults
        if any(word in question_lower for word in ['cheapest', 'lowest', 'minimum']):
            # Check if they want hourly patterns vs specific hours vs general stats
            if any(word in question_lower for word in ['hour', 'operational hour']) and any(word in question_lower for word in ['average', 'mean', 'across']):
                return {
                    'analysis_type': 'hourly_patterns',  # This gives average by hour of day
                    'parameters': {},
                    'confidence': 0.8
                }
            elif any(word in question_lower for word in ['average', 'mean', 'across']):
                return {
                    'analysis_type': 'price_statistics', 
                    'parameters': {'aggregate': True},
                    'confidence': 0.7
                }
            else:
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
