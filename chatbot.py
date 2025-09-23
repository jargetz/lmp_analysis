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
                "comparison_type": "type of comparison requested",
                "scope": "market|node|list (REQUIRED - see scope rules below)",
                "include_zero": boolean (REQUIRED - default false, true only if explicitly requested)
            }},
            "confidence": number between 0 and 1
        }}
        
        SCOPE SETTING RULES (ALWAYS SET scope field):
        - scope="market": For questions about "all nodes", market-wide analysis, or when no specific nodes mentioned
        - scope="node": For questions about a specific single node (when nodes array has exactly 1 item)
        - scope="list": For questions asking for "top/cheapest/highest/best/worst X nodes" or ranked comparisons
        
        INCLUDE_ZERO RULES (ALWAYS SET include_zero field):
        - include_zero=false: Default behavior (exclude zero/null prices)
        - include_zero=true: Only when user explicitly asks to "include zero prices", "show all prices including zero", or similar explicit requests
        
        SCOPE EXAMPLES:
        - "What's the spread of peak vs off-peak for all nodes?" → scope="market", include_zero=false
        - "Show node CSADIAB_7_N001 peak vs off-peak" → scope="node", include_zero=false  
        - "Top 10 nodes by peak premium" → scope="list", include_zero=false
        - "All nodes including zero prices" → scope="market", include_zero=true
        - "Which nodes have the lowest congestion?" → scope="list", include_zero=false
        - "Average price across all nodes" → scope="market", include_zero=false
        
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
                result = json.loads(content)
                # Validate and add fallback logic for scope and include_zero fields
                result = self._validate_and_enhance_intent(result, question)
                return result
            else:
                return self._fallback_intent_analysis(question)
            
        except Exception as e:
            # Fallback to basic analysis if OpenAI fails
            return self._fallback_intent_analysis(question)
    
    def _validate_and_enhance_intent(self, result, question):
        """Validate and enhance the intent result with scope and include_zero fallbacks"""
        parameters = result.get('parameters', {})
        
        # Ensure scope is set with deterministic fallback logic
        if 'scope' not in parameters or parameters['scope'] not in ['market', 'node', 'list']:
            scope = self._determine_scope_fallback(question, parameters)
            parameters['scope'] = scope
        
        # Ensure include_zero is set with proper fallback
        if 'include_zero' not in parameters or not isinstance(parameters['include_zero'], bool):
            include_zero = self._determine_include_zero_fallback(question)
            parameters['include_zero'] = include_zero
        
        result['parameters'] = parameters
        return result
    
    def _determine_scope_fallback(self, question, parameters):
        """Deterministic fallback logic for scope field"""
        question_lower = question.lower()
        nodes = parameters.get('nodes', [])
        
        # If specific node mentioned, scope=node
        if nodes and len(nodes) == 1:
            return 'node'
        
        # If asking for ranked/top/bottom results, scope=list
        ranking_keywords = ['top', 'bottom', 'best', 'worst', 'cheapest', 'most expensive', 
                          'highest', 'lowest', 'first', 'last', 'ranking', 'ranked']
        if any(keyword in question_lower for keyword in ranking_keywords):
            # Also check for number indicators like "top 10", "5 cheapest", etc.
            if any(word in question_lower for word in ['nodes', 'node']) and \
               (parameters.get('n_nodes') or any(char.isdigit() for char in question)):
                return 'list'
        
        # Default to market scope for general questions
        return 'market'
    
    def _determine_include_zero_fallback(self, question):
        """Deterministic fallback logic for include_zero field"""
        question_lower = question.lower()
        
        # Only set to True if explicitly requested
        zero_keywords = ['include zero', 'including zero', 'with zero', 'zero prices', 
                        'all prices', 'show zero', 'include all']
        
        return any(keyword in question_lower for keyword in zero_keywords)
    
    def _execute_analysis(self, analysis_result):
        """Execute the determined analysis using database queries"""
        analysis_type = analysis_result.get('analysis_type', 'general_query')
        params = analysis_result.get('parameters', {})
        
        self.logger.info(f"Executing analysis type: {analysis_type} with params: {params}")
        
        # Extract scope and include_zero parameters with defaults
        scope = params.get('scope', 'market')
        include_zero = params.get('include_zero', False)
        
        # Map include_zero to exclude_zero (include_zero=False → exclude_zero=True)
        exclude_zero = not include_zero
        
        # Map scope to market_summary for compatible methods
        market_summary = scope == 'market'
        
        self.logger.info(f"Intent mapping: scope={scope}, include_zero={include_zero} → exclude_zero={exclude_zero}, market_summary={market_summary}")
        
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
                        'exclude_zero': exclude_zero,  # Now derived from intent's include_zero
                        'during_cheap_hours': False,
                        'node': None,
                        'aggregate_nodes': None
                    }
                    
                    # Add market_summary parameter for compatible methods
                    methods_supporting_market_summary = [
                        'get_peak_vs_offpeak_analysis', 
                        'get_price_statistics', 
                        'get_hourly_averages'
                    ]
                    if method_name in methods_supporting_market_summary:
                        param_mapping['market_summary'] = market_summary
                    
                    # Handle special parameter logic for specific methods
                    if method_name == 'get_cheapest_hours':
                        nodes = params.get('nodes', [])
                        if len(nodes) == 1:
                            method_args['node'] = nodes[0]
                        elif len(nodes) > 1:
                            method_args['aggregate_nodes'] = nodes
                        method_args['n_hours'] = params.get('n_hours', 10)
                        method_args['exclude_zero'] = exclude_zero
                        
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
                            'exclude_zero': exclude_zero
                        }
                        
                    elif method_name == 'get_cheapest_operational_hours':
                        method_args = {
                            'n_hours': params.get('n_hours', 5),
                            'start_date': params.get('start_date'),
                            'end_date': params.get('end_date'),
                            'exclude_zero': exclude_zero
                        }
                        
                    elif method_name == 'get_lowest_congestion_hours':
                        method_args = {
                            'n_hours': params.get('n_hours', 20),
                            'during_cheap_hours': params.get('during_cheap_hours', False),
                            'start_date': params.get('start_date'),
                            'end_date': params.get('end_date'),
                            'exclude_zero': exclude_zero
                        }
                        
                    elif method_name == 'get_node_hourly_prices':
                        # Handle node-specific hourly price queries
                        nodes = params.get('nodes', [])
                        node = None
                        
                        # Extract node from different possible parameter formats
                        if nodes and len(nodes) > 0:
                            node = nodes[0]  # Take the first node if multiple specified
                        
                        if not node:
                            # This method requires a node, return error if none specified
                            return pd.DataFrame(), "Node-specific hourly price analysis requires a specific node to be specified"
                        
                        method_args = {
                            'node': node,
                            'start_date': params.get('start_date'),
                            'end_date': params.get('end_date'),
                            'aggregation_method': params.get('aggregation_method', 'avg'),
                            'exclude_zero': exclude_zero
                        }
                        
                    elif method_name in methods_supporting_market_summary:
                        # For methods that support market_summary parameter
                        for param_name in expected_params:
                            if param_name in param_mapping and param_mapping[param_name] is not None:
                                method_args[param_name] = param_mapping[param_name]
                        
                        # Ensure market_summary and exclude_zero are always included
                        method_args['market_summary'] = market_summary
                        method_args['exclude_zero'] = exclude_zero
                        
                    else:
                        # For other methods, map only the parameters they expect
                        for param_name in expected_params:
                            if param_name in param_mapping and param_mapping[param_name] is not None:
                                method_args[param_name] = param_mapping[param_name]
                        
                        # Ensure exclude_zero is always included for all methods
                        method_args['exclude_zero'] = exclude_zero
                    
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
                    # Apply scope and include_zero logic to fallback method too
                    fallback_args = {
                        'market_summary': market_summary,
                        'exclude_zero': exclude_zero
                    }
                    return self.analytics.get_price_statistics(**fallback_args)
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
        
        # Get scope from intent analysis
        params = intent_analysis.get('parameters', {})
        scope = params.get('scope', 'market')
        
        # Generate interpretation display
        interpretation = self._format_interpretation(intent_analysis)
        
        # Convert DataFrame to readable format
        if isinstance(result_df, pd.DataFrame):
            # Generate summary
            summary = self._generate_summary(result_df, intent_analysis.get('analysis_type', 'general'), intent_analysis)
            
            # Format response based on scope
            if scope == 'market':
                # Market scope: Show concise summaries without detailed tables
                response = f"{interpretation}\n\n{summary}"
                
                # Only show table for market scope if data is very limited and meaningful
                if len(result_df) <= 3 and any(col in result_df.columns for col in ['peak_avg', 'offpeak_avg', 'avg_price', 'min_price', 'max_price']):
                    display_df = result_df
                    table_str = display_df.to_string(index=False, float_format='%.2f')
                    response += f"\n\nKey Statistics:\n```\n{table_str}\n```"
                    
            elif scope == 'node':
                # Node scope: Show focused node details with relevant data
                display_df = result_df.head(10)  # Smaller limit for focused view
                table_str = display_df.to_string(index=False, float_format='%.2f')
                response = f"{interpretation}\n\n{summary}\n\nNode Details:\n```\n{table_str}\n```"
                
                if len(result_df) > 10:
                    response += f"\n\n(Showing first 10 results out of {len(result_df)} total results)"
                    
            elif scope == 'list':
                # List scope: Show clean ranked tables
                display_df = result_df.head(15)  # Show more for rankings
                table_str = display_df.to_string(index=False, float_format='%.2f')
                response = f"{interpretation}\n\n{summary}\n\nRanking Results:\n```\n{table_str}\n```"
                
                if len(result_df) > 15:
                    response += f"\n\n(Showing top 15 results out of {len(result_df)} total results)"
            
            else:
                # Fallback: Default table display
                display_df = result_df.head(20)
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
        
        # Get scope and include_zero information
        scope = parameters.get('scope', 'market')
        include_zero = parameters.get('include_zero', False)
        
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
        
        # Format scope information
        scope_descriptions = {
            'market': 'Market-wide analysis',
            'node': 'Node-specific analysis', 
            'list': 'Comparative ranking analysis'
        }
        scope_str = scope_descriptions.get(scope, scope)
        
        # Format parameters in a user-friendly way
        param_display = []
        for key, value in parameters.items():
            if value is not None and value != [] and value != '' and key not in ['scope', 'include_zero']:
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
        
        # Add include_zero information if relevant
        zero_filter_str = ""
        if include_zero:
            zero_filter_str = " • Including zero/null prices"
        
        interpretation = f"""🤖 **How I interpreted your question:**
**Analysis Type:** {friendly_name}
**Scope:** {scope_str}
**Parameters:** {params_str}{zero_filter_str}
**Confidence:** {confidence_str}

---"""
        
        return interpretation
    
    def _generate_summary(self, df, analysis_type, intent_analysis=None):
        """Generate a summary of the analysis results based on scope"""
        if df.empty:
            return "No results found for your query."
        
        # Get scope from intent analysis
        scope = 'market'  # default
        if intent_analysis:
            params = intent_analysis.get('parameters', {})
            scope = params.get('scope', 'market')
        
        # Generate scope-aware summaries
        if scope == 'market':
            return self._generate_market_summary(df, analysis_type)
        elif scope == 'node':
            return self._generate_node_summary(df, analysis_type)
        elif scope == 'list':
            return self._generate_list_summary(df, analysis_type)
        else:
            # Fallback to original logic
            return self._generate_fallback_summary(df, analysis_type)
    
    def _generate_market_summary(self, df, analysis_type):
        """Generate concise market-wide summaries"""
        if df.empty:
            return "No market data found."
        
        # Market summary formatting for specific analysis types
        if analysis_type == 'get_peak_vs_offpeak_analysis':
            if 'peak' in df.columns and 'off_peak' in df.columns and len(df) > 0:
                peak_price = df['peak'].iloc[0] if not df['peak'].isna().iloc[0] else 0
                offpeak_price = df['off_peak'].iloc[0] if not df['off_peak'].isna().iloc[0] else 0
                if peak_price > 0 and offpeak_price > 0:
                    premium = peak_price - offpeak_price
                    premium_pct = (premium / offpeak_price) * 100 if offpeak_price > 0 else 0
                    return f"Market Peak: ${peak_price:.2f}/MWh, Off-Peak: ${offpeak_price:.2f}/MWh, Premium: ${premium:.2f} ({premium_pct:.1f}%)"
                else:
                    return f"Market Peak: ${peak_price:.2f}/MWh, Off-Peak: ${offpeak_price:.2f}/MWh"
            else:
                return "Market peak vs off-peak analysis completed."
                
        elif analysis_type == 'get_price_statistics':
            if 'avg_price' in df.columns and 'min_price' in df.columns and 'max_price' in df.columns and len(df) > 0:
                avg_price = df['avg_price'].iloc[0]
                min_price = df['min_price'].iloc[0] 
                max_price = df['max_price'].iloc[0]
                return f"Market Average: ${avg_price:.2f}/MWh, Range: ${min_price:.2f}-${max_price:.2f}/MWh"
            elif 'mw' in df.columns:
                avg_price = df['mw'].mean()
                min_price = df['mw'].min()
                max_price = df['mw'].max()
                return f"Market Average: ${avg_price:.2f}/MWh, Range: ${min_price:.2f}-${max_price:.2f}/MWh"
            else:
                return "Market price statistics analysis completed."
                
        elif analysis_type == 'get_hourly_averages':
            if 'avg_price' in df.columns and len(df) > 0:
                min_hourly = df['avg_price'].min()
                max_hourly = df['avg_price'].max()
                return f"24-hour market profile shows prices ranging from ${min_hourly:.2f}-${max_hourly:.2f}/MWh"
            elif 'mw' in df.columns:
                min_price = df['mw'].min()
                max_price = df['mw'].max()
                return f"24-hour market profile shows prices ranging from ${min_price:.2f}-${max_price:.2f}/MWh"
            else:
                return "24-hour market profile analysis completed."
                
        else:
            # Generic market summary for other analysis types
            if 'mw' in df.columns and len(df) > 0:
                avg_price = df['mw'].mean()
                min_price = df['mw'].min()
                max_price = df['mw'].max()
                return f"Market analysis shows average price ${avg_price:.2f}/MWh with range ${min_price:.2f}-${max_price:.2f}/MWh across {len(df)} data points."
            else:
                return f"Market-wide analysis completed with {len(df)} results."
    
    def _generate_node_summary(self, df, analysis_type):
        """Generate individual node details"""
        if df.empty:
            return "No node data found."
        
        # Node-specific formatting
        if analysis_type == 'get_node_hourly_prices':
            if len(df) > 0 and 'node' in df.columns and 'price' in df.columns:
                node_name = df.iloc[0]['node']
                min_price = df['price'].min()
                max_price = df['price'].max()
                avg_price = df['price'].mean()
                hours_with_data = len(df)
                return f"Node {node_name}: {hours_with_data} hours analyzed, average ${avg_price:.2f}/MWh, range ${min_price:.2f}-${max_price:.2f}/MWh."
            else:
                return f"Node-specific analysis completed with {len(df)} results."
        
        elif 'node' in df.columns and 'mw' in df.columns and len(df) > 0:
            # Generic node analysis
            node_name = df.iloc[0]['node']
            avg_price = df['mw'].mean()
            min_price = df['mw'].min() 
            max_price = df['mw'].max()
            return f"Node {node_name}: Average ${avg_price:.2f}/MWh, range ${min_price:.2f}-${max_price:.2f}/MWh across {len(df)} time periods."
        
        else:
            return f"Node-specific analysis completed with {len(df)} results."
    
    def _generate_list_summary(self, df, analysis_type):
        """Generate ranking/comparison format summaries"""
        if df.empty:
            return "No ranking data found."
        
        # List/ranking formatting
        if analysis_type == 'get_cheapest_nodes_by_hour':
            if 'mw' in df.columns and len(df) > 0:
                cheapest_price = df['mw'].min()
                cheapest_node = df.iloc[0]['node']
                return f"Ranking of {len(df)} nodes by price. Cheapest: {cheapest_node} at ${cheapest_price:.2f}/MWh."
            else:
                return f"Ranking of {len(df)} nodes completed."
                
        elif analysis_type == 'cheapest_hours':
            if 'mw' in df.columns and len(df) > 0:
                cheapest_price = df['mw'].min()
                return f"Top {len(df)} cheapest hours identified. Lowest price: ${cheapest_price:.2f}/MWh."
            else:
                return f"Top {len(df)} cheapest hours analysis completed."
                
        elif analysis_type == 'cheapest_operational_hours':
            if 'avg_price' in df.columns and 'opr_hr' in df.columns and len(df) > 0:
                best_hour = df.iloc[0]['opr_hr']
                best_price = df.iloc[0]['avg_price']
                return f"Top {len(df)} cheapest operational hours ranked. Best: Hour {best_hour} at ${best_price:.2f}/MWh average."
            else:
                return f"Top {len(df)} operational hours ranked."
        
        else:
            # Generic ranking summary
            if 'mw' in df.columns and len(df) > 0:
                best_price = df['mw'].min()
                price_range = df['mw'].max() - df['mw'].min()
                return f"Ranked {len(df)} results. Best price: ${best_price:.2f}/MWh, range: ${price_range:.2f}/MWh."
            else:
                return f"Ranking analysis completed with {len(df)} results."
    
    def _generate_fallback_summary(self, df, analysis_type):
        """Fallback to original summary logic"""
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
            
        elif analysis_type == 'get_node_hourly_prices':
            if len(df) > 0 and 'node' in df.columns and 'price' in df.columns:
                node_name = df.iloc[0]['node']
                min_price = df['price'].min()
                max_price = df['price'].max()
                hours_with_data = len(df)
                return f"Node-specific hourly prices for {node_name}: {hours_with_data} operational hours with price range ${min_price:.2f} - ${max_price:.2f}/MWh."
            else:
                return f"Node-specific hourly analysis completed with {len(df)} results."
            
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
                    'parameters': {'scope': 'market', 'include_zero': False},
                    'confidence': 0.8
                }
            elif any(word in question_lower for word in ['average', 'mean', 'across']):
                return {
                    'analysis_type': 'price_statistics', 
                    'parameters': {'aggregate': True, 'scope': 'market', 'include_zero': False},
                    'confidence': 0.7
                }
            else:
                return {
                    'analysis_type': 'cheapest_hours',
                    'parameters': {'n_hours': 10, 'scope': 'list', 'include_zero': False},
                    'confidence': 0.7
                }
        elif any(word in question_lower for word in ['percentile', '%']):
            return {
                'analysis_type': 'price_percentile',
                'parameters': {'percentile': 10, 'scope': 'list', 'include_zero': False},
                'confidence': 0.7
            }
        elif any(word in question_lower for word in ['congestion', 'mcc']):
            return {
                'analysis_type': 'congestion_analysis',
                'parameters': {'n_hours': 20, 'scope': 'list', 'include_zero': False},
                'confidence': 0.7
            }
        elif any(word in question_lower for word in ['peak', 'hour']):
            return {
                'analysis_type': 'hourly_patterns',
                'parameters': {'scope': 'market', 'include_zero': False},
                'confidence': 0.6
            }
        else:
            return {
                'analysis_type': 'general_query',
                'parameters': {'scope': 'market', 'include_zero': False},
                'confidence': 0.5
            }
