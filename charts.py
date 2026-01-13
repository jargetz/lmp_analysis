"""
Modular chart functions for LMP Dashboard.

Each function accepts data and returns a Plotly figure.
Easy to customize: just modify the returned fig before displaying.

Usage in Streamlit:
    from charts import create_hourly_price_chart
    fig = create_hourly_price_chart(hourly_data)
    st.plotly_chart(fig, use_container_width=True)
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, Dict, Any


def create_hourly_price_chart(
    df: pd.DataFrame,
    x_col: str = 'hour',
    y_col: str = 'mw',
    title: str = 'Average Price by Hour of Day',
    color: str = '#1f77b4'
) -> go.Figure:
    """
    Create a line chart showing hourly price patterns.
    
    Args:
        df: DataFrame with hour and price columns
        x_col: Column name for hour (0-23)
        y_col: Column name for price
        title: Chart title
        color: Line color
        
    Returns:
        Plotly Figure object
    """
    if df.empty:
        return create_empty_chart("No hourly data available")
    
    fig = px.line(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: 'Hour of Day', y_col: 'Price ($/MWh)'}
    )
    
    fig.update_traces(line_color=color, line_width=2)
    fig.update_layout(
        hovermode='x unified',
        xaxis=dict(tickmode='linear', dtick=2),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_zone_hourly_chart(
    zone_data: dict,
    title: str = 'Hourly Price by Zone'
) -> go.Figure:
    """
    Create a multi-line chart showing hourly prices for all zones.
    
    Args:
        zone_data: Dict with zone names as keys, each containing list of 
                   {'hour': int, 'avg_price': float} dicts
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    
    colors = {
        'NP15': '#1f77b4',    # Blue
        'SP15': '#ff7f0e',    # Orange
        'ZP26': '#2ca02c',    # Green
        'Overall': '#7f7f7f'  # Gray
    }
    
    zone_order = ['NP15', 'SP15', 'ZP26', 'Overall']
    
    for zone in zone_order:
        data = zone_data.get(zone, [])
        if data:
            hours = [d['hour'] for d in data]
            prices = [d['avg_price'] for d in data]
            
            fig.add_trace(go.Scatter(
                x=hours,
                y=prices,
                mode='lines',
                name=zone,
                line=dict(color=colors.get(zone, '#000000'), width=2),
                hovertemplate=f'{zone}: $%{{y:.2f}}/MWh<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Hour of Day',
        yaxis_title='Price ($/MWh)',
        hovermode='x unified',
        xaxis=dict(tickmode='linear', dtick=2, range=[-0.5, 23.5]),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def create_bx_trend_chart(
    df: pd.DataFrame,
    date_col: str = 'opr_dt',
    price_col: str = 'avg_price',
    bx_type: int = 8,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a time series chart showing BX price trends over time.
    
    Args:
        df: DataFrame with date and price columns
        date_col: Column name for date
        price_col: Column name for average price
        bx_type: BX value for title (4-10)
        title: Custom title (default: "B{X} Average Price Trend")
        
    Returns:
        Plotly Figure object
    """
    if df.empty:
        return create_empty_chart(f"No B{bx_type} trend data available")
    
    title = title or f'B{bx_type} Average Price Trend'
    
    fig = px.line(
        df,
        x=date_col,
        y=price_col,
        title=title,
        labels={date_col: 'Date', price_col: 'Avg Price ($/MWh)'}
    )
    
    fig.update_traces(line_color='#2ca02c', line_width=2)
    fig.update_layout(
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_zone_comparison_bar(
    zone_data: Dict[str, float],
    title: str = 'Average Price by Zone',
    color_sequence: Optional[list] = None
) -> go.Figure:
    """
    Create a bar chart comparing prices across zones.
    
    Args:
        zone_data: Dict mapping zone names to average prices
        title: Chart title
        color_sequence: List of colors for bars
        
    Returns:
        Plotly Figure object
    """
    if not zone_data:
        return create_empty_chart("No zone data available")
    
    color_sequence = color_sequence or ['#1f77b4', '#ff7f0e', '#2ca02c']
    
    df = pd.DataFrame([
        {'zone': zone, 'avg_price': price}
        for zone, price in zone_data.items()
    ])
    
    fig = px.bar(
        df,
        x='zone',
        y='avg_price',
        title=title,
        labels={'zone': 'Zone', 'avg_price': 'Avg Price ($/MWh)'},
        color='zone',
        color_discrete_sequence=color_sequence
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_price_distribution_histogram(
    df: pd.DataFrame,
    price_col: str = 'mw',
    bins: int = 30,
    title: str = 'Price Distribution'
) -> go.Figure:
    """
    Create a histogram showing price distribution.
    
    Args:
        df: DataFrame with price data
        price_col: Column name for prices
        bins: Number of histogram bins
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    if df.empty:
        return create_empty_chart("No price data available")
    
    fig = px.histogram(
        df,
        x=price_col,
        nbins=bins,
        title=title,
        labels={price_col: 'Price ($/MWh)'}
    )
    
    fig.update_traces(marker_color='#9467bd')
    fig.update_layout(
        bargap=0.1,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_node_price_heatmap(
    df: pd.DataFrame,
    node_col: str = 'node',
    hour_col: str = 'opr_hr',
    price_col: str = 'mw',
    title: str = 'Price by Node and Hour'
) -> go.Figure:
    """
    Create a heatmap showing prices by node and hour.
    
    Args:
        df: DataFrame with node, hour, and price data
        node_col: Column name for node
        hour_col: Column name for hour
        price_col: Column name for price
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    if df.empty:
        return create_empty_chart("No data for heatmap")
    
    pivot = df.pivot_table(
        index=node_col,
        columns=hour_col,
        values=price_col,
        aggfunc='mean'
    )
    
    fig = px.imshow(
        pivot,
        title=title,
        labels=dict(x='Hour', y='Node', color='Price ($/MWh)'),
        color_continuous_scale='RdYlGn_r',
        aspect='auto'
    )
    
    fig.update_layout(
        margin=dict(l=100, r=40, t=50, b=40)
    )
    
    return fig


def create_top_nodes_bar(
    df: pd.DataFrame,
    node_col: str = 'node',
    price_col: str = 'mean',
    n_nodes: int = 10,
    ascending: bool = True,
    title: Optional[str] = None
) -> go.Figure:
    """
    Create a horizontal bar chart of top/bottom nodes by price.
    
    Args:
        df: DataFrame with node and price columns
        node_col: Column name for node
        price_col: Column name for price metric
        n_nodes: Number of nodes to show
        ascending: If True, show cheapest; if False, show most expensive
        title: Chart title (auto-generated if None)
        
    Returns:
        Plotly Figure object
    """
    if df.empty:
        return create_empty_chart("No node data available")
    
    sorted_df = df.sort_values(price_col, ascending=ascending).head(n_nodes)
    
    if title is None:
        title = f'{"Cheapest" if ascending else "Most Expensive"} {n_nodes} Nodes'
    
    color = '#2ca02c' if ascending else '#d62728'
    
    fig = px.bar(
        sorted_df,
        y=node_col,
        x=price_col,
        orientation='h',
        title=title,
        labels={node_col: 'Node', price_col: 'Avg Price ($/MWh)'}
    )
    
    fig.update_traces(marker_color=color)
    fig.update_layout(
        yaxis=dict(autorange='reversed'),
        margin=dict(l=120, r=40, t=50, b=40)
    )
    
    return fig


def create_summary_metrics(
    avg_price: float,
    min_price: float,
    max_price: float,
    node_count: int
) -> Dict[str, Any]:
    """
    Format summary metrics for display.
    
    This is not a chart but a helper for consistent metric formatting.
    
    Args:
        avg_price: Average price
        min_price: Minimum price
        max_price: Maximum price
        node_count: Number of nodes
        
    Returns:
        Dict with formatted values for Streamlit metrics
    """
    return {
        'avg': f'${avg_price:.2f}/MWh',
        'min': f'${min_price:.2f}/MWh',
        'max': f'${max_price:.2f}/MWh',
        'count': f'{node_count:,}'
    }


def create_empty_chart(message: str = "No data available") -> go.Figure:
    """
    Create an empty placeholder chart with a message.
    
    Args:
        message: Text to display in the empty chart
        
    Returns:
        Plotly Figure with centered message
    """
    fig = go.Figure()
    
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=16, color='gray')
    )
    
    fig.update_layout(
        xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def apply_dark_theme(fig: go.Figure) -> go.Figure:
    """
    Apply a dark theme to any chart.
    
    Usage:
        fig = create_hourly_price_chart(data)
        fig = apply_dark_theme(fig)
        st.plotly_chart(fig)
        
    Args:
        fig: Plotly Figure to modify
        
    Returns:
        Modified Figure with dark theme
    """
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig


def create_node_hourly_chart(
    hourly_data: list,
    title: str = 'Hourly Price (Selected Nodes)'
) -> go.Figure:
    """
    Create a line chart showing hourly prices for selected nodes.
    
    Args:
        hourly_data: List of {'hour': int, 'avg_price': float} dicts
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    if not hourly_data:
        return create_empty_chart("No hourly data available")
    
    hours = [d['hour'] for d in hourly_data]
    prices = [d['avg_price'] for d in hourly_data]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours,
        y=prices,
        mode='lines',
        name='Average',
        line=dict(color='#1f77b4', width=2),
        hovertemplate='Hour %{x}: $%{y:.2f}/MWh<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Hour of Day',
        yaxis_title='Price ($/MWh)',
        hovermode='x unified',
        xaxis=dict(tickmode='linear', dtick=2, range=[-0.5, 23.5]),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_node_hourly_lines_chart(
    per_node_data: dict,
    title: str = 'Hourly Price by Node'
) -> go.Figure:
    """
    Create a multi-line chart showing hourly prices for each node individually.
    
    Args:
        per_node_data: Dict with node names as keys, each containing list of 
                       {'hour': int, 'avg_price': float} dicts
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    if not per_node_data:
        return create_empty_chart("No per-node data available")
    
    fig = go.Figure()
    
    colors = px.colors.qualitative.Set2 + px.colors.qualitative.Set1
    
    for i, (node, data) in enumerate(sorted(per_node_data.items())):
        if data:
            hours = [d['hour'] for d in data]
            prices = [d['avg_price'] for d in data]
            color = colors[i % len(colors)]
            
            fig.add_trace(go.Scatter(
                x=hours,
                y=prices,
                mode='lines',
                name=node[:20] + '...' if len(node) > 20 else node,
                line=dict(color=color, width=1.5),
                hovertemplate=f'{node}<br>Hour %{{x}}: $%{{y:.2f}}/MWh<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Hour of Day',
        yaxis_title='Price ($/MWh)',
        hovermode='x unified',
        xaxis=dict(tickmode='linear', dtick=2, range=[-0.5, 23.5]),
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        showlegend=len(per_node_data) <= 10
    )
    
    return fig


def create_node_month_hour_heatmap(
    heatmap_data: list,
    title: str = 'Price Heatmap (Selected Nodes)'
) -> go.Figure:
    """
    Create a month x hour heatmap for selected node data.
    Hours on x-axis, months on y-axis for consistency with other charts.
    
    Args:
        heatmap_data: List of {'month': int, 'hour': int, 'avg_price': float} dicts
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    if not heatmap_data:
        return create_empty_chart("No heatmap data available")
    
    df = pd.DataFrame(heatmap_data)
    
    month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                   7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
    df['month_name'] = df['month'].map(month_names)
    
    pivot = df.pivot_table(values='avg_price', index='month', columns='hour', aggfunc='mean')
    
    pivot = pivot.reindex(index=sorted(pivot.index))
    pivot = pivot.reindex(columns=sorted(pivot.columns))
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=[month_names.get(m, str(m)) for m in pivot.index],
        colorscale='RdYlGn_r',
        hovertemplate='Hour: %{x}<br>Month: %{y}<br>Avg Price: $%{z:.2f}/MWh<extra></extra>',
        colorbar=dict(title='$/MWh')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Hour of Day',
        yaxis_title='Month',
        xaxis=dict(tickmode='linear', dtick=2),
        margin=dict(l=40, r=40, t=50, b=40)
    )
    
    return fig


def create_zone_bx_trend_chart(
    zone_data: dict,
    bx_type: int,
    title: str = None
) -> go.Figure:
    """
    Create a multi-line chart showing BX price trend for all zones.
    
    Args:
        zone_data: Dict with zone names as keys, each containing list of
                   {'date': date, 'avg_price': float} dicts
        bx_type: BX value (4-10) for title
        title: Custom title (default: "B{X} Price Trend by Zone")
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    title = title or f'B{bx_type} Price Trend by Zone'
    
    colors = {
        'NP15': '#1f77b4',
        'SP15': '#ff7f0e',
        'ZP26': '#2ca02c',
        'Overall': '#7f7f7f'
    }
    
    zone_order = ['NP15', 'SP15', 'ZP26', 'Overall']
    
    for zone in zone_order:
        data = zone_data.get(zone, [])
        if data:
            dates = [d['date'] for d in data]
            prices = [d['avg_price'] for d in data]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines+markers',
                name=zone,
                line=dict(color=colors.get(zone, '#000000'), width=2),
                marker=dict(size=6),
                hovertemplate=f'{zone}: $%{{y:.2f}}/MWh<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Avg Price ($/MWh)',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def create_node_bx_trend_chart(
    node_data: dict,
    bx_type: int,
    title: str = None
) -> go.Figure:
    """
    Create a multi-line chart showing BX price trend for each node.
    
    Args:
        node_data: Dict with node names as keys, each containing list of
                   {'date': date, 'avg_price': float} dicts
        bx_type: BX value (4-10) for title
        title: Custom title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    title = title or f'B{bx_type} Price Trend by Node'
    
    for i, (node, data) in enumerate(node_data.items()):
        if data:
            dates = [d['date'] for d in data]
            prices = [d['avg_price'] for d in data]
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=node,
                line=dict(width=1.5),
                hovertemplate=f'{node}: $%{{y:.2f}}/MWh<extra></extra>'
            ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Avg Price ($/MWh)',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=len(node_data) <= 10,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    
    return fig


def create_month_hour_heatmap(
    data: list,
    title: str = 'Averages - Day Ahead LMP',
    zone: str = None
) -> go.Figure:
    """
    Create a heatmap table showing average prices by month (rows) and hour (columns).
    
    Args:
        data: List of dicts with 'month', 'hour', 'avg_price'
        title: Chart title
        zone: Optional zone name for title
        
    Returns:
        Plotly Figure object
    """
    if not data:
        return create_empty_chart("No data for heatmap")
    
    df = pd.DataFrame(data)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    pivot = df.pivot(index='month', columns='hour', values='avg_price')
    pivot = pivot.reindex(range(1, 13))
    pivot.columns = [int(h) + 1 for h in pivot.columns]
    
    z_values = pivot.values
    x_labels = [str(h) for h in range(1, 25)]
    y_labels = month_names[:len(pivot)]
    
    text_values = [[f'{val:.2f}' if pd.notna(val) else '' for val in row] for row in z_values]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_values,
        texttemplate='%{text}',
        textfont=dict(size=10),
        colorscale=[
            [0.0, '#3366cc'],
            [0.25, '#66aaff'],
            [0.5, '#ffff99'],
            [0.75, '#ff9966'],
            [1.0, '#cc3300']
        ],
        hovertemplate='%{y} Hour %{x}: $%{z:.2f}/MWh<extra></extra>',
        showscale=True,
        colorbar=dict(title='$/MWh', tickformat='.0f')
    ))
    
    display_title = f"{title} - {zone}" if zone else title
    
    fig.update_layout(
        title=display_title,
        xaxis_title='Hour Ending',
        yaxis_title='',
        xaxis=dict(
            tickmode='linear',
            dtick=1,
            side='bottom'
        ),
        yaxis=dict(
            autorange='reversed',
            tickmode='array',
            ticktext=y_labels,
            tickvals=list(range(len(y_labels)))
        ),
        margin=dict(l=60, r=40, t=50, b=60),
        height=400
    )
    
    return fig


def create_node_box_plot(
    stats_data: list,
    title: str = 'Price Distribution by Node'
) -> go.Figure:
    """
    Create a box plot showing price distribution for each node.
    
    Args:
        stats_data: List of dicts with node, min, q1, median, q3, max, mean
        title: Chart title
        
    Returns:
        Plotly Figure object
    """
    if not stats_data:
        return create_empty_chart("No data for box plot")
    
    fig = go.Figure()
    
    nodes = [stat['node'] for stat in stats_data]
    
    for stat in stats_data:
        fig.add_trace(go.Box(
            name=stat['node'],
            y=[stat['min'], stat['q1'], stat['median'], stat['q3'], stat['max']],
            boxpoints=False,
            hoverinfo='name+y'
        ))
    
    # Add mean markers as separate scatter trace
    fig.add_trace(go.Scatter(
        x=nodes,
        y=[stat['mean'] for stat in stats_data],
        mode='markers',
        name='Mean',
        marker=dict(symbol='diamond', size=10, color='red'),
        hovertemplate='Mean: $%{y:.2f}/MWh<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        yaxis_title='Price ($/MWh)',
        showlegend=True,
        margin=dict(l=40, r=40, t=50, b=100),
        xaxis=dict(tickangle=45)
    )
    
    return fig
