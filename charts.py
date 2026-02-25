"""
Modular chart functions for LMP Dashboard.

Each function accepts data and returns a Plotly figure.
Easy to customize: just modify the returned fig before displaying.

Usage in Streamlit:
    from charts import create_hourly_price_chart
    fig = create_hourly_price_chart(hourly_data)
    st.plotly_chart(fig, use_container_width=True)
"""

import numpy as np
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
):
    """
    Create a month x hour heatmap for selected node data.
    
    Returns:
        Tuple of (Plotly Figure, clipping_info dict or None)
    """
    if not heatmap_data:
        return create_empty_chart("No heatmap data available"), None
    
    df = pd.DataFrame(heatmap_data)
    
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    pivot = df.pivot_table(values='avg_price', index='month', columns='hour', aggfunc='mean')
    pivot = pivot.reindex(index=range(1, 13))
    pivot.columns = [int(h) for h in pivot.columns]
    
    z_values = pivot.values
    x_labels = [str(h) for h in range(1, 25)]
    y_labels = month_names[:len(pivot)]
    
    import numpy as np
    flat_vals = [v for row in z_values for v in row if pd.notna(v)]
    clipping_info = None
    if flat_vals:
        actual_min = float(min(flat_vals))
        actual_max = float(max(flat_vals))
        zmin = float(np.percentile(flat_vals, 2))
        zmax = float(np.percentile(flat_vals, 98))
        if zmin == zmax:
            zmin, zmax = actual_min, actual_max
        if actual_min < zmin or actual_max > zmax:
            clipping_info = {
                'zmin': round(zmin, 2), 'zmax': round(zmax, 2),
                'actual_min': round(actual_min, 2), 'actual_max': round(actual_max, 2),
                'clipped_below': actual_min < zmin, 'clipped_above': actual_max > zmax
            }
    else:
        zmin, zmax = None, None
    
    text_values = [[f'{val:.2f}' if pd.notna(val) else '' for val in row] for row in z_values]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=y_labels,
        text=text_values,
        texttemplate='%{text}',
        textfont=dict(size=7, color='black'),
        colorscale=[
            [0.0, '#3366cc'],
            [0.25, '#66aaff'],
            [0.5, '#ffff99'],
            [0.75, '#ff9966'],
            [1.0, '#cc3300']
        ],
        zmin=zmin,
        zmax=zmax,
        hovertemplate='%{y} Hour %{x}: $%{z:.2f}/MWh<extra></extra>',
        showscale=True,
        colorbar=dict(title='$/MWh', tickformat='.0f')
    ))
    
    fig.update_layout(
        title=title,
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
    
    return fig, clipping_info


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
    node_data,
    bx_type: int,
    title: str = None
) -> go.Figure:
    """
    Create a multi-line chart showing BX price trend for each node.
    
    Args:
        node_data: Either dict with node names as keys, or list of
                   {'date': date, 'node': str, 'avg_price': float} dicts
        bx_type: BX value (4-10) for title
        title: Custom title
        
    Returns:
        Plotly Figure object
    """
    fig = go.Figure()
    title = title or f'B{bx_type} Price Trend by Node'
    
    # Convert list format to dict format if needed
    if isinstance(node_data, list):
        node_dict = {}
        for item in node_data:
            node = item.get('node', 'Unknown')
            if node not in node_dict:
                node_dict[node] = []
            node_dict[node].append({'date': item['date'], 'avg_price': item['avg_price']})
        node_data = node_dict
    
    all_prices = []
    for i, (node, data) in enumerate(node_data.items()):
        if data:
            dates = [d['date'] for d in data]
            prices = [d['avg_price'] for d in data]
            all_prices.extend(prices)
            
            fig.add_trace(go.Scatter(
                x=dates,
                y=prices,
                mode='lines',
                name=node,
                line=dict(width=1.5),
                hovertemplate=f'{node}: $%{{y:.2f}}/MWh<extra></extra>'
            ))
    
    import numpy as np
    clipping_info = None
    yaxis_range = None
    if all_prices:
        actual_min = float(min(all_prices))
        actual_max = float(max(all_prices))
        p2 = float(np.percentile(all_prices, 2))
        p98 = float(np.percentile(all_prices, 98))
        margin = (p98 - p2) * 0.05
        if p2 != p98 and (actual_min < p2 or actual_max > p98):
            yaxis_range = [p2 - margin, p98 + margin]
            clipping_info = {
                'ymin': round(p2, 2), 'ymax': round(p98, 2),
                'actual_min': round(actual_min, 2), 'actual_max': round(actual_max, 2),
                'clipped_below': actual_min < p2, 'clipped_above': actual_max > p98
            }
    
    layout_kwargs = dict(
        title=title,
        xaxis_title='Date',
        yaxis_title='Avg Price ($/MWh)',
        hovermode='x unified',
        margin=dict(l=40, r=40, t=50, b=40),
        showlegend=len(node_data) <= 10,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
    )
    if yaxis_range:
        layout_kwargs['yaxis'] = dict(range=yaxis_range)
    fig.update_layout(**layout_kwargs)
    
    return fig, clipping_info


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
        textfont=dict(size=7, color='black'),
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


def create_8760_heatmap(
    data: list,
    title: str = 'Full Year Hourly Prices',
    year: int = None
):
    """
    Create a heatmap showing all 8760 hours of the year.
    
    Returns:
        Tuple of (Plotly Figure, clipping_info dict or None)
    """
    if not data:
        return create_empty_chart("No data for 8760 heatmap"), None
    
    df = pd.DataFrame(data)
    df['opr_dt'] = pd.to_datetime(df['opr_dt'])
    df['month'] = df['opr_dt'].dt.month
    df['day'] = df['opr_dt'].dt.day
    
    df = df.sort_values(['month', 'day', 'opr_hr'])
    
    df['row_label'] = df['opr_dt'].dt.strftime('%b %d')
    
    unique_dates = df[['opr_dt', 'row_label', 'month', 'day']].drop_duplicates().sort_values('opr_dt')
    date_order = unique_dates['row_label'].tolist()
    
    pivot = df.pivot_table(
        index='row_label', 
        columns='opr_hr', 
        values='avg_price', 
        aggfunc='mean'
    )
    pivot = pivot.reindex(date_order)
    pivot.columns = [int(h) for h in pivot.columns]
    pivot = pivot.reindex(columns=range(1, 25))
    
    z_values = pivot.values
    x_labels = [str(h) for h in range(1, 25)]
    y_labels = pivot.index.tolist()
    
    import numpy as np
    flat_vals = [v for row in z_values for v in row if pd.notna(v)]
    clipping_info = None
    if flat_vals:
        actual_min = float(min(flat_vals))
        actual_max = float(max(flat_vals))
        zmin = float(np.percentile(flat_vals, 2))
        zmax = float(np.percentile(flat_vals, 98))
        if zmin == zmax:
            zmin, zmax = actual_min, actual_max
        if actual_min < zmin or actual_max > zmax:
            clipping_info = {
                'zmin': round(zmin, 2), 'zmax': round(zmax, 2),
                'actual_min': round(actual_min, 2), 'actual_max': round(actual_max, 2),
                'clipped_below': actual_min < zmin, 'clipped_above': actual_max > zmax
            }
    else:
        zmin, zmax = None, None
    
    month_positions = []
    month_labels = []
    current_month = None
    for i, label in enumerate(y_labels):
        month_abbr = label[:3]
        if month_abbr != current_month:
            month_positions.append(i)
            month_labels.append(month_abbr)
            current_month = month_abbr
    
    fig = go.Figure(data=go.Heatmap(
        z=z_values,
        x=x_labels,
        y=list(range(len(y_labels))),
        colorscale=[
            [0.0, '#3366cc'],
            [0.25, '#66aaff'],
            [0.5, '#ffff99'],
            [0.75, '#ff9966'],
            [1.0, '#cc3300']
        ],
        zmin=zmin,
        zmax=zmax,
        hovertemplate='%{customdata}: Hour %{x}<br>$%{z:.2f}/MWh<extra></extra>',
        customdata=[[y_labels[i]] * 24 for i in range(len(y_labels))],
        showscale=True,
        colorbar=dict(title='$/MWh', tickformat='.0f')
    ))
    
    display_title = f"{title} - {year}" if year else title
    
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
            tickvals=month_positions,
            ticktext=month_labels
        ),
        margin=dict(l=60, r=40, t=50, b=60),
        height=800
    )
    
    return fig, clipping_info


def create_node_box_plot(
    stats_data: list,
    title: str = 'Price Distribution by Node',
    price_floor: float = -150.0
):
    """
    Create a box plot showing price distribution for each node.
    
    Values below price_floor (default -$150/MWh) are capped — daily BX averages
    below this threshold are transient anomalies that distort the chart.
    
    Returns:
        Tuple of (Plotly Figure, clipping_info dict or None)
    """
    if not stats_data:
        return create_empty_chart("No data for box plot"), None
    
    fig = go.Figure()
    
    nodes = [stat['node'] for stat in stats_data]
    
    clipped_count = 0
    original_mins = []
    for stat in stats_data:
        orig_min = stat['min']
        original_mins.append(orig_min)
        capped_min = max(orig_min, price_floor)
        if orig_min < price_floor:
            clipped_count += 1
        
        fig.add_trace(go.Box(
            name=stat['node'],
            y=[capped_min, stat['q1'], stat['median'], stat['q3'], stat['max']],
            boxpoints=False,
            hoverinfo='name+y'
        ))
    
    fig.add_trace(go.Scatter(
        x=nodes,
        y=[stat.get('mean', stat.get('avg', 0)) for stat in stats_data],
        mode='markers',
        name='Mean',
        marker=dict(symbol='diamond', size=10, color='red'),
        hovertemplate='Mean: $%{y:.2f}/MWh<extra></extra>'
    ))
    
    import numpy as np
    all_vals = []
    for stat in stats_data:
        all_vals.extend([max(stat['min'], price_floor), stat['q1'], stat['median'], stat['q3'], stat['max']])
    clipping_info = None
    yaxis_kwargs = {}
    if all_vals:
        p2 = float(np.percentile(all_vals, 2))
        p98 = float(np.percentile(all_vals, 98))
        margin = (p98 - p2) * 0.1
        if p2 != p98:
            yaxis_kwargs['range'] = [p2 - margin, p98 + margin]
    
    if clipped_count > 0:
        worst_min = min(original_mins)
        clipping_info = {
            'floor': price_floor,
            'clipped_count': clipped_count,
            'worst_original_min': round(worst_min, 2)
        }
    
    fig.update_layout(
        title=title,
        yaxis_title='Price ($/MWh)',
        showlegend=True,
        margin=dict(l=40, r=40, t=50, b=100),
        xaxis=dict(tickangle=45),
        yaxis=yaxis_kwargs
    )
    
    return fig, clipping_info


def _add_facility_traces(fig: go.Figure, facilities: list) -> None:
    """Add CARB facility markers to an existing mapbox figure."""
    if not facilities:
        return
    fdf = pd.DataFrame(facilities)
    for covered, label, color, size in [
        ('Yes', 'Covered Facilities', 'red', 12),
        ('No',  'Non-Covered Facilities', '#999999', 9),
    ]:
        sub = fdf[fdf['cap_and_trade'] == covered]
        if sub.empty:
            continue
        hover = (
            '<b>' + sub['facility'].astype(str) + '</b><br>'
            'Sector: ' + sub['primary_sector'].astype(str) + ' | ' + sub['county'].astype(str) + ' Co.<br>'
            'Cap-and-Trade: ' + sub['cap_and_trade'].astype(str) + '<br>'
            'Total GHG: ' + sub['total_ghg'].apply(lambda x: f'{x:,.0f}') + ' MT CO₂e (2023)<br>'
            'CO₂: ' + sub['co2'].apply(lambda x: f'{x:,.0f}') + '  '
            'NOx: ' + sub['nox'].apply(lambda x: f'{x:,.1f}') + '  '
            'SOx: ' + sub['sox'].apply(lambda x: f'{x:,.1f}') + '  '
            'PM2.5: ' + sub['pm25'].apply(lambda x: f'{x:,.1f}')
        )
        fig.add_trace(go.Scattermapbox(
            lat=sub['lat'].tolist(),
            lon=sub['lon'].tolist(),
            mode='markers',
            name=label,
            marker=dict(symbol='circle', size=size, color=color, opacity=0.9),
            customdata=hover.tolist(),
            hovertemplate='%{customdata}<extra></extra>',
        ))


def create_pnode_map(data: list, bx_label: str, color_by: str = 'zone',
                     facilities: list = None,
                     selected_facility: dict = None,
                     nearest_node: dict = None) -> go.Figure:
    """
    Create a geographic scatter map of PNODE BX prices with optional facility overlay.

    Args:
        data: List of dicts with keys: pnode_id, lat, lon, node_type, area, zone, avg_price
        bx_label: e.g. "B8" for hover/title text
        color_by: 'zone' or 'price'
        facilities: Optional list of facility dicts from get_facility_emissions()
        selected_facility: Optional single facility dict to highlight and centre the map on

    Returns:
        Plotly Figure
    """
    if not data:
        return create_empty_chart("No coordinate data available for the selected period")

    df = pd.DataFrame(data)
    df = df.dropna(subset=['lat', 'lon'])

    if df.empty:
        return create_empty_chart("No nodes with valid coordinates found")

    group_size = df.groupby(['lat', 'lon'])['lat'].transform('count')
    dup_mask = group_size > 1
    if dup_mask.any():
        RADIUS = 0.12
        rank = df.groupby(['lat', 'lon']).cumcount()
        angles = 2 * np.pi * rank[dup_mask] / group_size[dup_mask]
        df.loc[dup_mask, 'lat'] = df.loc[dup_mask, 'lat'] + RADIUS * np.sin(angles)
        df.loc[dup_mask, 'lon'] = df.loc[dup_mask, 'lon'] + RADIUS * np.cos(angles)

    zone_colors = {'NP15': '#1f77b4', 'SP15': '#ff7f0e', 'ZP26': '#2ca02c'}
    default_color = '#aec7e8'

    if color_by == 'zone':
        df['color_zone'] = df['zone'].fillna('Other')
        color_discrete_map = {
            'NP15': zone_colors['NP15'],
            'SP15': zone_colors['SP15'],
            'ZP26': zone_colors['ZP26'],
            'Other': '#999999',
        }
        df['hover_text'] = (
            '<b>' + df['pnode_id'].astype(str) + '</b><br>' +
            'Zone: ' + df['color_zone'].astype(str) + '<br>' +
            'Type: ' + df['node_type'].fillna('').astype(str) + '<br>' +
            'Area: ' + df['area'].fillna('').astype(str) + '<br>' +
            bx_label + ' Avg: $' + df['avg_price'].apply(
                lambda x: f'{x:.2f}' if pd.notna(x) else 'N/A'
            ) + '/MWh'
        )
        fig = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            color='color_zone',
            color_discrete_map=color_discrete_map,
            custom_data=['hover_text'],
            zoom=5,
            center={'lat': 37.0, 'lon': -119.0},
            mapbox_style='carto-positron',
            title=f'PNODE {bx_label} Price Map — Colored by Zone',
            height=650,
        )
        fig.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>',
            marker=dict(size=6, opacity=0.75)
        )
        fig.update_layout(
            legend_title_text='Zone',
            margin=dict(l=0, r=0, t=50, b=0)
        )

    else:
        df = df.dropna(subset=['avg_price'])
        if df.empty:
            return create_empty_chart("No price data available for coloring")

        p2 = float(np.percentile(df['avg_price'], 2))
        p98 = float(np.percentile(df['avg_price'], 98))

        df['hover_text'] = (
            '<b>' + df['pnode_id'].astype(str) + '</b><br>' +
            'Zone: ' + df['zone'].fillna('Other').astype(str) + '<br>' +
            'Type: ' + df['node_type'].fillna('').astype(str) + '<br>' +
            'Area: ' + df['area'].fillna('').astype(str) + '<br>' +
            bx_label + ' Avg: $' + df['avg_price'].apply(lambda x: f'{x:.2f}') + '/MWh'
        )
        fig = px.scatter_mapbox(
            df,
            lat='lat',
            lon='lon',
            color='avg_price',
            color_continuous_scale='RdYlGn_r',
            range_color=[p2, p98],
            custom_data=['hover_text'],
            zoom=5,
            center={'lat': 37.0, 'lon': -119.0},
            mapbox_style='carto-positron',
            title=f'PNODE {bx_label} Price Map — Colored by Price',
            height=650,
        )
        fig.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>',
            marker=dict(size=6, opacity=0.75)
        )
        fig.update_layout(
            coloraxis_colorbar=dict(title='$/MWh'),
            margin=dict(l=0, r=0, t=50, b=0)
        )

    _add_facility_traces(fig, facilities)

    if selected_facility:
        slat = selected_facility['lat']
        slon = selected_facility['lon']
        hover_sel = (
            f"<b>{selected_facility['facility']}</b><br>"
            f"Sector: {selected_facility['primary_sector']} | {selected_facility['county']} Co.<br>"
            f"Cap-and-Trade: {selected_facility['cap_and_trade']}<br>"
            f"Total GHG: {selected_facility['total_ghg']:,.0f} MT CO₂e (2023)<br>"
            f"CO₂: {selected_facility['co2']:,.0f}  "
            f"NOx: {selected_facility['nox']:,.1f}  "
            f"SOx: {selected_facility['sox']:,.1f}  "
            f"PM2.5: {selected_facility['pm25']:,.1f}"
        )
        fig.add_trace(go.Scattermapbox(
            lat=[slat],
            lon=[slon],
            mode='markers',
            name='Selected Facility',
            marker=dict(size=20, color='gold', opacity=1.0),
            customdata=[hover_sel],
            hovertemplate='%{customdata}<extra></extra>',
        ))
        fig.update_layout(
            mapbox=dict(
                center=dict(lat=slat, lon=slon),
                zoom=11,
                style='carto-positron',
            )
        )

    if nearest_node and nearest_node.get('lat') is not None:
        price_str = f"${nearest_node['avg_price']:.2f}/MWh" if nearest_node.get('avg_price') is not None else "N/A"
        hover_nn = (
            f"<b>Nearest node: {nearest_node['pnode_id']}</b><br>"
            f"Zone: {nearest_node.get('zone') or '—'}<br>"
            f"{bx_label} avg: {price_str}"
        )
        fig.add_trace(go.Scattermapbox(
            lat=[nearest_node['lat']],
            lon=[nearest_node['lon']],
            mode='markers',
            name='Nearest Node',
            marker=dict(size=16, color='cyan', opacity=1.0),
            customdata=[hover_nn],
            hovertemplate='%{customdata}<extra></extra>',
        ))

    return fig


def create_pnode_price_histogram(data: list, bx_label: str) -> go.Figure:
    """
    Create a stacked bar histogram of node count by price bin, colored by zone.

    Args:
        data: List of dicts with keys: pnode_id, avg_price, zone
        bx_label: e.g. "B8" for axis labels

    Returns:
        Plotly Figure
    """
    if not data:
        return create_empty_chart("No data for histogram")

    df = pd.DataFrame(data).dropna(subset=['avg_price'])
    if df.empty:
        return create_empty_chart("No price data for histogram")

    p2 = float(np.percentile(df['avg_price'], 2))
    p98 = float(np.percentile(df['avg_price'], 98))
    df_clip = df[(df['avg_price'] >= p2) & (df['avg_price'] <= p98)].copy()
    df_clip['zone_label'] = df_clip['zone'].fillna('Other')

    bin_size = 5
    bin_min = (p2 // bin_size) * bin_size
    bin_max = ((p98 // bin_size) + 1) * bin_size
    bins = np.arange(bin_min, bin_max + bin_size, bin_size)
    df_clip['bin'] = pd.cut(df_clip['avg_price'], bins=bins, right=False)
    df_clip['bin_mid'] = df_clip['bin'].apply(lambda x: (x.left + x.right) / 2 if pd.notna(x) else None)

    zone_color_map = {
        'NP15': '#1f77b4',
        'SP15': '#ff7f0e',
        'ZP26': '#2ca02c',
    }
    zone_order = ['NP15', 'SP15', 'ZP26']

    df_clip = df_clip[df_clip['zone_label'].isin(zone_order)]

    fig = go.Figure()
    for zone in zone_order:
        zone_df = df_clip[df_clip['zone_label'] == zone]
        if zone_df.empty:
            continue
        counts = zone_df.groupby('bin_mid', observed=True).size().reset_index(name='count')
        fig.add_trace(go.Bar(
            x=counts['bin_mid'],
            y=counts['count'],
            name=zone,
            marker_color=zone_color_map.get(zone, '#999'),
            hovertemplate=f'<b>{zone}</b><br>Price: $%{{x:.0f}}/MWh<br>Nodes: %{{y}}<extra></extra>',
        ))

    fig.update_layout(
        barmode='stack',
        title=f'{bx_label} Price Distribution by Node',
        xaxis_title=f'{bx_label} Avg Price ($/MWh)',
        yaxis_title='Node Count',
        legend_title='Zone',
        bargap=0.05,
        height=260,
        margin=dict(l=40, r=40, t=45, b=40),
    )
    return fig
