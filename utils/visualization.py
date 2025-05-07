import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import networkx as nx
from pyvis.network import Network
import matplotlib.pyplot as plt
from io import StringIO
import tempfile

def plot_association_network(rules_df, min_lift=1.2, max_nodes=50):
    """
    Create an interactive network visualization of product associations
    
    Parameters:
    -----------
    rules_df : pandas DataFrame
        Association rules DataFrame
    min_lift : float, optional
        Minimum lift value for displaying rules
    max_nodes : int, optional
        Maximum number of nodes to display for better visualization
    
    Returns:
    --------
    str
        HTML string of the network visualization
    """
    if rules_df.empty:
        # Return empty HTML if no rules
        return "<p>No association rules to display</p>"
        
    # Filter rules by lift
    rules_df = rules_df[rules_df['lift'] >= min_lift]
    
    if len(rules_df) == 0:
        return "<p>No association rules meet the minimum lift criteria</p>"
    
    # Create a network graph
    G = nx.Graph()
    
    # Limit the number of rules for better visualization
    if len(rules_df) > max_nodes:
        rules_df = rules_df.sort_values('lift', ascending=False).head(max_nodes)
    
    # Add nodes and edges
    for _, row in rules_df.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        
        # Add all antecedents and consequents as nodes
        for item in antecedents + consequents:
            if not G.has_node(item):
                G.add_node(item)
        
        # Add edges between antecedents and consequents
        for ant in antecedents:
            for cons in consequents:
                if G.has_edge(ant, cons):
                    # If edge exists, update weight based on lift
                    current_weight = G[ant][cons]['weight']
                    G[ant][cons]['weight'] = max(current_weight, row['lift'])
                    G[ant][cons]['confidence'] = max(G[ant][cons].get('confidence', 0), row['confidence'])
                else:
                    # Add new edge
                    G.add_edge(ant, cons, weight=row['lift'], confidence=row['confidence'])
    
    # Create a PyVis network
    net = Network(height="500px", width="100%", notebook=True, directed=False)
    
    # Calculate node size based on degree centrality
    degree_centrality = nx.degree_centrality(G)
    
    # Add nodes with size based on centrality
    for node in G.nodes():
        size = 20 + (degree_centrality[node] * 50)
        net.add_node(node, label=str(node), size=size, title=f"Product: {node}")
    
    # Add edges with width based on lift and color based on confidence
    for u, v, data in G.edges(data=True):
        width = 1 + (data['weight'] - min_lift) * 3  # Scale width based on lift
        
        # Color based on confidence (green for high, yellow for medium, red for low)
        confidence = data['confidence']
        if confidence >= 0.75:
            color = "#00CC00"  # Green
        elif confidence >= 0.5:
            color = "#FFCC00"  # Yellow
        else:
            color = "#FF6666"  # Light red
        
        title = f"Lift: {data['weight']:.2f}, Confidence: {data['confidence']:.2f}"
        net.add_edge(u, v, width=width, title=title, color=color)
    
    # Set physics layout
    net.barnes_hut(gravity=-5000, central_gravity=0.1, spring_length=200)
    
    # Generate HTML
    html_string = net.generate_html()
    
    return html_string

def plot_forecasting_results(data, predictions, products_to_forecast, n_rows=2):
    """
    Plot forecasting results with actual vs predicted values
    
    Parameters:
    -----------
    data : pandas DataFrame
        Historical data
    predictions : pandas DataFrame
        Forecasted data
    products_to_forecast : list
        List of product IDs to display
    n_rows : int, optional
        Number of rows in the subplot grid
    """
    if len(products_to_forecast) == 0:
        st.warning("No products selected for forecasting.")
        return
    
    # Calculate number of columns needed
    n_cols = (len(products_to_forecast) + n_rows - 1) // n_rows
    
    # Create subplots
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=[f"Product {p}" for p in products_to_forecast],
        shared_xaxes=False
    )
    
    # Add traces for each product
    for i, product_id in enumerate(products_to_forecast):
        # Calculate row and column indices
        row_idx = i // n_cols + 1
        col_idx = i % n_cols + 1
        
        # Filter historical data for this product
        historical = data[data['ProductID'] == product_id].copy()
        
        if len(historical) > 0:
            # Aggregate by date
            historical_daily = historical.groupby('Date')['Quantity'].sum().reset_index()
            
            # Add historical trace
            fig.add_trace(
                go.Scatter(
                    x=historical_daily['Date'],
                    y=historical_daily['Quantity'],
                    mode='lines',
                    name=f'Historical - {product_id}',
                    line=dict(color='blue'),
                    showlegend=False
                ),
                row=row_idx, col=col_idx
            )
        
        # Filter predictions for this product
        product_predictions = predictions[predictions['ProductID'] == product_id]
        
        if len(product_predictions) > 0:
            # Add prediction trace
            fig.add_trace(
                go.Scatter(
                    x=product_predictions['Date'],
                    y=product_predictions['Predicted_Quantity'],
                    mode='lines',
                    name=f'Forecast - {product_id}',
                    line=dict(color='red', dash='dash'),
                    showlegend=False
                ),
                row=row_idx, col=col_idx
            )
    
    # Update layout
    fig.update_layout(
        height=300 * n_rows,
        width=1000,
        title_text="Product Demand Forecasts",
        hovermode="closest"
    )
    
    # Add a single legend
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name='Historical',
            line=dict(color='blue'),
            showlegend=True
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash'),
            showlegend=True
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_top_rules_table(rules_df, top_n=10):
    """
    Display a formatted table of top association rules
    
    Parameters:
    -----------
    rules_df : pandas DataFrame
        Association rules DataFrame
    top_n : int, optional
        Number of top rules to display
    """
    if rules_df.empty:
        st.warning("No association rules found.")
        return
    
    # Sort by lift and get top N rules
    top_rules = rules_df.sort_values('lift', ascending=False).head(top_n)
    
    # Format the rules for display
    formatted_rules = []
    
    for _, row in top_rules.iterrows():
        antecedents = ", ".join(list(row['antecedents']))
        consequents = ", ".join(list(row['consequents']))
        support = row['support']
        confidence = row['confidence']
        lift = row['lift']
        
        formatted_rules.append({
            "Antecedents": antecedents,
            "Consequents": consequents,
            "Support": f"{support:.3f}",
            "Confidence": f"{confidence:.3f}",
            "Lift": f"{lift:.3f}"
        })
    
    # Display as a table
    st.table(pd.DataFrame(formatted_rules))

def plot_product_associations_heatmap(rules_df, top_products):
    """
    Create a heatmap of product associations
    
    Parameters:
    -----------
    rules_df : pandas DataFrame
        Association rules DataFrame
    top_products : list
        List of top product IDs to include in the heatmap
    """
    if rules_df.empty or len(top_products) == 0:
        st.warning("No association rules or products to display.")
        return
    
    # Create a matrix of lift values between products
    lift_matrix = pd.DataFrame(0, index=top_products, columns=top_products)
    
    # Fill the matrix with lift values from rules
    for _, row in rules_df.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        lift = row['lift']
        
        # Check all combinations of antecedents and consequents
        for ant in antecedents:
            if ant in top_products:
                for cons in consequents:
                    if cons in top_products:
                        # Update matrix with max lift
                        current_lift = lift_matrix.loc[ant, cons]
                        lift_matrix.loc[ant, cons] = max(current_lift, lift)
                        
                        # Make matrix symmetric
                        lift_matrix.loc[cons, ant] = lift_matrix.loc[ant, cons]
    
    # Set diagonal to NaN (no self-association)
    for product in top_products:
        lift_matrix.loc[product, product] = np.nan
    
    # Create heatmap with Plotly
    fig = px.imshow(
        lift_matrix,
        labels=dict(x="Product", y="Product", color="Lift"),
        x=lift_matrix.columns,
        y=lift_matrix.index,
        color_continuous_scale="Viridis",
        title="Product Association Heatmap (Lift Values)"
    )
    
    fig.update_layout(
        height=600,
        width=700,
        xaxis=dict(tickangle=45),
    )
    
    st.plotly_chart(fig, use_container_width=True)

def plot_product_sales_trend(data, selected_products, time_aggregation='week'):
    """
    Plot sales trends for selected products over time
    
    Parameters:
    -----------
    data : pandas DataFrame
        Historical transaction data
    selected_products : list
        List of product IDs to display
    time_aggregation : str, optional
        Time aggregation level ('day', 'week', 'month')
    """
    if len(selected_products) == 0:
        st.warning("No products selected for visualization.")
        return
    
    # Filter data for selected products
    filtered_data = data[data['ProductID'].isin(selected_products)].copy()
    
    if len(filtered_data) == 0:
        st.warning("No data available for the selected products.")
        return
    
    # Aggregate by time period
    if time_aggregation == 'day':
        # Daily aggregation
        time_group = filtered_data.groupby(['Date', 'ProductID'])['Quantity'].sum().reset_index()
    elif time_aggregation == 'week':
        # Weekly aggregation
        filtered_data['Week'] = filtered_data['Date'].dt.to_period('W').dt.start_time
        time_group = filtered_data.groupby(['Week', 'ProductID'])['Quantity'].sum().reset_index()
        time_group.rename(columns={'Week': 'Date'}, inplace=True)
    elif time_aggregation == 'month':
        # Monthly aggregation
        filtered_data['Month'] = filtered_data['Date'].dt.to_period('M').dt.start_time
        time_group = filtered_data.groupby(['Month', 'ProductID'])['Quantity'].sum().reset_index()
        time_group.rename(columns={'Month': 'Date'}, inplace=True)
    
    # Create line chart
    fig = px.line(
        time_group,
        x='Date',
        y='Quantity',
        color='ProductID',
        labels={'Quantity': 'Sales Quantity', 'Date': f'{time_aggregation.capitalize()} Date'},
        title=f'Product Sales Trends ({time_aggregation.capitalize()}ly)'
    )
    
    # Customize layout
    fig.update_layout(
        xaxis_title=f"{time_aggregation.capitalize()} Date",
        yaxis_title="Sales Quantity",
        legend_title="Product ID",
        hovermode="x unified",
        width=900,  # Fixed width for better visualization without needing fullscreen
        height=600, # Fixed height
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show descriptive statistics
    st.subheader(f"Sales Statistics ({time_aggregation.capitalize()}ly)")
    
    stats = time_group.groupby('ProductID')['Quantity'].agg([
        ('Mean', 'mean'),
        ('Median', 'median'),
        ('Min', 'min'),
        ('Max', 'max'),
        ('Total', 'sum')
    ]).reset_index()
    
    # Format statistics
    stats['Mean'] = stats['Mean'].round(2)
    stats['Median'] = stats['Median'].round(2)
    
    st.dataframe(stats)
