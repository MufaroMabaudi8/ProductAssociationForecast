import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def calculate_safety_stock(historical_demand, service_level=0.95, lead_time_days=7):
    """
    Calculate safety stock based on demand variability and desired service level
    
    Parameters:
    -----------
    historical_demand : pandas DataFrame
        Historical daily demand data for each product
    service_level : float, optional
        Desired service level (default: 0.95 or 95%)
    lead_time_days : int, optional
        Lead time in days for replenishment
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with safety stock recommendations for each product
    """
    # Group by product and calculate statistics
    demand_stats = historical_demand.groupby('ProductID')['Quantity'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('max', 'max'),
        ('min', 'min')
    ]).reset_index()
    
    # Get service factor based on service level (assuming normal distribution)
    # 95% service level corresponds to 1.645 standard deviations
    service_factors = {
        0.90: 1.282,
        0.95: 1.645,
        0.98: 2.054,
        0.99: 2.326
    }
    service_factor = service_factors.get(service_level, 1.645)
    
    # Calculate safety stock
    demand_stats['lead_time_factor'] = np.sqrt(lead_time_days)
    demand_stats['safety_stock'] = demand_stats['std'] * service_factor * demand_stats['lead_time_factor']
    
    # Handle cases where standard deviation is zero (no variability)
    demand_stats.loc[demand_stats['std'] == 0, 'safety_stock'] = demand_stats.loc[demand_stats['std'] == 0, 'mean'] * 0.2
    
    # Round up to nearest integer
    demand_stats['safety_stock'] = np.ceil(demand_stats['safety_stock']).astype(int)
    
    return demand_stats[['ProductID', 'mean', 'std', 'safety_stock']]

def calculate_reorder_points(historical_demand, forecast_demand, lead_time_days=7, service_level=0.95):
    """
    Calculate reorder points for each product based on lead time and forecast
    
    Parameters:
    -----------
    historical_demand : pandas DataFrame
        Historical daily demand data for each product
    forecast_demand : pandas DataFrame
        Forecasted daily demand data for each product
    lead_time_days : int, optional
        Lead time in days for replenishment
    service_level : float, optional
        Desired service level (default: 0.95 or 95%)
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with reorder points for each product
    """
    # Get safety stock
    safety_stock_df = calculate_safety_stock(historical_demand, service_level, lead_time_days)
    
    # Calculate lead time demand from forecast
    today = datetime.now().date()
    lead_time_end = today + timedelta(days=lead_time_days)
    
    # Filter forecast for lead time period
    lead_time_forecast = forecast_demand[
        (forecast_demand['Date'].dt.date >= today) & 
        (forecast_demand['Date'].dt.date <= lead_time_end)
    ]
    
    # Calculate total demand during lead time for each product
    lead_time_demand = lead_time_forecast.groupby('ProductID')['Predicted_Quantity'].sum().reset_index()
    lead_time_demand.rename(columns={'Predicted_Quantity': 'lead_time_demand'}, inplace=True)
    
    # Merge with safety stock
    reorder_points = pd.merge(safety_stock_df, lead_time_demand, on='ProductID', how='left')
    
    # Fill NaN values in lead time demand (products without forecast)
    reorder_points['lead_time_demand'].fillna(
        reorder_points['mean'] * lead_time_days, 
        inplace=True
    )
    
    # Calculate reorder point = lead time demand + safety stock
    reorder_points['reorder_point'] = np.ceil(
        reorder_points['lead_time_demand'] + reorder_points['safety_stock']
    ).astype(int)
    
    return reorder_points[['ProductID', 'mean', 'std', 'safety_stock', 'lead_time_demand', 'reorder_point']]

def calculate_economic_order_quantity(historical_demand, holding_cost_pct=0.25, ordering_cost=25):
    """
    Calculate Economic Order Quantity (EOQ) for each product
    
    Parameters:
    -----------
    historical_demand : pandas DataFrame
        Historical daily demand data with ProductID, Quantity, and Price columns
    holding_cost_pct : float, optional
        Annual holding cost as percentage of product value
    ordering_cost : float, optional
        Fixed cost per order in currency units
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with EOQ recommendations for each product
    """
    # Verify required columns exist
    required_cols = ['ProductID', 'Quantity', 'Price']
    for col in required_cols:
        if col not in historical_demand.columns:
            # Return empty DataFrame if required columns don't exist
            return pd.DataFrame(columns=['ProductID', 'annual_demand', 'unit_price', 'eoq'])

    # Calculate annual demand per product
    days_in_data = (historical_demand['Date'].max() - historical_demand['Date'].min()).days
    if days_in_data < 1:
        days_in_data = 1
    
    # Scaling factor to convert to annual demand
    annual_factor = 365 / days_in_data
    
    # Get average price per product
    price_data = historical_demand.groupby('ProductID')['Price'].mean().reset_index()
    
    # Calculate total demand per product
    demand_data = historical_demand.groupby('ProductID')['Quantity'].sum().reset_index()
    demand_data['annual_demand'] = demand_data['Quantity'] * annual_factor
    
    # Combine price and demand data
    product_data = pd.merge(demand_data, price_data, on='ProductID')
    
    # Calculate holding cost per unit
    product_data['holding_cost'] = product_data['Price'] * holding_cost_pct
    
    # Calculate EOQ
    product_data['eoq'] = np.sqrt(
        (2 * product_data['annual_demand'] * ordering_cost) / 
        product_data['holding_cost']
    )
    
    # Round to nearest integer
    product_data['eoq'] = np.ceil(product_data['eoq']).astype(int)
    
    return product_data[['ProductID', 'annual_demand', 'Price', 'eoq']]

def get_inventory_recommendations(historical_data, forecast_data, lead_time_days=7, service_level=0.95):
    """
    Generate comprehensive inventory recommendations based on forecasts and historical data
    
    Parameters:
    -----------
    historical_data : pandas DataFrame
        Historical transaction data
    forecast_data : pandas DataFrame
        Forecasted demand data
    lead_time_days : int, optional
        Lead time in days for replenishment
    service_level : float, optional
        Desired service level
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with inventory recommendations
    """
    # Calculate reorder points
    reorder_points = calculate_reorder_points(
        historical_data, 
        forecast_data, 
        lead_time_days, 
        service_level
    )
    
    # Calculate EOQ if price data is available
    if 'Price' in historical_data.columns:
        eoq_data = calculate_economic_order_quantity(historical_data)
        
        # Merge reorder points with EOQ
        if not eoq_data.empty:
            recommendations = pd.merge(
                reorder_points, 
                eoq_data[['ProductID', 'eoq']], 
                on='ProductID', 
                how='left'
            )
        else:
            recommendations = reorder_points
            recommendations['eoq'] = np.nan
    else:
        recommendations = reorder_points
        recommendations['eoq'] = np.nan
    
    # Fill missing EOQ values with a simple formula
    if 'eoq' in recommendations.columns:
        recommendations['eoq'].fillna(recommendations['reorder_point'] * 2, inplace=True)
        recommendations['eoq'] = recommendations['eoq'].astype(int)
    
    # Calculate days of supply based on average daily demand
    recommendations['days_of_supply'] = np.ceil(
        recommendations['eoq'] / recommendations['mean']
    ).astype(int)
    
    # Clean up column names for display
    recommendations.rename(columns={
        'mean': 'avg_daily_demand',
        'std': 'demand_std_dev',
        'lead_time_demand': 'lead_time_demand'
    }, inplace=True)
    
    return recommendations

def get_bundle_inventory_recommendations(historical_data, forecast_data, product_bundles, lead_time_days=7, service_level=0.95):
    """
    Generate inventory recommendations for product bundles
    
    Parameters:
    -----------
    historical_data : pandas DataFrame
        Historical transaction data
    forecast_data : pandas DataFrame
        Forecasted demand data
    product_bundles : list
        List of product bundles [(bundle_items, confidence, lift), ...]
    lead_time_days : int, optional
        Lead time in days for replenishment
    service_level : float, optional
        Desired service level (default: 0.95 or 95%)
    
    Returns:
    --------
    pandas DataFrame
        DataFrame with bundle inventory recommendations
    """
    bundle_recommendations = []
    
    for bundle_items, confidence, lift in product_bundles:
        # Skip bundles with less than 2 items
        if len(bundle_items) < 2:
            continue
        
        # Extract product IDs from bundle
        product_ids = list(bundle_items)
        
        # Get individual product recommendations
        individual_recs = get_inventory_recommendations(
            historical_data[historical_data['ProductID'].isin(product_ids)],
            forecast_data[forecast_data['ProductID'].isin(product_ids)],
            lead_time_days=lead_time_days,
            service_level=service_level
        )
        
        # Calculate average recommendation values for the bundle
        if not individual_recs.empty:
            bundle_rec = {
                'bundle_id': '-'.join(product_ids),
                'products': ', '.join(product_ids),
                'confidence': confidence,
                'lift': lift,
                'avg_safety_stock': round(individual_recs['safety_stock'].mean()),
                'avg_reorder_point': round(individual_recs['reorder_point'].mean()),
                'min_days_supply': individual_recs['days_of_supply'].min(),
                'suggested_bundle_stock': round(
                    (individual_recs['reorder_point'].mean() * confidence) * 
                    (1 + (0.1 * (lift - 1)))  # Adjust by lift impact
                )
            }
            bundle_recommendations.append(bundle_rec)
    
    # Convert to DataFrame
    if bundle_recommendations:
        return pd.DataFrame(bundle_recommendations)
    else:
        return pd.DataFrame(columns=[
            'bundle_id', 'products', 'confidence', 'lift', 
            'avg_safety_stock', 'avg_reorder_point', 
            'min_days_supply', 'suggested_bundle_stock'
        ])