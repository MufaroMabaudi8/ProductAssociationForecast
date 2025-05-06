import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def train_forecasting_model(data, products_to_forecast, association_rules=None, train_ratio=0.8):
    """
    Train XGBoost forecasting model for the selected products
    
    Parameters:
    -----------
    data : pandas DataFrame
        The transaction data
    products_to_forecast : list
        List of product IDs to forecast
    association_rules : pandas DataFrame, optional
        Association rules to include as features
    train_ratio : float, optional
        Ratio of data to use for training
    
    Returns:
    --------
    XGBRegressor
        Trained XGBoost model
    """
    from utils.data_processing import prepare_forecast_features
    
    # Prepare features for forecasting
    forecast_data = prepare_forecast_features(data, association_rules)
    
    # Filter for selected products
    forecast_data = forecast_data[forecast_data['ProductID'].isin(products_to_forecast)]
    
    # Drop rows with missing values
    forecast_data = forecast_data.dropna()
    
    if len(forecast_data) == 0:
        raise ValueError("No valid data for forecasting after preprocessing.")
    
    # Prepare feature matrix
    # Use all columns except Date, ProductID, and Quantity (target)
    feature_columns = [col for col in forecast_data.columns 
                       if col not in ['Date', 'ProductID', 'Quantity']]
    
    X = forecast_data[feature_columns]
    y = forecast_data['Quantity']
    
    # One-hot encode ProductID
    X = pd.get_dummies(X, columns=['ProductID'], drop_first=False)
    
    # Split data chronologically
    # Sort by date to ensure proper chronological splitting
    forecast_data = forecast_data.sort_values('Date')
    
    # Calculate split point
    split_index = int(len(forecast_data) * train_ratio)
    
    # Get train data
    train_data = forecast_data.iloc[:split_index]
    X_train = train_data[feature_columns]
    y_train = train_data['Quantity']
    
    # One-hot encode ProductID for training data
    X_train = pd.get_dummies(X_train, columns=['ProductID'], drop_first=False)
    
    # Train XGBoost model
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    
    model.fit(X_train, y_train)
    
    # Store feature names for later use
    model.feature_names_in_ = X_train.columns.tolist()
    
    return model

def predict_demand(model, data, products_to_forecast, horizon=30):
    """
    Generate demand forecasts for the specified products
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained XGBoost model
    data : pandas DataFrame
        The transaction data
    products_to_forecast : list
        List of product IDs to forecast
    horizon : int, optional
        Forecast horizon in days
    
    Returns:
    --------
    pandas DataFrame
        Forecasted demand for each product by day
    """
    from utils.data_processing import aggregate_daily_sales, create_time_features
    
    # Get the latest date in the data
    latest_date = data['Date'].max()
    
    # Create a DataFrame for future dates
    future_dates = [latest_date + timedelta(days=i+1) for i in range(horizon)]
    
    # Create predictions for each product
    all_predictions = []
    
    for product_id in products_to_forecast:
        # Filter historical data for this product
        product_data = data[data['ProductID'] == product_id]
        
        if len(product_data) == 0:
            continue
        
        # Aggregate to daily level
        daily_data = aggregate_daily_sales(product_data)
        
        # Add time features
        daily_data = create_time_features(daily_data)
        
        # Get the latest values for lag features
        latest_values = {}
        
        # Get the last 30 days of data for calculating lag features
        last_30d_data = daily_data.sort_values('Date', ascending=False).head(30)
        
        if len(last_30d_data) > 0:
            # Get the most recent quantity value
            latest_qty = last_30d_data.iloc[0]['Quantity']
            
            # Get recent values for different lags
            latest_values['Quantity_Lag1'] = latest_qty
            
            # 7-day lag (get value from 7 days ago if available)
            if len(last_30d_data) >= 7:
                latest_values['Quantity_Lag7'] = last_30d_data.iloc[6]['Quantity']
            else:
                latest_values['Quantity_Lag7'] = latest_qty
            
            # 14-day lag
            if len(last_30d_data) >= 14:
                latest_values['Quantity_Lag14'] = last_30d_data.iloc[13]['Quantity']
            else:
                latest_values['Quantity_Lag14'] = latest_qty
            
            # 30-day lag
            if len(last_30d_data) >= 30:
                latest_values['Quantity_Lag30'] = last_30d_data.iloc[29]['Quantity']
            else:
                latest_values['Quantity_Lag30'] = latest_qty
            
            # Rolling means
            latest_values['Quantity_RollingMean7'] = last_30d_data.head(7)['Quantity'].mean()
            latest_values['Quantity_RollingMean14'] = last_30d_data.head(14)['Quantity'].mean()
            latest_values['Quantity_RollingMean30'] = last_30d_data['Quantity'].mean()
            
            # Rolling standard deviation
            latest_values['Quantity_RollingStd7'] = last_30d_data.head(7)['Quantity'].std()
            if np.isnan(latest_values['Quantity_RollingStd7']):
                latest_values['Quantity_RollingStd7'] = 0
        else:
            # If no recent data, use zeros
            for key in ['Quantity_Lag1', 'Quantity_Lag7', 'Quantity_Lag14', 'Quantity_Lag30',
                       'Quantity_RollingMean7', 'Quantity_RollingMean14', 'Quantity_RollingMean30',
                       'Quantity_RollingStd7']:
                latest_values[key] = 0
        
        # Create a DataFrame for future predictions
        future_df = pd.DataFrame({'Date': future_dates})
        future_df['ProductID'] = product_id
        
        # Add time features
        future_df = create_time_features(future_df)
        
        # Initialize lag features with the latest values
        for key, value in latest_values.items():
            future_df[key] = value
        
        # If there are association features, initialize with zeros (simplified)
        if 'AssociatedProductSales' in model.feature_names_in_:
            future_df['AssociatedProductSales'] = 0
        
        # Create one-hot encoded product ID columns
        for prod_id in products_to_forecast:
            col_name = f'ProductID_{prod_id}'
            future_df[col_name] = 1 if prod_id == product_id else 0
        
        # Prepare feature columns for prediction
        feature_columns = [col for col in model.feature_names_in_ if col in future_df.columns]
        
        # Create a copy of feature_names_in_ as a set for faster lookup
        feature_names_set = set(model.feature_names_in_)
        
        # Check for missing columns and add them with zeros
        for col in model.feature_names_in_:
            if col not in future_df.columns:
                future_df[col] = 0
        
        # Make predictions
        X_future = future_df[model.feature_names_in_]
        future_df['Predicted_Quantity'] = model.predict(X_future)
        
        # Ensure predictions are non-negative
        future_df['Predicted_Quantity'] = future_df['Predicted_Quantity'].clip(lower=0)
        
        # Add to all predictions
        all_predictions.append(future_df[['Date', 'ProductID', 'Predicted_Quantity']])
    
    # Combine all predictions
    if all_predictions:
        combined_predictions = pd.concat(all_predictions, ignore_index=True)
        return combined_predictions
    else:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Date', 'ProductID', 'Predicted_Quantity'])

def evaluate_forecast_accuracy(actual, predicted):
    """
    Evaluate the accuracy of forecasts using MAPE and RMSE
    
    Parameters:
    -----------
    actual : Series or array
        Actual values
    predicted : Series or array
        Predicted values
    
    Returns:
    --------
    dict
        Dictionary with MAPE and RMSE metrics
    """
    from sklearn.metrics import mean_squared_error
    
    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    
    # Calculate MAPE
    actual_array = np.array(actual)
    pred_array = np.array(predicted)
    
    # Avoid division by zero
    mask = actual_array != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((actual_array[mask] - pred_array[mask]) / actual_array[mask])) * 100
    else:
        mape = np.nan
    
    return {
        'RMSE': rmse,
        'MAPE': mape
    }

def generate_scenario_forecasts(model, base_predictions, scenario_adjustments):
    """
    Generate forecasts for different scenarios by adjusting base predictions
    
    Parameters:
    -----------
    model : XGBRegressor
        Trained XGBoost model
    base_predictions : pandas DataFrame
        Base forecasts
    scenario_adjustments : dict
        Dictionary with scenario adjustments
    
    Returns:
    --------
    dict
        Dictionary with scenario forecasts
    """
    scenarios = {}
    
    # Create a copy of base predictions
    base_df = base_predictions.copy()
    
    # Generate forecasts for each scenario
    for scenario_name, adjustments in scenario_adjustments.items():
        # Create a copy for this scenario
        scenario_df = base_df.copy()
        
        # Apply adjustments
        for product_id, adjustment_factor in adjustments.items():
            scenario_df.loc[scenario_df['ProductID'] == product_id, 'Predicted_Quantity'] *= adjustment_factor
        
        scenarios[scenario_name] = scenario_df
    
    return scenarios
