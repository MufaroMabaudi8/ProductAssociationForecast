import pandas as pd
import numpy as np
from datetime import datetime

def load_and_preprocess_data(file_object):
    """
    Load and preprocess data from a CSV or Excel file
    
    Parameters:
    -----------
    file_object : file object
        Uploaded file object from Streamlit
    
    Returns:
    --------
    tuple
        (DataFrame with preprocessed data, file type string)
    """
    file_type = file_object.name.split('.')[-1].lower()
    
    if file_type == 'csv':
        df = pd.read_csv(file_object)
    elif file_type in ['xlsx', 'xls']:
        df = pd.read_excel(file_object)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or Excel file.")
    
    # Convert column names to standard format
    column_mapping = standardize_column_names(df.columns)
    df = df.rename(columns=column_mapping)
    
    # Ensure required columns exist
    required_columns = ['Date', 'ProductID', 'Quantity']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' is missing from the data.")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ensure Transaction ID exists, create if it doesn't
    if 'TransactionID' not in df.columns:
        # If there's a transaction identifier, use it
        for candidate in ['Transaction', 'OrderID', 'Order', 'InvoiceNo', 'Invoice']:
            if candidate in df.columns:
                df['TransactionID'] = df[candidate]
                break
        else:
            # Generate transaction IDs based on Date and row number
            # This is a heuristic for when real transaction IDs are missing
            df['TransactionID'] = df['Date'].dt.strftime('%Y%m%d') + '_' + df.groupby('Date').cumcount().astype(str)
    
    # Ensure Quantity is numeric
    df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
    
    # Handle missing values
    df = df.dropna(subset=['Date', 'ProductID', 'Quantity', 'TransactionID'])
    
    # Ensure product IDs are strings
    df['ProductID'] = df['ProductID'].astype(str)
    
    # Ensure transaction IDs are strings
    df['TransactionID'] = df['TransactionID'].astype(str)
    
    return df, file_type

def standardize_column_names(columns):
    """
    Create a mapping to standardize column names.
    
    Parameters:
    -----------
    columns : list
        List of column names from the dataframe
    
    Returns:
    --------
    dict
        Mapping from original column names to standardized names
    """
    column_mapping = {}
    
    # Define patterns to match
    date_patterns = ['date', 'order_date', 'transaction_date', 'invoice_date']
    product_patterns = ['product', 'product_id', 'productid', 'item', 'item_id', 'itemid', 'sku']
    quantity_patterns = ['quantity', 'qty', 'amount', 'units']
    transaction_patterns = ['transaction', 'transaction_id', 'transactionid', 'order', 'order_id', 
                            'orderid', 'invoice', 'invoice_no', 'invoiceno']
    
    # Check each column against patterns
    for col in columns:
        col_lower = col.lower().replace(' ', '_').replace('-', '_')
        
        if any(pattern in col_lower for pattern in date_patterns):
            column_mapping[col] = 'Date'
        elif any(pattern in col_lower for pattern in product_patterns):
            column_mapping[col] = 'ProductID'
        elif any(pattern in col_lower for pattern in quantity_patterns):
            column_mapping[col] = 'Quantity'
        elif any(pattern in col_lower for pattern in transaction_patterns):
            column_mapping[col] = 'TransactionID'
    
    return column_mapping

def validate_data(data):
    """
    Validate that the data contains required columns and correct data types
    
    Parameters:
    -----------
    data : pandas DataFrame
        The data to validate
    
    Returns:
    --------
    tuple
        (bool indicating if data is valid, error message if not valid)
    """
    try:
        # Check for required columns
        required_columns = ['Date', 'ProductID', 'Quantity']
        for col in required_columns:
            if col not in data.columns:
                return False, f"Required column '{col}' is missing from the data."
        
        # Check for minimum number of records
        if len(data) < 10:
            return False, "Data has fewer than 10 records. More data is needed for meaningful analysis."
        
        # Check for unique products
        if len(data['ProductID'].unique()) < 2:
            return False, "Data needs at least 2 unique products for association analysis."
        
        # Check for unique transactions
        if len(data['TransactionID'].unique()) < 5:
            return False, "Data needs at least 5 unique transactions for meaningful analysis."
        
        # Check for time range
        date_range = (data['Date'].max() - data['Date'].min()).days
        if date_range < 7:
            return False, "Data spans less than 7 days. More historical data is needed for forecasting."
        
        # Check data types
        if not pd.api.types.is_datetime64_any_dtype(data['Date']):
            return False, "Date column is not in a valid datetime format."
        
        if not pd.api.types.is_numeric_dtype(data['Quantity']):
            return False, "Quantity column is not numeric."
        
        # Check for negative or zero quantities
        if (data['Quantity'] <= 0).any():
            # Log a warning, but don't invalidate the data
            return True, "Warning: Some quantity values are negative or zero. These will be filtered out for analysis."
        
        return True, "Data validation successful."
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"

def aggregate_daily_sales(data):
    """
    Aggregate sales data to daily level for each product
    
    Parameters:
    -----------
    data : pandas DataFrame
        The transaction data
    
    Returns:
    --------
    pandas DataFrame
        Daily aggregated sales data
    """
    # Group by date and product, sum the quantities
    daily_sales = data.groupby(['Date', 'ProductID'])['Quantity'].sum().reset_index()
    
    return daily_sales

def create_time_features(data):
    """
    Create time-based features for forecasting
    
    Parameters:
    -----------
    data : pandas DataFrame
        The daily sales data with Date column
    
    Returns:
    --------
    pandas DataFrame
        Data with additional time features
    """
    # Make a copy to avoid modifying the original
    df = data.copy()
    
    # Extract basic time components
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['DayOfWeek'] = df['Date'].dt.dayofweek
    df['Quarter'] = df['Date'].dt.quarter
    
    # Create seasonal features
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    
    # Add holiday flag (simplified)
    # This could be expanded with a proper holiday calendar
    df['IsWeekend'] = df['DayOfWeek'].isin([5, 6]).astype(int)
    
    # Month start/end flags
    df['IsMonthStart'] = df['Date'].dt.is_month_start.astype(int)
    df['IsMonthEnd'] = df['Date'].dt.is_month_end.astype(int)
    
    return df

def prepare_forecast_features(data, association_rules=None):
    """
    Prepare features for forecasting, including association-based features
    
    Parameters:
    -----------
    data : pandas DataFrame
        The transaction data
    association_rules : pandas DataFrame, optional
        Association rules from Apriori algorithm
    
    Returns:
    --------
    pandas DataFrame
        Data with forecasting features
    """
    # Aggregate to daily sales
    daily_sales = aggregate_daily_sales(data)
    
    # Add time features
    daily_sales = create_time_features(daily_sales)
    
    # Create lag features (previous day, week, month sales)
    daily_sales_with_lags = create_lag_features(daily_sales)
    
    # Add association-based features if rules are provided
    if association_rules is not None and not association_rules.empty:
        daily_sales_with_lags = add_association_features(daily_sales_with_lags, association_rules, data)
    
    return daily_sales_with_lags

def create_lag_features(daily_sales, lags=[1, 7, 14, 30]):
    """
    Create lag features for time series forecasting
    
    Parameters:
    -----------
    daily_sales : pandas DataFrame
        The daily sales data
    lags : list, optional
        List of lag periods to create
    
    Returns:
    --------
    pandas DataFrame
        Data with lag features
    """
    # Make a copy to avoid modifying the original
    df = daily_sales.copy()
    
    # Sort by date and product
    df = df.sort_values(['ProductID', 'Date'])
    
    # Create lag features
    for lag in lags:
        df[f'Quantity_Lag{lag}'] = df.groupby('ProductID')['Quantity'].shift(lag)
    
    # Create rolling means
    df['Quantity_RollingMean7'] = df.groupby('ProductID')['Quantity'].transform(
        lambda x: x.rolling(window=7, min_periods=1).mean()
    )
    
    df['Quantity_RollingMean14'] = df.groupby('ProductID')['Quantity'].transform(
        lambda x: x.rolling(window=14, min_periods=1).mean()
    )
    
    df['Quantity_RollingMean30'] = df.groupby('ProductID')['Quantity'].transform(
        lambda x: x.rolling(window=30, min_periods=1).mean()
    )
    
    # Create rolling standard deviations (for volatility)
    df['Quantity_RollingStd7'] = df.groupby('ProductID')['Quantity'].transform(
        lambda x: x.rolling(window=7, min_periods=1).std()
    )
    
    # Fill NaN values created by shifts and rolling windows
    for col in df.columns:
        if col.startswith('Quantity_Lag') or col.startswith('Quantity_Rolling'):
            df[col] = df[col].fillna(0)
    
    return df

def add_association_features(daily_sales, association_rules, transaction_data):
    """
    Add features based on product associations
    
    Parameters:
    -----------
    daily_sales : pandas DataFrame
        The daily sales data with features
    association_rules : pandas DataFrame
        Association rules from Apriori algorithm
    transaction_data : pandas DataFrame
        Original transaction data
    
    Returns:
    --------
    pandas DataFrame
        Data with association-based features
    """
    # Make a copy to avoid modifying the original
    df = daily_sales.copy()
    
    # Create a mapping of associated products
    product_associations = {}
    
    # Process association rules to extract product relationships
    for _, row in association_rules.iterrows():
        antecedents = list(row['antecedents'])
        consequents = list(row['consequents'])
        confidence = row['confidence']
        
        # For each antecedent, store its consequents with confidence
        for antecedent in antecedents:
            if antecedent not in product_associations:
                product_associations[antecedent] = []
            
            for consequent in consequents:
                product_associations[antecedent].append((consequent, confidence))
    
    # For each product, add features based on associated product sales
    # Get unique product IDs and dates
    unique_product_ids = df['ProductID'].unique()
    unique_dates = df['Date'].unique()
    
    for product_id in unique_product_ids:
        if product_id in product_associations:
            # Get associated products
            associated_products = product_associations[product_id]
            
            # For each date
            for date in unique_dates:
                # Get associated product sales for this date
                associated_sales = 0
                associated_count = 0
                
                for associated_product, confidence in associated_products:
                    # Find sales for the associated product on this date
                    sales_value = df[(df['ProductID'] == associated_product) & 
                                     (df['Date'] == date)]['Quantity'].values
                    
                    if len(sales_value) > 0:
                        associated_sales += sales_value[0] * confidence
                        associated_count += 1
                
                # Calculate weighted average sales of associated products
                if associated_count > 0:
                    avg_associated_sales = associated_sales / associated_count
                else:
                    avg_associated_sales = 0
                
                # Add as a feature
                df.loc[(df['ProductID'] == product_id) & (df['Date'] == date), 
                       'AssociatedProductSales'] = avg_associated_sales
    
    # Fill missing values (products with no associations)
    if 'AssociatedProductSales' in df.columns:
        df['AssociatedProductSales'] = df['AssociatedProductSales'].fillna(0)
    else:
        df['AssociatedProductSales'] = 0
    
    return df
