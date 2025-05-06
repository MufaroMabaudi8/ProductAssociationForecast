import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import os
import pickle
import time

# Import utility modules
from utils.data_processing import load_and_preprocess_data, validate_data
from utils.association_analysis import perform_association_analysis
from utils.forecasting import train_forecasting_model, predict_demand
from utils.visualization import (
    plot_association_network, 
    plot_forecasting_results,
    plot_product_associations_heatmap,
    plot_product_sales_trend,
    plot_top_rules_table
)

# Page configuration
st.set_page_config(
    page_title="Demand Forecasting Based on Product Association",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Application title and description
st.title("📊 Demand Forecasting Based on Product Association")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Home", "Data Upload", "Association Analysis", "Demand Forecasting", "Visualization", "Reports"]
)

# Initialize session state for storing data and models
if "data" not in st.session_state:
    st.session_state.data = None
if "preprocessed_data" not in st.session_state:
    st.session_state.preprocessed_data = None
if "association_rules" not in st.session_state:
    st.session_state.association_rules = None
if "forecasting_model" not in st.session_state:
    st.session_state.forecasting_model = None
if "predictions" not in st.session_state:
    st.session_state.predictions = None
if "frequent_itemsets" not in st.session_state:
    st.session_state.frequent_itemsets = None
if "transaction_data" not in st.session_state:
    st.session_state.transaction_data = None
if "product_list" not in st.session_state:
    st.session_state.product_list = None

# Home page
if page == "Home":
    st.markdown("""
    ## Welcome to the Demand Forecasting Platform
    
    This application helps you analyze product associations and forecast demand based on historical sales data.
    
    ### Core Features:
    
    - **Product Association Analysis**: Discover which products are frequently purchased together using the Apriori algorithm
    - **Demand Forecasting**: Predict future product demand with XGBoost machine learning models
    - **Interactive Visualizations**: Explore associations with force-directed graphs and other visual tools
    - **Downloadable Reports**: Export your analysis results for business planning
    
    ### How to Use:
    
    1. Upload your transaction data in CSV or Excel format
    2. Run association analysis to discover product relationships
    3. Generate demand forecasts based on historical patterns
    4. Explore visualizations to gain insights
    5. Download reports for your business planning
    
    ### Data Format Requirements:
    
    Your data should include at least the following columns:
    - **Date**: Transaction date (YYYY-MM-DD)
    - **Product ID**: Unique identifier for each product
    - **Quantity**: Number of items sold
    
    Other useful columns:
    - **Transaction ID**: Unique identifier for each transaction
    - **Product Category**: Category of the product
    - **Price**: Price of the product
    
    """)
    
    st.info("Start by uploading your data in the 'Data Upload' section from the sidebar.")

# Data Upload page
elif page == "Data Upload":
    st.header("Data Upload")
    
    # Sample data download
    st.subheader("Sample Data Format")
    st.markdown("""
    Download the sample CSV file to see the required format. Your data should include 
    at minimum: transaction date, product ID, and quantity.
    """)
    
    sample_data_path = "assets/sample_data_format.csv"
    st.download_button(
        label="Download Sample Data Format",
        data=open(sample_data_path, "r").read(),
        file_name="sample_data_format.csv",
        mime="text/csv"
    )
    
    # Data upload
    st.subheader("Upload Your Data")
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Loading progress indicator
            with st.spinner("Loading and validating data..."):
                # Load the data
                data, file_type = load_and_preprocess_data(uploaded_file)
                
                # Validate the data format
                is_valid, message = validate_data(data)
                
                if is_valid:
                    st.session_state.data = data
                    st.success(f"Successfully loaded data with {len(data)} records.")
                    
                    # Display data preview
                    st.subheader("Data Preview")
                    st.dataframe(data.head(10))
                    
                    # Display data statistics
                    st.subheader("Data Statistics")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Transactions", len(data['TransactionID'].unique()))
                    
                    with col2:
                        st.metric("Unique Products", len(data['ProductID'].unique()))
                    
                    with col3:
                        st.metric("Date Range", f"{data['Date'].min().date()} to {data['Date'].max().date()}")
                    
                    # Process for association analysis
                    with st.spinner("Preparing data for analysis..."):
                        # Create transaction data for association analysis
                        transaction_data = data.groupby(['TransactionID', 'ProductID'])['Quantity'].sum().reset_index()
                        transaction_data = transaction_data[transaction_data['Quantity'] > 0]
                        
                        # Store product list for later use
                        product_list = data['ProductID'].unique()
                        
                        # Store processed data in session state
                        st.session_state.preprocessed_data = data
                        st.session_state.transaction_data = transaction_data
                        st.session_state.product_list = product_list
                        
                        # Clear any existing results when new data is uploaded
                        st.session_state.association_rules = None
                        st.session_state.forecasting_model = None
                        st.session_state.predictions = None
                        st.session_state.frequent_itemsets = None
                        
                        st.success("Data is ready for analysis. Please proceed to Association Analysis or Demand Forecasting.")
                else:
                    st.error(f"Data validation failed: {message}")
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
    else:
        st.info("Please upload a file to begin analysis.")

# Association Analysis page
elif page == "Association Analysis":
    st.header("Product Association Analysis")
    
    if st.session_state.preprocessed_data is None:
        st.warning("No data available. Please upload data first.")
    else:
        st.markdown("""
        This section uses the Apriori algorithm to discover associations between products. 
        These associations represent products that are frequently purchased together.
        """)
        
        st.subheader("Association Analysis Parameters")
        
        col1, col2 = st.columns(2)
        
        with col1:
            min_support = st.slider(
                "Minimum Support", 
                min_value=0.001, 
                max_value=0.5, 
                value=0.01, 
                step=0.001,
                help="Minimum support threshold for itemsets (how frequently items appear together)"
            )
        
        with col2:
            min_confidence = st.slider(
                "Minimum Confidence", 
                min_value=0.1, 
                max_value=1.0, 
                value=0.5, 
                step=0.05,
                help="Minimum confidence threshold for rules (how reliable the association is)"
            )
        
        if st.button("Run Association Analysis"):
            with st.spinner("Performing association analysis..."):
                # Get transaction data
                transaction_data = st.session_state.transaction_data
                
                # Perform association analysis
                frequent_itemsets, association_rules = perform_association_analysis(
                    transaction_data,
                    min_support=min_support,
                    min_confidence=min_confidence
                )
                
                # Store results in session state
                st.session_state.frequent_itemsets = frequent_itemsets
                st.session_state.association_rules = association_rules
                
                st.success(f"Association analysis complete. Found {len(association_rules)} rules.")
        
        # Display association results if available
        if st.session_state.association_rules is not None:
            st.subheader("Association Rules")
            
            # Display top rules in a table
            st.markdown("### Top Association Rules by Lift")
            rules_df = st.session_state.association_rules
            
            if len(rules_df) > 0:
                plot_top_rules_table(rules_df)
                
                # Plot association network
                st.subheader("Product Association Network")
                st.markdown("This network graph shows how products are associated with each other:")
                
                # Create a network graph of product associations
                html_network = plot_association_network(rules_df)
                st.components.v1.html(html_network, height=600)
                
                # Download network HTML
                network_download = st.download_button(
                    label="Download Network Graph (HTML)",
                    data=html_network,
                    file_name=f"product_association_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                    mime="text/html"
                )
            else:
                st.warning("No association rules found with the current parameters. Try reducing minimum support or confidence.")

# Demand Forecasting page
elif page == "Demand Forecasting":
    st.header("Demand Forecasting")
    
    if st.session_state.preprocessed_data is None:
        st.warning("No data available. Please upload data first.")
    else:
        st.markdown("""
        This section uses XGBoost to forecast product demand based on historical sales patterns 
        and product associations. The model takes into account both individual product demand 
        and the influence of associated products.
        """)
        
        st.subheader("Forecasting Parameters")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            forecast_horizon = st.number_input(
                "Forecast Horizon (days)", 
                min_value=1,
                max_value=365,
                value=30,
                help="Number of days to forecast into the future"
            )
        
        with col2:
            train_ratio = st.slider(
                "Training Data Ratio", 
                min_value=0.5, 
                max_value=0.9, 
                value=0.8, 
                step=0.05,
                help="Proportion of data to use for training (rest for validation)"
            )
        
        with col3:
            use_associations = st.checkbox(
                "Use Product Associations", 
                value=True,
                help="Include product association features in the forecasting model"
            )
        
        # Product selection
        if st.session_state.product_list is not None:
            products_to_forecast = st.multiselect(
                "Select Products to Forecast",
                options=st.session_state.product_list,
                default=st.session_state.product_list[:5] if len(st.session_state.product_list) > 5 else st.session_state.product_list,
                help="Select products for which you want to generate forecasts"
            )
        else:
            products_to_forecast = []
        
        if st.button("Run Forecasting Model"):
            if len(products_to_forecast) == 0:
                st.warning("Please select at least one product to forecast.")
            else:
                with st.spinner("Training forecasting model and generating predictions..."):
                    # Get data
                    data = st.session_state.preprocessed_data
                    association_rules = st.session_state.association_rules
                    
                    # Train the model
                    model = train_forecasting_model(
                        data,
                        products_to_forecast,
                        association_rules if use_associations else None,
                        train_ratio=train_ratio
                    )
                    
                    # Generate predictions
                    predictions = predict_demand(
                        model,
                        data,
                        products_to_forecast,
                        horizon=forecast_horizon
                    )
                    
                    # Store in session state
                    st.session_state.forecasting_model = model
                    st.session_state.predictions = predictions
                    
                    st.success(f"Forecasting complete. Generated predictions for {len(products_to_forecast)} products over {forecast_horizon} days.")
        
        # Display forecasting results if available
        if st.session_state.predictions is not None:
            st.subheader("Demand Forecast Results")
            
            # Plot forecasting results
            plot_forecasting_results(
                st.session_state.preprocessed_data,
                st.session_state.predictions,
                products_to_forecast
            )
            
            # Feature importance
            if st.session_state.forecasting_model:
                st.subheader("Feature Importance")
                model = st.session_state.forecasting_model
                
                # Display feature importance if model has feature_importances_ attribute
                if hasattr(model, 'feature_importances_'):
                    feature_importance = pd.DataFrame({
                        'Feature': model.feature_names_in_,
                        'Importance': model.feature_importances_
                    }).sort_values('Importance', ascending=False)
                    
                    st.bar_chart(feature_importance.set_index('Feature'))

# Visualization page
elif page == "Visualization":
    st.header("Visualization Dashboard")
    
    if st.session_state.preprocessed_data is None:
        st.warning("No data available. Please upload data first.")
    else:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Product Sales Trends", "Association Heatmap", "Forecast Comparison"])
        
        with tab1:
            st.subheader("Product Sales Trends")
            
            # Get data
            data = st.session_state.preprocessed_data
            
            # Product selection for trends
            if st.session_state.product_list is not None:
                selected_products = st.multiselect(
                    "Select Products to Visualize",
                    options=st.session_state.product_list,
                    default=st.session_state.product_list[:5] if len(st.session_state.product_list) > 5 else st.session_state.product_list
                )
                
                if selected_products:
                    # Time aggregation option
                    time_agg = st.selectbox(
                        "Time Aggregation",
                        options=["Day", "Week", "Month"],
                        index=1
                    )
                    
                    # Plot product sales trends
                    plot_product_sales_trend(data, selected_products, time_aggregation=time_agg.lower())
                else:
                    st.info("Please select products to visualize trends.")
        
        with tab2:
            st.subheader("Product Association Heatmap")
            
            if st.session_state.association_rules is not None:
                # Get top products by frequency
                data = st.session_state.preprocessed_data
                top_n = st.slider("Number of Top Products", 5, 30, 15)
                
                # Get top products by sales frequency
                product_counts = data.groupby('ProductID')['Quantity'].sum().nlargest(top_n)
                top_products = product_counts.index.tolist()
                
                # Plot association heatmap
                plot_product_associations_heatmap(
                    st.session_state.association_rules,
                    top_products
                )
            else:
                st.info("Run association analysis first to view the heatmap.")
        
        with tab3:
            st.subheader("Forecast vs Actual Comparison")
            
            if st.session_state.predictions is not None and st.session_state.forecasting_model is not None:
                # Get data
                data = st.session_state.preprocessed_data
                predictions = st.session_state.predictions
                
                # Product selection for comparison
                if st.session_state.product_list is not None:
                    comparison_product = st.selectbox(
                        "Select Product to Compare",
                        options=predictions['ProductID'].unique()
                    )
                    
                    if comparison_product:
                        # Filter predictions for selected product
                        prod_predictions = predictions[predictions['ProductID'] == comparison_product]
                        
                        # Get actual data for the product
                        actual_data = data[data['ProductID'] == comparison_product]
                        
                        # Plot comparison
                        st.subheader(f"Forecast vs Actual for Product {comparison_product}")
                        
                        # Prepare data for plotting
                        actual_daily = actual_data.groupby('Date')['Quantity'].sum().reset_index()
                        
                        # Create Plotly figure
                        import plotly.graph_objects as go
                        from plotly.subplots import make_subplots
                        
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        # Add actual values
                        fig.add_trace(
                            go.Scatter(
                                x=actual_daily['Date'],
                                y=actual_daily['Quantity'],
                                mode='lines',
                                name='Actual Sales',
                                line=dict(color='blue')
                            )
                        )
                        
                        # Add predicted values
                        forecast_dates = pd.date_range(
                            start=data['Date'].max() + pd.Timedelta(days=1),
                            periods=len(prod_predictions),
                            freq='D'
                        )
                        
                        fig.add_trace(
                            go.Scatter(
                                x=forecast_dates,
                                y=prod_predictions['Predicted_Quantity'].values,
                                mode='lines',
                                name='Forecasted Sales',
                                line=dict(color='red', dash='dash')
                            )
                        )
                        
                        # Update layout
                        fig.update_layout(
                            title=f"Sales Forecast for Product {comparison_product}",
                            xaxis_title="Date",
                            yaxis_title="Quantity",
                            hovermode="x unified",
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Display forecast statistics
                        st.subheader("Forecast Statistics")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            avg_pred = prod_predictions['Predicted_Quantity'].mean()
                            st.metric("Average Predicted Demand", f"{avg_pred:.2f}")
                        
                        with col2:
                            max_pred = prod_predictions['Predicted_Quantity'].max()
                            st.metric("Maximum Predicted Demand", f"{max_pred:.2f}")
                        
                        with col3:
                            total_pred = prod_predictions['Predicted_Quantity'].sum()
                            st.metric("Total Predicted Demand", f"{total_pred:.2f}")
                            
            else:
                st.info("Run forecasting first to compare predictions with actual values.")

# Reports page
elif page == "Reports":
    st.header("Reports & Downloads")
    
    if st.session_state.preprocessed_data is None:
        st.warning("No data available. Please upload data first.")
    else:
        st.markdown("""
        Generate and download reports based on your analysis results. These reports 
        provide insights that can help with inventory management and business planning.
        """)
        
        # Association Rules Report
        st.subheader("Association Rules Report")
        
        if st.session_state.association_rules is not None:
            rules_df = st.session_state.association_rules
            
            # Report configuration
            min_lift = st.slider(
                "Minimum Lift Value", 
                min_value=1.0, 
                max_value=float(rules_df['lift'].max()) if len(rules_df) > 0 else 10.0, 
                value=1.2, 
                step=0.1
            )
            
            # Filter rules by lift
            filtered_rules = rules_df[rules_df['lift'] >= min_lift]
            
            if len(filtered_rules) > 0:
                st.write(f"Found {len(filtered_rules)} rules with lift >= {min_lift}")
                st.dataframe(filtered_rules)
                
                # Download as CSV
                csv = filtered_rules.to_csv(index=False)
                st.download_button(
                    label="Download Association Rules CSV",
                    data=csv,
                    file_name=f"association_rules_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.warning(f"No rules found with lift >= {min_lift}. Try lowering the minimum lift value.")
        else:
            st.info("Run association analysis first to generate association rules report.")
        
        # Demand Forecast Report
        st.subheader("Demand Forecast Report")
        
        if st.session_state.predictions is not None:
            predictions_df = st.session_state.predictions
            
            # Allow selection of specific products
            selected_products_forecast = st.multiselect(
                "Select Products for Forecast Report",
                options=predictions_df['ProductID'].unique(),
                default=predictions_df['ProductID'].unique()[:5] if len(predictions_df['ProductID'].unique()) > 5 else predictions_df['ProductID'].unique()
            )
            
            if selected_products_forecast:
                # Filter predictions for selected products
                selected_predictions = predictions_df[predictions_df['ProductID'].isin(selected_products_forecast)]
                
                # Add date information
                last_date = st.session_state.preprocessed_data['Date'].max()
                forecast_dates = pd.date_range(
                    start=last_date + pd.Timedelta(days=1),
                    periods=len(selected_predictions) // len(selected_products_forecast),
                    freq='D'
                )
                
                # Create a date mapping
                date_mapping = {}
                for i, date in enumerate(forecast_dates):
                    date_mapping[i] = date
                
                # Add date to predictions
                selected_predictions['ForecastDay'] = selected_predictions.groupby('ProductID').cumcount()
                selected_predictions['ForecastDate'] = selected_predictions['ForecastDay'].map(date_mapping)
                
                # Display predictions
                st.dataframe(selected_predictions[['ProductID', 'ForecastDate', 'Predicted_Quantity']])
                
                # Download as CSV
                forecast_csv = selected_predictions[['ProductID', 'ForecastDate', 'Predicted_Quantity']].to_csv(index=False)
                st.download_button(
                    label="Download Forecast Report CSV",
                    data=forecast_csv,
                    file_name=f"demand_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
                # Generate summary report
                st.subheader("Forecast Summary Report")
                
                # Calculate forecast summary
                forecast_summary = selected_predictions.groupby('ProductID')['Predicted_Quantity'].agg([
                    ('total', 'sum'),
                    ('average', 'mean'),
                    ('min', 'min'),
                    ('max', 'max')
                ]).reset_index()
                
                st.dataframe(forecast_summary)
                
                # Download summary
                summary_csv = forecast_summary.to_csv(index=False)
                st.download_button(
                    label="Download Forecast Summary CSV",
                    data=summary_csv,
                    file_name=f"forecast_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("Please select products for the forecast report.")
        else:
            st.info("Run forecasting first to generate demand forecast report.")
        
        # Combined Insights Report
        st.subheader("Combined Insights Report")
        
        if st.session_state.association_rules is not None and st.session_state.predictions is not None:
            st.markdown("""
            This report combines association analysis with demand forecasting to provide insights 
            on which product combinations will likely see increased demand together.
            """)
            
            # Generate insights
            with st.spinner("Generating combined insights..."):
                # Get association rules and predictions
                rules_df = st.session_state.association_rules
                predictions_df = st.session_state.predictions
                
                # Get top predicted products
                top_predicted = predictions_df.groupby('ProductID')['Predicted_Quantity'].sum().nlargest(10)
                
                # Find associations for top predicted products
                insights = []
                
                for product in top_predicted.index:
                    # Find rules where this product is an antecedent
                    product_as_antecedent = rules_df[rules_df['antecedents'].apply(lambda x: product in x)]
                    
                    if len(product_as_antecedent) > 0:
                        # Sort by lift
                        top_associations = product_as_antecedent.sort_values('lift', ascending=False).head(3)
                        
                        for _, row in top_associations.iterrows():
                            # Get consequent products
                            consequents = list(row['consequents'])
                            
                            # Get predicted demand for antecedent
                            antecedent_demand = top_predicted.loc[product] if product in top_predicted else 0
                            
                            # Get predicted demand for consequents if available
                            consequent_demands = []
                            for cons_prod in consequents:
                                if cons_prod in top_predicted:
                                    consequent_demands.append((cons_prod, top_predicted.loc[cons_prod]))
                            
                            # Create insight
                            insight = {
                                'Main Product': product,
                                'Main Product Forecast': antecedent_demand,
                                'Associated Products': consequents,
                                'Association Confidence': row['confidence'],
                                'Association Lift': row['lift']
                            }
                            
                            insights.append(insight)
                
                if insights:
                    # Convert insights to DataFrame
                    insights_df = pd.DataFrame(insights)
                    
                    # Display insights
                    st.dataframe(insights_df)
                    
                    # Download insights
                    insights_csv = insights_df.to_csv(index=False)
                    st.download_button(
                        label="Download Combined Insights CSV",
                        data=insights_csv,
                        file_name=f"combined_insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                    # Display actionable recommendations
                    st.subheader("Actionable Recommendations")
                    
                    for i, insight in enumerate(insights[:5]):  # Display top 5 recommendations
                        st.markdown(f"""
                        **Recommendation {i+1}:**
                        
                        Product **{insight['Main Product']}** is forecasted to have high demand 
                        ({insight['Main Product Forecast']:.2f} units). It's frequently purchased with 
                        {', '.join([str(p) for p in insight['Associated Products']])}.
                        
                        **Action:** Consider bundling these products or ensuring sufficient inventory 
                        for the associated products as well. The association has a confidence of 
                        {insight['Association Confidence']:.2f} and a lift of {insight['Association Lift']:.2f}.
                        """)
                else:
                    st.info("No significant combined insights found with current data.")
        else:
            st.info("Run both association analysis and forecasting to generate combined insights.")

        # Inventory Planning Report
        st.subheader("Inventory Planning Report")
        
        if st.session_state.predictions is not None:
            st.markdown("""
            Use this report to plan your inventory based on demand forecasts.
            """)
            
            # Allow input of lead time and safety stock factor
            col1, col2 = st.columns(2)
            
            with col1:
                lead_time = st.number_input(
                    "Lead Time (days)",
                    min_value=1,
                    max_value=30,
                    value=7,
                    help="Number of days it takes to replenish inventory"
                )
            
            with col2:
                safety_factor = st.number_input(
                    "Safety Stock Factor",
                    min_value=0.1,
                    max_value=3.0,
                    value=1.5,
                    step=0.1,
                    help="Multiplier for safety stock calculation (higher means more buffer)"
                )
            
            # Generate inventory planning
            if st.button("Generate Inventory Plan"):
                predictions_df = st.session_state.predictions
                
                # Calculate inventory requirements
                inventory_plan = []
                
                for product in predictions_df['ProductID'].unique():
                    # Get predictions for this product
                    product_predictions = predictions_df[predictions_df['ProductID'] == product]
                    
                    # Calculate average daily demand
                    avg_demand = product_predictions['Predicted_Quantity'].mean()
                    
                    # Calculate max daily demand
                    max_demand = product_predictions['Predicted_Quantity'].max()
                    
                    # Calculate safety stock
                    safety_stock = (max_demand - avg_demand) * safety_factor if max_demand > avg_demand else avg_demand * 0.2
                    
                    # Calculate reorder point
                    reorder_point = avg_demand * lead_time + safety_stock
                    
                    # Calculate order quantity (2 weeks of average demand)
                    order_quantity = avg_demand * 14
                    
                    inventory_plan.append({
                        'ProductID': product,
                        'Average Daily Demand': avg_demand,
                        'Maximum Daily Demand': max_demand,
                        'Safety Stock': safety_stock,
                        'Reorder Point': reorder_point,
                        'Suggested Order Quantity': order_quantity
                    })
                
                # Convert to DataFrame
                inventory_df = pd.DataFrame(inventory_plan)
                
                # Display inventory plan
                st.dataframe(inventory_df)
                
                # Download inventory plan
                inventory_csv = inventory_df.to_csv(index=False)
                st.download_button(
                    label="Download Inventory Planning CSV",
                    data=inventory_csv,
                    file_name=f"inventory_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        else:
            st.info("Run forecasting first to generate inventory planning report.")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center">
        <p>Demand Forecasting Application | Created with Streamlit</p>
    </div>
    """, 
    unsafe_allow_html=True
)
