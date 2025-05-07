import streamlit as st
import pandas as pd
import numpy as np
import io
import base64
from datetime import datetime
import os
import pickle
import time
import re

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
from utils.authentication import (
    initialize_authentication,
    register_user,
    authenticate_user,
    is_authenticated,
    login_user,
    logout_user,
    get_current_user
)
from utils.inventory_optimization import (
    get_inventory_recommendations,
    get_bundle_inventory_recommendations,
    calculate_safety_stock,
    calculate_reorder_points
)

# Page configuration
st.set_page_config(
    page_title="Demand Forecasting Based on Product Association",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize the authentication system
initialize_authentication()

# Custom CSS for enhanced UI with modern professional dark theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 98%;
    }
    
    /* Heading styles */
    h1, h2, h3 {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.2rem !important;
        color: #fff !important;
        margin-bottom: 1.5rem !important;
    }
    
    h2 {
        font-size: 1.5rem !important;
        color: #fff !important;
    }
    
    h3 {
        font-size: 1.2rem !important;
        color: #f8f9fa !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4B56D2;
        color: white;
        border-radius: 6px;
        padding: 0.6rem 1.2rem;
        font-weight: 500;
        border: none;
        margin-top: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.15);
        transition: all 0.2s ease;
    }
    
    .stButton>button:hover {
        background-color: #5D6AD2;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transform: translateY(-2px);
    }
    
    /* Alert styling */
    .stAlert {
        background-color: #1E1E1E !important;
        border-radius: 6px !important;
        border-left: 3px solid #4B56D2 !important;
        box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(75, 86, 210, 0.2);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: #1A1A1A !important;
        border-right: 1px solid rgba(75, 86, 210, 0.1);
    }
    
    /* Metrics styling */
    .css-1xarl3l, [data-testid="stMetric"] {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(75, 86, 210, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08) !important;
    }
    
    /* Input elements styling */
    .stSlider, .stSelectbox, .stFileUploader, .stMultiSelect, .stNumberInput {
        padding: 0.5rem 0;
    }
    
    .stSlider > div > div, .stSelectbox > div > div, .stMultiSelect > div > div, .stNumberInput > div > div {
        background-color: #1E1E1E !important;
        border: 1px solid rgba(75, 86, 210, 0.2) !important;
        border-radius: 6px !important;
    }
    
    /* Chart styling */
    .js-plotly-plot {
        border-radius: 8px;
        background-color: rgba(30, 30, 30, 0.7);
        padding: 1rem;
        margin: 1rem 0;
        border: 1px solid rgba(75, 86, 210, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Card effect */
    .card {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(75, 86, 210, 0.1);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.15);
        transform: translateY(-4px);
    }
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #121212;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(75, 86, 210, 0.4);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(75, 86, 210, 0.6);
    }
    
    /* File uploader */
    .stFileUploader > div {
        background-color: #1E1E1E;
        border: 1px dashed rgba(75, 86, 210, 0.3);
        border-radius: 6px;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background-color: #4B56D2 !important;
    }
    
    /* Tooltip */
    .stTooltipIcon {
        color: rgba(75, 86, 210, 0.7) !important;
    }
    
    /* Table */
    .stTable {
        border: 1px solid rgba(75, 86, 210, 0.1);
        border-radius: 8px;
    }
    
    /* Code block */
    .stCode {
        border-radius: 6px;
    }
    
    /* Link color */
    a {
        color: #6C78DD !important;
        text-decoration: none !important;
    }
    
    a:hover {
        text-decoration: underline !important;
    }
    
    /* Section divider */
    hr {
        border-color: rgba(75, 86, 210, 0.1);
        margin: 2rem 0;
    }
    
    /* Custom classes */
    .gradient-text {
        background: linear-gradient(90deg, #4B56D2, #818DFE);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .highlight-border {
        border: 1px solid #4B56D2;
        box-shadow: 0 0 8px rgba(75, 86, 210, 0.2);
    }
    
    .stats-container {
        display: flex;
        justify-content: space-between;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background-color: #1E1E1E;
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid #4B56D2;
        flex: 1;
        margin: 0 0.5rem;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* Dashboard card */
    .dashboard-card {
        background-color: #1E1E1E;
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(75, 86, 210, 0.1);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        height: 100%;
    }
    
    .dashboard-card h4 {
        color: #fff;
        font-size: 1rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    
    .dashboard-card p {
        font-size: 0.85rem;
        color: #ccc;
    }
    
    /* Section headers */
    .section-header {
        display: flex;
        align-items: center;
        margin-bottom: 1.5rem;
    }
    
    .section-header h3 {
        margin: 0;
        margin-right: 1rem;
    }
    
    .section-line {
        flex-grow: 1;
        height: 1px;
        background: linear-gradient(90deg, rgba(75, 86, 210, 0.5), transparent);
    }
</style>
""", unsafe_allow_html=True)

# Application title and description with enhanced styling
st.markdown("<h1 style='text-align: center;'>Demand Forecasting Based on <span class='gradient-text'>Product Association</span></h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size: 1rem; margin-bottom: 2rem; opacity: 0.8;'>Advanced sales analysis and prediction platform using association rules</p>", unsafe_allow_html=True)

# User Authentication Display in Sidebar
if is_authenticated():
    user = get_current_user()
    st.sidebar.markdown(f"""
    <div style="background-color: #1E1E1E; padding: 10px; border-radius: 5px; margin-bottom: 20px; border-left: 3px solid #4B56D2;">
        <p style="margin: 0; font-size: 0.9rem;">Logged in as:</p>
        <p style="margin: 0; font-weight: bold; color: #4B56D2;">{user['full_name']}</p>
        <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">{user['email']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.sidebar.button("Sign Out", key="signout"):
        logout_user()
        st.rerun()
else:
    st.sidebar.markdown("""
    <div style="background-color: #1E1E1E; padding: 10px; border-radius: 5px; margin-bottom: 20px; border-left: 3px solid #FF6B6B;">
        <p style="margin: 0; font-size: 0.9rem;">Not logged in</p>
        <p style="margin: 0; font-size: 0.8rem; opacity: 0.7;">Please sign in to access all features</p>
    </div>
    """, unsafe_allow_html=True)

# Styled sidebar navigation
st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 2rem; color: #4B56D2; font-size: 1.3rem;'>Navigation</h2>", unsafe_allow_html=True)

if is_authenticated():
    page = st.sidebar.radio(
        "Select Section",
        ["Home", "Data Upload", "Association Analysis", "Demand Forecasting", "Inventory Optimization", "Visualization", "User Profile"],
        label_visibility="collapsed"  # Hide label but maintain accessibility
    )
else:
    page = st.sidebar.radio(
        "Select Section",
        ["Home", "Login", "Register"],
        label_visibility="collapsed"  # Hide label but maintain accessibility
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
    # Create a welcome banner
    st.markdown("""
    <div class="card" style="padding: 25px; margin-bottom: 30px;">
        <h2 style="margin-top: 0; margin-bottom: 15px; color: #4B56D2;">Product Demand Analytics</h2>
        <p style="font-size: 1rem; margin: 0 0 10px 0; line-height: 1.5;">This platform combines product association analysis with demand forecasting to help you optimize inventory and identify cross-selling opportunities based on customer purchase patterns.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Dashboard layout
    st.markdown("""
    <div class="section-header">
        <h3>Dashboard</h3>
        <div class="section-line"></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create a three-column layout for sample visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample product association network visualization using local SVG
        st.markdown("""
        <div class="dashboard-card">
            <h4>Product Association Network</h4>
            <div style="height: 250px; display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
                <img src="assets/images/network_graph.svg" style="max-width: 100%; max-height: 100%; object-fit: contain;" alt="Sample Network Graph">
            </div>
            <p>Network visualization showing product relationships, with nodes representing products and edges showing co-purchase frequency.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Sample demand forecast visualization using local SVG
        st.markdown("""
        <div class="dashboard-card">
            <h4>Demand Forecast Trends</h4>
            <div style="height: 250px; display: flex; justify-content: center; align-items: center; margin-bottom: 15px;">
                <img src="assets/images/forecast_chart.svg" style="max-width: 100%; max-height: 100%; object-fit: contain;" alt="Sample Forecast Chart">
            </div>
            <p>Time series forecasts that incorporate both historical sales patterns and product associations for more accurate demand prediction.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Add sample metrics
    st.markdown("<div class='section-header' style='margin-top: 30px;'><h3>Key Metrics</h3><div class='section-line'></div></div>", unsafe_allow_html=True)
    
    # Create metrics row
    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    
    with metric_col1:
        st.metric("Average Association Strength", "0.68", "+0.12")
    
    with metric_col2:
        st.metric("Forecast Accuracy", "89.4%", "+2.3%")
    
    with metric_col3:
        st.metric("Optimal Bundle Count", "12", "+3")
    
    with metric_col4:
        st.metric("Cross-Sell Opportunities", "26", "+8")
    
    # Workflow explanation
    st.markdown("""
    <div class='section-header' style='margin-top: 30px;'>
        <h3>How It Works</h3>
        <div class='section-line'></div>
    </div>
    """, unsafe_allow_html=True)
    
    # Create workflow steps
    steps = [
        {
            "icon": "📊", 
            "title": "Data Upload", 
            "description": "Upload transaction data containing product purchases and dates"
        },
        {
            "icon": "🔗", 
            "title": "Association Discovery", 
            "description": "The system discovers which products are frequently purchased together using the Apriori algorithm"
        },
        {
            "icon": "📈", 
            "title": "Demand Prediction", 
            "description": "XGBoost models predict demand for both individual products and product bundles"
        },
        {
            "icon": "🔍", 
            "title": "Visual Analysis", 
            "description": "Explore interactive visualizations of product relationships and demand forecasts"
        }
    ]
    
    # Display steps in a horizontal layout
    cols = st.columns(4)
    for i, step in enumerate(steps):
        with cols[i]:
            st.markdown(f"""
            <div style="background-color: rgba(30, 33, 48, 0.8); padding: 15px; border-radius: 4px; height: 160px; display: flex; flex-direction: column; align-items: center; text-align: center; box-shadow: 0 2px 5px rgba(0, 0, 0, 0.2); border: 1px solid rgba(138, 84, 253, 0.1);">
                <div style="font-size: 24px; color: #8A54FD; margin-bottom: 10px;">{step['icon']}</div>
                <h4 style="margin: 0 0 10px 0; color: #8A54FD; font-weight: 600; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px;">{step['title']}</h4>
                <p style="margin: 0; font-size: 0.8rem; line-height: 1.4;">{step['description']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Call to action
    st.markdown("""
    <div style="background-color: rgba(30, 33, 48, 0.8); padding: 18px; border-radius: 4px; margin-top: 30px; text-align: center; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15); border: 1px solid rgba(138, 84, 253, 0.2);">
        <p style="margin: 0; color: rgba(138, 84, 253, 1); font-size: 16px; font-weight: 500; letter-spacing: 1px;">⟶ START BY UPLOADING YOUR DATA IN THE DATA UPLOAD SECTION</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Data requirements
    st.markdown("""
    <h3 style="margin: 30px 0 20px 0; color: #8A54FD; letter-spacing: 1.5px; text-transform: uppercase; font-size: 1.1rem;">
        <span style="border-bottom: 2px solid #8A54FD; padding-bottom: 5px;">DATA REQUIREMENTS</span>
    </h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="card">
            <h4 style="color: #8A54FD; margin-top: 0; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem;">Required Columns</h4>
            <ul style="padding-left: 25px; margin-top: 10px;">
                <li style="margin-bottom: 8px; line-height: 1.4;"><strong style="color: rgba(138, 84, 253, 0.9);">Date</strong>: Transaction date (YYYY-MM-DD)</li>
                <li style="margin-bottom: 8px; line-height: 1.4;"><strong style="color: rgba(138, 84, 253, 0.9);">Product ID</strong>: Unique identifier for each product</li>
                <li style="margin-bottom: 8px; line-height: 1.4;"><strong style="color: rgba(138, 84, 253, 0.9);">Quantity</strong>: Number of items sold</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="card">
            <h4 style="color: #8A54FD; margin-top: 0; text-transform: uppercase; letter-spacing: 1px; font-size: 0.9rem;">Optional Columns</h4>
            <ul style="padding-left: 25px; margin-top: 10px;">
                <li style="margin-bottom: 8px; line-height: 1.4;"><strong style="color: rgba(138, 84, 253, 0.7);">Transaction ID</strong>: Unique identifier for each transaction</li>
                <li style="margin-bottom: 8px; line-height: 1.4;"><strong style="color: rgba(138, 84, 253, 0.7);">Product Category</strong>: Category of the product</li>
                <li style="margin-bottom: 8px; line-height: 1.4;"><strong style="color: rgba(138, 84, 253, 0.7);">Price</strong>: Price of the product</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# Login page
elif page == "Login":
    st.markdown("<h2 style='text-align: center;'>Sign In</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 8px; border: 1px solid rgba(75, 86, 210, 0.2); margin-bottom: 20px;">
            <h4 style="text-align: center; color: #4B56D2; margin-bottom: 20px; font-size: 1.2rem;">Welcome Back</h4>
            <p style="text-align: center; font-size: 0.9rem; margin-bottom: 20px;">Sign in to access your forecasting dashboard and analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="your.email@example.com")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <a href="#" onclick="document.getElementById('register_link').click();" style="font-size: 0.8rem;">Need an account? Register</a>
                <button id="register_link" style="display: none;"></button>
                """, unsafe_allow_html=True)
            
            submit_button = st.form_submit_button(label="Sign In")
            
            if submit_button:
                if not email or not password:
                    st.error("Please enter your email and password.")
                else:
                    with st.spinner("Authenticating..."):
                        success, user = authenticate_user(email, password)
                        if success:
                            login_user(user)
                            st.success("Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid email or password. Please try again.")

# Register page
elif page == "Register":
    st.markdown("<h2 style='text-align: center;'>Create Account</h2>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 8px; border: 1px solid rgba(75, 86, 210, 0.2); margin-bottom: 20px;">
            <h4 style="text-align: center; color: #4B56D2; margin-bottom: 20px; font-size: 1.2rem;">Join Our Platform</h4>
            <p style="text-align: center; font-size: 0.9rem; margin-bottom: 20px;">Create your account to access advanced forecasting tools and analytics</p>
        </div>
        """, unsafe_allow_html=True)
        
        with st.form("register_form"):
            full_name = st.text_input("Full Name", placeholder="John Doe")
            email = st.text_input("Email Address", placeholder="your.email@example.com")
            company = st.text_input("Company Name (Optional)", placeholder="Your Company Inc.")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            confirm_password = st.text_input("Confirm Password", type="password", placeholder="••••••••")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <a href="#" onclick="document.getElementById('login_link').click();" style="font-size: 0.8rem;">Already have an account? Sign in</a>
                <button id="login_link" style="display: none;"></button>
                """, unsafe_allow_html=True)
            
            submit_button = st.form_submit_button(label="Create Account")
            
            if submit_button:
                # Validate inputs
                if not full_name or not email or not password:
                    st.error("Please fill in all required fields.")
                elif password != confirm_password:
                    st.error("Passwords do not match.")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters long.")
                elif not re.match(r"[^@]+@[^@]+\.[^@]+", email):
                    st.error("Please enter a valid email address.")
                else:
                    with st.spinner("Creating your account..."):
                        success, message = register_user(full_name, email, password, company)
                        if success:
                            st.success(message)
                            st.info("Please sign in with your new account.")
                            time.sleep(2)
                            # Redirect to login page
                            st.session_state.page = "Login"
                            st.rerun()
                        else:
                            st.error(message)

# User Profile page
elif page == "User Profile":
    if not is_authenticated():
        st.warning("You must be logged in to view this page.")
        st.stop()
    
    user = get_current_user()
    
    st.markdown("<h2 style='text-align: center;'>User Profile</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 20px; border-radius: 8px; border: 1px solid rgba(75, 86, 210, 0.2); margin-bottom: 20px;">
            <div style="font-size: 60px; color: #4B56D2; text-align: center; margin-bottom: 10px;">👤</div>
            <h3 style="text-align: center; margin-top: 0;">{user['full_name']}</h3>
            <p style="text-align: center; font-size: 0.9rem; opacity: 0.7;">{user['email']}</p>
            {f'<p style="text-align: center; font-size: 0.9rem; color: #4B56D2;">{user["company"]}</p>' if user.get('company') else ''}
            <p style="text-align: center; font-size: 0.8rem; opacity: 0.5;">Joined: {datetime.fromisoformat(user['created_at']).strftime("%B %d, %Y")}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Update profile section
        st.subheader("Update Profile")
        
        with st.form("update_profile_form"):
            new_name = st.text_input("Full Name", value=user['full_name'])
            new_company = st.text_input("Company Name", value=user.get('company', ''))
            
            update_profile_button = st.form_submit_button(label="Update Profile")
            
            if update_profile_button:
                success, message = update_profile(user['id'], full_name=new_name, company=new_company)
                if success:
                    st.success(message)
                    # Force refresh of session state
                    success, updated_user = authenticate_user(user['email'], password=user.get('password_hash'))
                    if success:
                        login_user(updated_user)
                        st.rerun()
                else:
                    st.error(message)
        
        # Change password section
        st.subheader("Change Password")
        
        with st.form("change_password_form"):
            current_password = st.text_input("Current Password", type="password")
            new_password = st.text_input("New Password", type="password")
            confirm_new_password = st.text_input("Confirm New Password", type="password")
            
            change_password_button = st.form_submit_button(label="Change Password")
            
            if change_password_button:
                if not current_password or not new_password or not confirm_new_password:
                    st.error("Please fill in all password fields.")
                elif new_password != confirm_new_password:
                    st.error("New passwords do not match.")
                elif len(new_password) < 6:
                    st.error("New password must be at least 6 characters long.")
                else:
                    success, message = change_password(user['id'], current_password, new_password)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)

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
        <div style="background-color: rgba(124, 77, 255, 0.05); padding: 20px; border-radius: 10px; border-left: 4px solid #7C4DFF; margin-bottom: 20px;">
            <h3 style="color: #7C4DFF; margin-top: 0;">Product Association Mining</h3>
            <p>This section uses the Apriori algorithm to discover associations between products. 
            These associations represent products that are frequently purchased together, helping you identify 
            cross-selling opportunities and optimize product bundling strategies.</p>
        </div>
        """, unsafe_allow_html=True)
        
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

# Inventory Optimization page
elif page == "Inventory Optimization":
    st.header("Inventory Optimization")
    
    if not is_authenticated():
        st.warning("You must be logged in to access this feature.")
        st.stop()
    
    if st.session_state.preprocessed_data is None:
        st.warning("No data available. Please upload data first.")
    elif st.session_state.predictions is None:
        st.warning("Please run demand forecasting first to generate inventory recommendations.")
    else:
        st.markdown("""
        <div style="background-color: rgba(124, 77, 255, 0.05); padding: 20px; border-radius: 10px; border-left: 4px solid #7C4DFF; margin-bottom: 20px;">
            <h3 style="color: #7C4DFF; margin-top: 0;">Inventory Optimization</h3>
            <p>This section uses forecasted demand and historical patterns to recommend optimal inventory levels, 
            reorder points, and safety stock to minimize stockouts while avoiding excess inventory.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Parameters for inventory calculations
        col1, col2, col3 = st.columns(3)
        
        with col1:
            lead_time = st.slider(
                "Lead Time (Days)", 
                min_value=1, 
                max_value=30, 
                value=7,
                help="Number of days between placing an order and receiving it"
            )
        
        with col2:
            service_level = st.select_slider(
                "Service Level", 
                options=[90, 95, 98, 99],
                value=95,
                help="Target level of service - higher means less stockouts but more inventory"
            )
        
        with col3:
            holding_cost = st.slider(
                "Annual Holding Cost (%)", 
                min_value=10, 
                max_value=50, 
                value=25,
                help="Annual cost of holding inventory as a percentage of item value"
            )
        
        # Product selection for optimization
        st.subheader("Select Products for Optimization")
        
        all_products = st.session_state.product_list
        default_selection = all_products[:min(5, len(all_products))]
        
        selected_products = st.multiselect(
            "Choose products to optimize",
            options=all_products,
            default=default_selection,
            help="Select specific products for inventory optimization"
        )
        
        if not selected_products:
            st.warning("Please select at least one product to optimize.")
            st.stop()
        
        if st.button("Calculate Inventory Recommendations"):
            with st.spinner("Generating inventory recommendations..."):
                # Filter data for selected products
                historical_data = st.session_state.preprocessed_data[
                    st.session_state.preprocessed_data['ProductID'].isin(selected_products)
                ]
                
                forecast_data = st.session_state.predictions[
                    st.session_state.predictions['ProductID'].isin(selected_products)
                ]
                
                # Calculate recommendations
                inventory_recommendations = get_inventory_recommendations(
                    historical_data,
                    forecast_data,
                    lead_time_days=lead_time,
                    service_level=service_level/100
                )
                
                # Display inventory recommendations
                st.subheader("Individual Product Recommendations")
                
                # Add some visual improvements to the dataframe
                def highlight_reorder_point(val):
                    """Highlight reorder point values"""
                    return f'background-color: rgba(124, 77, 255, 0.1)' if val.name == 'reorder_point' else ''
                
                styled_recommendations = inventory_recommendations.style.apply(highlight_reorder_point)
                
                # Round numeric columns
                numeric_cols = ['avg_daily_demand', 'demand_std_dev', 'safety_stock', 
                                'lead_time_demand', 'reorder_point', 'eoq', 'days_of_supply']
                
                for col in numeric_cols:
                    if col in inventory_recommendations.columns:
                        inventory_recommendations[col] = inventory_recommendations[col].round(2)
                
                st.dataframe(inventory_recommendations)
                
                # Calculate bundle recommendations if association rules exist
                if st.session_state.association_rules is not None and not st.session_state.association_rules.empty:
                    st.subheader("Product Bundle Recommendations")
                    
                    # Get product bundles from association rules
                    rules = st.session_state.association_rules
                    if 'antecedents' in rules.columns and 'consequents' in rules.columns:
                        # Extract product bundles
                        product_bundles = []
                        for _, row in rules.iterrows():
                            antecedents = list(row['antecedents'])
                            consequents = list(row['consequents'])
                            bundle_items = antecedents + consequents
                            confidence = row['confidence']
                            lift = row['lift']
                            
                            if len(bundle_items) >= 2:  # Only consider bundles with at least 2 items
                                product_bundles.append((bundle_items, confidence, lift))
                        
                        # Get bundle recommendations
                        bundle_recommendations = get_bundle_inventory_recommendations(
                            historical_data,
                            forecast_data,
                            product_bundles,
                            lead_time_days=lead_time,
                            service_level=service_level/100
                        )
                        
                        if not bundle_recommendations.empty:
                            # Round numeric columns
                            numeric_cols = ['confidence', 'lift', 'avg_safety_stock', 
                                          'avg_reorder_point', 'min_days_supply', 'suggested_bundle_stock']
                            
                            for col in numeric_cols:
                                if col in bundle_recommendations.columns:
                                    bundle_recommendations[col] = bundle_recommendations[col].round(2)
                            
                            st.dataframe(bundle_recommendations)
                        else:
                            st.info("No significant product bundles found for inventory optimization.")
                
                # Show inventory insights
                st.subheader("Inventory Insights")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("""
                    <div class="card">
                        <h4 style="color: #7C4DFF; margin-top: 0;">Reorder Points</h4>
                        <p>Products should be reordered when inventory reaches these levels to prevent stockouts during lead time.</p>
                        <ul>
                            <li>Higher service levels require higher reorder points</li>
                            <li>Products with variable demand need more safety stock</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div class="card">
                        <h4 style="color: #7C4DFF; margin-top: 0;">Economic Order Quantities</h4>
                        <p>These are the optimal order quantities that balance ordering costs against inventory holding costs.</p>
                        <ul>
                            <li>Frequent small orders = higher ordering costs</li>
                            <li>Large infrequent orders = higher holding costs</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Data visualization for inventory
                st.subheader("Inventory Visualization")
                
                # Create a bar chart of reorder points
                if not inventory_recommendations.empty:
                    # Get top 10 products by reorder point
                    top_products = inventory_recommendations.nlargest(10, 'reorder_point')
                    
                    # Create a horizontal bar chart
                    import plotly.express as px
                    
                    fig = px.bar(
                        top_products,
                        x='reorder_point',
                        y='ProductID',
                        orientation='h',
                        labels={'reorder_point': 'Reorder Point', 'ProductID': 'Product'},
                        title='Top Products by Reorder Point',
                        color='reorder_point',
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(
                        height=500,
                        width=800,
                        xaxis_title="Quantity",
                        yaxis_title="Product ID",
                        margin=dict(l=50, r=50, t=80, b=50)
                    )
                    
                    st.plotly_chart(fig)

# Demand Forecasting page
elif page == "Demand Forecasting":
    st.header("Demand Forecasting")
    
    if st.session_state.preprocessed_data is None:
        st.warning("No data available. Please upload data first.")
    else:
        st.markdown("""
        <div style="background-color: rgba(124, 77, 255, 0.05); padding: 20px; border-radius: 10px; border-left: 4px solid #7C4DFF; margin-bottom: 20px;">
            <h3 style="color: #7C4DFF; margin-top: 0;">Advanced Demand Forecasting</h3>
            <p>This section uses XGBoost machine learning to forecast product demand based on historical sales patterns 
            and product associations. The model takes into account both individual product demand trends 
            and the influence of associated products, providing more accurate forecasts for inventory planning.</p>
        </div>
        """, unsafe_allow_html=True)
        
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
