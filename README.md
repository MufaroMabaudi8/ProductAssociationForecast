# Demand Forecasting Based on Product Association

A sophisticated demand forecasting application that transforms complex inventory and sales data into actionable insights through advanced machine learning techniques.

## Features

- **User Authentication**: Secure login and registration system
- **Data Management**: Upload and process CSV/Excel sales data
- **Product Association Analysis**: Discover product relationships using the Apriori algorithm
- **Demand Forecasting**: Predict future demand with XGBoost machine learning
- **Inventory Optimization**: Generate inventory recommendations based on forecasts
- **Interactive Visualization**: Beautiful, interactive charts and dashboards

## Requirements

- Python 3.8 or higher
- Required Python packages (install using `pip install -r requirements.txt`):
  - streamlit
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - mlxtend
  - matplotlib
  - plotly
  - networkx
  - pyvis

## Getting Started

### Running the Application Locally

#### Windows Users:
1. Double-click the `run.bat` file
2. Open your browser and navigate to: http://localhost:8501

#### Mac/Linux Users:
1. Open Terminal in the project directory
2. Run: `./run.sh`
3. Open your browser and navigate to: http://localhost:8501

#### Alternative Method (All Platforms):
1. Open a command prompt or terminal in the project directory
2. Run: `python run.py`
3. Open your browser and navigate to: http://localhost:8501

### Manual Setup
If the quick start scripts don't work, you can run the application manually:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py --server.port 8501 --server.address localhost
```

## Data Format

The application expects transaction data in CSV or Excel format with the following columns:
- `TransactionID`: Unique identifier for each transaction
- `Date`: Transaction date (in a standard date format)
- `ProductID`: Unique identifier for each product
- `Quantity`: Number of items purchased
- `Price`: Unit price of the product (optional)

## Developer Information

© 2025 Mufaro Mabaudi - All Rights Reserved