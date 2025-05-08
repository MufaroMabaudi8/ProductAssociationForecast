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

### Running the Application in 100% Offline Mode

The application is designed to run completely offline without any external dependencies or internet connection.

#### Windows Users:
1. Double-click the `run.bat` file
2. The application will automatically:
   - Find an available port on your system
   - Start the Streamlit server in offline mode
   - Open your default browser to the application
3. If your browser doesn't open automatically, you can manually open:
   - http://127.0.0.1:8501 (or the port displayed in the console)

#### Mac/Linux Users:
1. Open Terminal in the project directory
2. Make the script executable (first time only): `chmod +x run.sh`
3. Run: `./run.sh`
4. The application will automatically start and attempt to open in your browser
5. If your browser doesn't open automatically, open the URL shown in the terminal

#### Alternative Method (All Platforms):
1. Open a command prompt or terminal in the project directory
2. Run: `python run.py` (or `python3 run.py` on Mac/Linux)
3. Follow the instructions displayed in the console

### Manual Setup
If the quick start scripts don't work, try this manual approach:

```bash
# Install dependencies (only needed once)
pip install -r requirements_local.txt

# Run the application in offline mode
streamlit run app.py --server.port 8501 --server.address 127.0.0.1 --server.headless true --server.enableCORS false --server.enableXsrfProtection false
```

### Troubleshooting Connection Issues

If you see "Hmmm… can't reach this page" or similar errors:

1. **Wrong Address**: Don't use `0.0.0.0` in your browser - always use either:
   - `127.0.0.1` (recommended) or
   - `localhost`

2. **Port Already in Use**:
   - Try a different port: `--server.port 8502` (or any available port)
   - Close other applications that might be using the port

3. **Firewall Issues**:
   - Ensure your firewall allows local connections to the application

4. **Path Issues**:
   - Make sure you're running the commands from the project's root directory

5. **Browser Issues**:
   - Try a different browser if one doesn't work

## Data Format

The application expects transaction data in CSV or Excel format with the following columns:
- `TransactionID`: Unique identifier for each transaction
- `Date`: Transaction date (in a standard date format)
- `ProductID`: Unique identifier for each product
- `Quantity`: Number of items purchased
- `Price`: Unit price of the product (optional)

## Developer Information

© 2025 Mufaro Mabaudi - All Rights Reserved