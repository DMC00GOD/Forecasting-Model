# Forecasting-Model

## Project Objective

This project aims to build a robust and advanced forecasting model for predicting e-commerce sales.  
The focus is on accurately modeling seasonal trends, holiday effects, and promotional impacts to enhance decision-making for inventory management and marketing strategies.

## Project Structure

│── data/ # Raw and processed data
│── notebooks/ # Jupyter Notebooks for EDA
│── scripts/ # Scripts for data processing and modeling
│── models/ # Trained models and performance metrics
│── reports/ # Documentation and results
│── README.md # Project overview
│── requirements.txt # Dependencies
│── main.py # Main script


## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Prophet
- ARIMA
- LSTM (Keren/TensorFlow)
- Scikit-Learn

## Key Features



##  Dataset
- The dataset consists of daily sales data, including promotional and holiday information.

##  How to Run the Project

1. Clone the repository:  "https://github.com/DMC00GOD/Forecasting-Model.git"

2. Install the dependencies: "pip install -r requirements.txt"

3. Run the preprocessing script: "python scripts/data_preprocessing.py"

4. Run the main forecasting model: "python main.py"

# Ecommerce Sales Forecasting Project

### ✅ Project Overview
This project focuses on building an advanced sales forecasting model using Prophet for an e-commerce business. The objective is to accurately predict future sales while identifying significant peaks and troughs using enhanced visualization techniques.

### 📦 Project Structure
```
Forecasting-Model/
│── data/                   # Raw and processed data
│   ├── cleaned_sales.csv
│   ├── sales_forecast.csv
│   └── enhanced_sales_forecast.png
│── notebooks/              # EDA and analysis
│── scripts/                # Data processing and model scripts
│   ├── data_preprocessing.py
│   ├── prophet_model.py
│   └── forecast_visualization.py
│── README.md               # Project overview
│── requirements.txt        # Project dependencies
```

### 🚀 Technologies and Libraries
- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Prophet
- Scikit-Learn

### 🔍 Data Processing and Forecasting
- **Data Preprocessing:** Outlier detection, scaling, and feature engineering.
- **Prophet Model:** Implementation of Prophet for trend and seasonality analysis.
- **Enhanced Visualization:** Marking significant forecast points (peaks and troughs) and improved grid design for clarity.

### 📈 Evaluation Metrics
- **MAE (Mean Absolute Error)**: Measures average absolute errors in predictions.
- **MSE (Mean Squared Error)**: Penalizes large errors more heavily than MAE.
- **RMSE (Root Mean Squared Error)**: Provides error metric in original sales units.

### 🛠️ Usage Instructions
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Data Preprocessing:**
   ```bash
   python scripts/data_preprocessing.py
   ```

3. **Run Prophet Model:**
   ```bash
   python scripts/prophet_model.py
   ```

4. **Run Enhanced Visualization:**
   ```bash
   python scripts/forecast_visualization.py
   ```
   - Outputs: `enhanced_sales_forecast.png` (highlighting peaks and troughs)

### ✅ Key Takeaways
- Comprehensive forecast visualization with improved interpretability.
- Data-driven insights through peak/trough identification.
- Effective implementation of error metrics for model evaluation.
