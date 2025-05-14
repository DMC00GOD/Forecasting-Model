# prophet_model.py

import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
import os
import plotly.express as px
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def run_prophet():
    # Define data path
    data_path = "C:/Users/premb/Desktop/Forecasting-Model/data/cleaned_sales.csv"

    try:
        # Verify data path
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at path: {data_path}")

        # Load data
        data = pd.read_csv(data_path)

        # Ensure date column is in datetime format
        data['date'] = pd.to_datetime(data['date'])

        # Rename columns to Prophet's expected format
        data = data.rename(columns={"date": "ds", "sales": "y"})

        # Handle missing data
        data.fillna(0, inplace=True)

        # Initialize the Prophet model with hyperparameters
        model = Prophet(
            changepoint_prior_scale=0.5,  # Sensitivity to trend changes
            seasonality_mode='additive',
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        # Add regressors if present
        for reg in ['promo', 'holiday', 'is_weekend']:
            if reg in data.columns:
                model.add_regressor(reg)

        # Fit the model
        model.fit(data)

        # Create a DataFrame for future dates
        future = model.make_future_dataframe(periods=30)
        future['promo'] = 0  # Default value
        future['holiday'] = 0  # Default value
        future['is_weekend'] = (future['ds'].dt.dayofweek >= 5).astype(int)

        # Predict the future
        forecast = model.predict(future)

        # Plot the forecast
        fig = model.plot(forecast)
        plt.title("Enhanced Prophet Forecast for Sales")
        plt.show()

        # Plot forecast components
        model.plot_components(forecast)
        plt.show()

        # Calculate Evaluation Metrics
        y_true = data['y']
        y_pred = forecast['yhat'][:len(y_true)]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

        # Save the forecast to CSV
        output_path = "C:/Users/premb/Desktop/Forecasting-Model/data/sales_forecast.csv"
        forecast.to_csv(output_path, index=False)
        print(f"Forecast saved to {output_path}")

    except Exception as e:
        print(f"Error in Prophet model: {e}")


if __name__ == "__main__":
    run_prophet()
