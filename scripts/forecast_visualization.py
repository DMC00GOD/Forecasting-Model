# forecast_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error


def plot_forecast(forecast_file):
    """Visualize the forecast data with enhanced seasonality components and improved scaling."""
    
    # Load the forecast data
    try:
        forecast_data = pd.read_csv(forecast_file)
        print(f"Successfully loaded data from {forecast_file}")
    except FileNotFoundError:
        print(f"Error: {forecast_file} not found.")
        return
    
    # Display initial data info
    print("\nData Info:")
    print(forecast_data.info())
    print("\nData Head:")
    print(forecast_data.head())

    # Cap and floor the forecast values to prevent unrealistic negative predictions
    forecast_data['yhat'] = np.maximum(forecast_data['yhat'], 0)
    forecast_data['yhat_lower'] = np.maximum(forecast_data['yhat_lower'], 0)
    forecast_data['yhat_upper'] = np.maximum(forecast_data['yhat_upper'], 0)

    # Calculate evaluation metrics
    if 'y' in forecast_data.columns:
        mae = mean_absolute_error(forecast_data['y'], forecast_data['yhat'])
        mse = mean_squared_error(forecast_data['y'], forecast_data['yhat'])
        rmse = np.sqrt(mse)
        print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")

    # Enhanced Plotting
    plt.figure(figsize=(14, 7))
    plt.plot(forecast_data['ds'], forecast_data['yhat'], label='Forecast', color='blue', linewidth=2)
    plt.fill_between(forecast_data['ds'], forecast_data['yhat_lower'], forecast_data['yhat_upper'], color='gray', alpha=0.3, label='Confidence Interval')

    # Highlight key points
    max_point = forecast_data['yhat'].max()
    min_point = forecast_data['yhat'].min()
    max_date = forecast_data.loc[forecast_data['yhat'] == max_point, 'ds'].values[0]
    min_date = forecast_data.loc[forecast_data['yhat'] == min_point, 'ds'].values[0]
    plt.scatter(max_date, max_point, color='red', s=100, zorder=5, label='Peak Forecast')
    plt.scatter(min_date, min_point, color='green', s=100, zorder=5, label='Lowest Forecast')

    # Formatting
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Sales Forecast')
    plt.title('Enhanced Sales Forecast Visualization with Seasonality Adjustments')
    plt.grid(alpha=0.3, linestyle='--')
    plt.legend()
    plt.tight_layout()

    # Save the plot
    output_dir = os.path.join(os.path.dirname(forecast_file), "plots")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "enhanced_sales_forecast.png")
    plt.savefig(output_file)
    plt.show()
    
    print(f"Enhanced forecast visualization saved to {output_file}")

if __name__ == "__main__":
    # Define the path to the forecast data
    forecast_path = os.path.join("..", "data", "sales_forecast.csv")
    print(f"Forecast path: {forecast_path}")
    plot_forecast(forecast_path)
