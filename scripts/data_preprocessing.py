## data_preprocessing.py

import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(file_path):
    """ Advanced data preprocessing for e-commerce sales forecasting """
    
    try:
        # Load the dataset
        data = pd.read_csv(file_path)
        data['date'] = pd.to_datetime(data['date'])

        # Handle missing sales data using median
        data['sales'].fillna(data['sales'].median(), inplace=True)

        # Create weekend indicator
        data['is_weekend'] = data['date'].dt.dayofweek.apply(lambda x: 1 if x >= 5 else 0)

        # Encode seasons (Winter: 1, Spring: 2, Summer: 3, Fall: 4)
        season_mapping = {'winter': 1, 'spring': 2, 'summer': 3, 'fall': 4}
        data['season'] = data['season'].map(season_mapping)

        # Normalize the economic index
        scaler = MinMaxScaler()
        data['economic_index'] = scaler.fit_transform(data[['economic_index']])

        # Handle outliers using IQR method
        Q1 = data['sales'].quantile(0.25)
        Q3 = data['sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data['sales'] >= lower_bound) & (data['sales'] <= upper_bound)]

        # Save processed data
        output_path = os.path.join("data", "cleaned_sales.csv")
        data.to_csv(output_path, index=False)
        print(f"Data processed and saved to {output_path}")

    except Exception as e:
        print(f"Error during preprocessing: {e}")

if __name__ == "__main__":
    preprocess_data("C:/Users/premb/Desktop/Forecasting-Model/data/ecommerce_sales.csv")

