import numpy as np
import json
import os as os
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

# Catcategorical variables mapping
def map_categorical_columns(df, directory='..\json'):
    # Function to load mappings from JSON files
    def load_mappings_from_directory(directory):
        import os
        mappings = {}
        for filename in os.listdir(directory):
            if filename.endswith(".json"):
                column_name = os.path.splitext(filename)[0]
                with open(os.path.join(directory, filename), 'r') as f:
                    mappings[column_name] = json.load(f)
        return mappings

    # Load mappings
    mappings = load_mappings_from_directory(directory)

    # Function to map categorical column
    def map_categorical_column(value, mapping):
        return mapping.get(value, value)

    # Map categorical columns
    df_mapped = df.copy()
    for column, mapping in mappings.items():
        df_mapped[column] = df_mapped[column].apply(lambda x: map_categorical_column(x, mapping))

    return df_mapped

#Feature Engineering
def transform_date_feature(df):
    X_copy=df.copy()
    X_copy['arrival_date'] = pd.to_datetime(X_copy['arrival_date_year'].astype(str) + '-' + X_copy['arrival_date_month'] + '-' + X_copy['arrival_date_day_of_month'].astype(str), format='%Y-%B-%d')
    X_copy['arrival_date_month'] = X_copy['arrival_date_month'].apply(lambda x: datetime.strptime(x, "%B").month)
    X_copy['reservation_status_date'] = pd.to_datetime(X_copy['reservation_status_date'])
    return X_copy

def featuredrop(df):
    X_copy=df.copy()
    X_copy.drop(columns=['arrival_date_year', 'arrival_date_week_number'], inplace=True)
    X_copy.drop(columns=['reservation_status_date', 'arrival_date'], inplace=True)
    X_copy.drop(columns=['agent','company'], inplace=True) #Removing agent and company column as they reprent Agent or company ID
    return X_copy

def normalisation(df):
    # Initialize the MinMaxScaler
    scaler = MinMaxScaler()
    # Fit and transform the DataFrame
    normalized_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    return normalized_df

def Preprocess_inputs(df):
    df1=map_categorical_columns(df)
    df2=transform_date_feature(df1)
    df3=featuredrop(df2)
    df4=normalisation(df3)
    return df4