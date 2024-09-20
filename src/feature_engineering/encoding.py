import pandas as pd
from sklearn.preprocessing import LabelEncoder
import numpy as np



columns = ['descriptor', 'mid_label', 'card_bank'] 

encoding_strategy = {
    "business_id": "frequency",
    "auth_status": "label",
    "eci": "label",
    "descriptor": "label",
    "auth_failure_reason": "label",
    "charge_failure_reason": "label",
    "mid_label": "label",
    "bank_merchant_id": "frequency",
    "authentication_type": "label",
    "card_bank": "label",
    "card_brand": "label",
    "cavv": "label",
    "country": "label",
    "currency": "label",
    "ip_address": "frequency or target",
    "card_holder_name": "frequency",
    "card_type": "label"
}



def encode_categories(df: pd.DataFrame, encoding_strategy: dict = encoding_strategy) -> pd.DataFrame:
    """
    Encodes categorical variables in a pandas DataFrame based on a provided encoding strategy.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing categorical variables to be encoded.
        encoding_strategy (dict): A dictionary where keys are column names and values are encoding strategies ('label' or 'frequency').

    Returns:
        pd.DataFrame: The modified DataFrame with encoded categorical variables.
    """
    label_encoder = LabelEncoder()
    
    for key, value in encoding_strategy.items():
        if value == "label":
            df[key] = label_encoder.fit_transform(df[key])
        elif value == "frequency":
            frequency_encoding = df[key].value_counts().to_dict()
            df[key] = df[key].map(frequency_encoding)
    
    return df


def group_low_frequency_categories(df: pd.DataFrame, columns: list = columns, percentile: float = 0.90) -> pd.DataFrame:
    """
    Groups low-frequency categories in a DataFrame column into an 'Others' category.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing categorical variables to be grouped.
        columns (list): A list of column names to be grouped.
        percentile (float): The percentile threshold for determining low-frequency categories (default is 0.9).
    
    Returns:
        pd.DataFrame: The modified DataFrame with grouped categorical variables.
    """
    for column in columns:
        # Calculate the frequency of each category
        frequency_counts = df[column].value_counts()
        
        # Determine the threshold for the 90th percentile
        threshold = frequency_counts.quantile(percentile)
        
        # Create a mask for categories below the threshold
        low_frequency_categories = frequency_counts[frequency_counts < threshold].index
        
        # Replace low-frequency categories with 'Others'
        df[column] = df[column].replace(low_frequency_categories, 'Others')
        
        # Print the cardinality before and after grouping
        before_cardinality = frequency_counts.size  # Unique categories before
        after_cardinality = df[column].nunique()  # Unique categories after
        print(f"{column}: {before_cardinality} --> {after_cardinality} unique categories")

    return df


numeric_columns = ['amount', 'authorized_amount', 'capture_amount', 'card_expiration_month','card_expiration_year']
def log_encode(df: pd.DataFrame, columns: list = numeric_columns) -> pd.DataFrame:
    """
    Applies log encoding to specified columns in the DataFrame and replaces NaN values with zero.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to apply log encoding.

    Returns:
    pd.DataFrame: The modified DataFrame with log-encoded columns and NaN values replaced by zero.
    """
    for column in columns:
        if column in df.columns:
            # Replace zero or negative values with NaN to avoid log(0) or log(negative)
            df[column] = np.where(df[column] <= 0, np.nan, np.log(df[column]))
        else:
            print(f"Warning: {column} not found in DataFrame.")
    
    # Replace NaN values with zero
    df[columns] = df[columns].fillna(0)
    
    return df