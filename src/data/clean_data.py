import pandas as pd

def drop_columns_if_exist(dataframe: pd.DataFrame, columns_to_remove: list) -> pd.DataFrame:
    """Drop columns from a DataFrame if they exist.

    Args:
        dataframe (pd.DataFrame): The DataFrame from which to drop columns.
        columns_to_remove (list): A list of column names to remove.

    Returns:
        pd.DataFrame: The DataFrame with the specified columns removed.
    """
    columns_to_remove = [column for column in columns_to_remove if column in dataframe.columns]
    return dataframe.drop(columns=columns_to_remove, errors="ignore")

# Function to convert and clean DataFrame
def update_dtypes(df: pd.DataFrame, dtypes_dict: dict) -> pd.DataFrame:
    """
    Update the data types of a DataFrame based on a dictionary of column names and data types.

    This function will attempt to convert the columns of the DataFrame to the specified data type.
    If the conversion fails, it will drop the rows that contain the error and print a message
    indicating the column and the error.

    Args:
        df (pd.DataFrame): The DataFrame to update.
        dtypes_dict (dict): A dictionary where the keys are column names and the values are the
            desired data types.

    Returns:
        pd.DataFrame: The DataFrame with the updated data types.
    """
    total_rows = len(df)
    rows_dropped = 0
    drop_indices = []

    for column, dtype in dtypes_dict.items():
        if column in df.columns:
            try:
                # Attempt to convert the column to the specified data type
                if dtype == 'datetime64[ns]':
                    df[column] = pd.to_datetime(df[column], errors='coerce')
                elif dtype == 'float32':
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('float32')
                elif dtype == 'int8':
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('int8')
                elif dtype == 'int16':
                    df[column] = pd.to_numeric(df[column], errors='coerce').astype('int16')
                elif dtype == 'bool':
                    # Convert the column to a boolean type
                    # We need to replace the values because the column might contain strings
                    df[column] = df[column].replace({1: True, 0: False, 'true': True, 'false': False, 'True': True, 'False': False}).astype('boolean')
                elif dtype == 'category':
                    df[column] = df[column].astype('category')
            except Exception as e:
                # If the conversion fails, drop the rows that contain the error
                print(f"Error converting column '{column}': {e}")
                drop_indices.extend(df.index[df[column].isna()])

    drop_indices = set(drop_indices)
    df = df.drop(index=drop_indices)
    rows_dropped = len(drop_indices)
    print(f"Total rows: {total_rows}")
    print(f"Number of rows dropped: {rows_dropped}")

    return df

