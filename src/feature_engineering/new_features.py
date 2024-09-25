import pandas as pd

def calculate_time_delta(df: pd.DataFrame, time_col: str, ref_col: str) -> pd.Series:
    """
    Calculate the time difference with full precision (including microseconds) 
    between consecutive rows, grouped by a reference column.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the time and reference columns.
    time_col : str
        The name of the datetime64 column representing the time of each transaction/event.
    ref_col : str
        The name of the column to group by (e.g., a unique identifier for each business, user, etc.).
    
    Returns:
    --------
    pd.Series
        A Pandas Series containing the time difference (in seconds with microsecond precision) 
        between consecutive rows for each group. The first transaction for each group will have NaN as its delta.
    """
    
    # Sort by the reference column and time column to ensure correct order
    df = df.sort_values(by=[ref_col, time_col])
    
    # Calculate the time difference in seconds (with full precision) for each group
    time_delta = df.groupby(ref_col)[time_col].diff().dt.total_seconds()
    
    # Return the Series with NaN for the first transaction in each group
    return time_delta

def extract_bin(df: pd.DataFrame, num_col: str) -> pd.Series:
    """
    Extract the Bank Identification Number (BIN) from a column containing credit card numbers.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the credit card numbers.
    num_col : str
        The name of the column containing the credit card numbers as strings or integers.

    Returns:
    --------
    pd.Series
        A Pandas Series containing the extracted BIN (first 6 digits) from each credit card number.
    """
    # Ensure the credit card numbers are strings, then extract the first 6 digits (BIN)
    bin_series = df[num_col].astype(str).str[:6]
    
    return bin_series

def extract_check_digit(df: pd.DataFrame, num_col: str) -> pd.Series:
    """
    Extract the check digit (last digit) from a column containing credit card numbers.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the credit card numbers.
    num_col : str
        The name of the column containing the credit card numbers as strings or integers.

    Returns:
    --------
    pd.Series
        A Pandas Series containing the check digit (last digit) from each credit card number.
    """
    # Ensure the credit card numbers are strings, then extract the last digit (check digit)
    check_digits = df[num_col].astype(str).str[-1]
    
    return check_digits

def extract_datetime_component(df: pd.DataFrame, datetime_col: str, component: str) -> pd.Series:
    # Ensure the column is in datetime format
    df[datetime_col] = pd.to_datetime(df[datetime_col], errors='coerce')
    
    # Initialize the result with NaN
    result = pd.Series(index=df.index)

    if component == 'month':
        result = df[datetime_col].dt.month.astype(int)
    elif component == 'day':
        result = df[datetime_col].dt.day.astype(int)
    elif component == 'hour':
        result = df[datetime_col].dt.hour.astype(int)
    elif component == 'minute':
        result = df[datetime_col].dt.minute.astype(int)
    elif component == 'time':
        result = df[datetime_col].dt.strftime('%H:%M')
    else:
        raise ValueError("Invalid component. Choose from 'month', 'day', 'hour', 'minute', or 'time'.")
    
    return result


def average_transactions_per_time(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Calculate the average number of transactions per time in HH:MM format.
    Returns a Series with the same index as the original DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the transaction data.
    time_col : str
        The name of the column containing the time of transactions in HH:MM format.

    Returns:
    --------
    pd.Series
        A Pandas Series indexed by the original DataFrame's index, with the average number of transactions.
    """
    # Group by the time column and count the number of transactions for each time
    transaction_counts = df[time_col].value_counts()
    
    # Calculate the mean number of transactions
    mean_transactions = transaction_counts.mean()
    
    # Create a Series with the average counts for each time
    average_series = transaction_counts / mean_transactions
    
    # Reindex to match the original DataFrame's index and fill missing values with 0
    return df[time_col].map(average_series).fillna(0)