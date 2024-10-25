import pandas as pd
from collections import deque
from datetime import timedelta
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

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

def process_chunk(chunk, time_col, ref_cols, interval_minutes):
    window = deque()
    last_interval_counts = []
    
    for _, row in chunk.iterrows():
        current_time = row[time_col]
        interval_ago = current_time - timedelta(minutes=interval_minutes)
        
        while window and window[0][1] < interval_ago:
            window.popleft()
        
        window.append((tuple(row[ref_cols]), current_time))
        count_in_window = sum(1 for ref, _ in window if ref == tuple(row[ref_cols]))
        last_interval_counts.append(count_in_window)
    
    return pd.Series(last_interval_counts, index=chunk.index)

def transactions_in_last_interval(df: pd.DataFrame, time_col: str, ref_cols: list, interval_minutes: int = 1, n_processes: int = None) -> pd.Series:
    """
    Calculate how many transactions occurred in the last 'interval_minutes' for each row,
    grouped by multiple reference columns, using parallel processing.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the transaction data.
    time_col : str
        The name of the datetime64 column representing the time of each transaction/event.
    ref_cols : list
        A list of reference columns to group by (e.g., ['business_id', 'card_bin']).
    interval_minutes : int, optional
        The time interval in minutes to calculate the transaction count (default is 1 minute).
    n_processes : int, optional
        The number of processes to use for parallel processing. If None, it uses all available CPU cores.
    
    Returns:
    --------
    pd.Series
        A Pandas Series with the count of transactions within the specified interval for each row,
        with the same index as the original DataFrame.
    """
    
    # Sort the DataFrame by the reference columns and the time column
    sorted_df = df.sort_values(by=ref_cols + [time_col])
    
    # Determine the number of processes to use
    if n_processes is None:
        n_processes = mp.cpu_count()
    
    # Split the DataFrame into chunks
    chunk_size = len(sorted_df) // n_processes
    chunks = [sorted_df.iloc[i:i + chunk_size] for i in range(0, len(sorted_df), chunk_size)]
    
    # Create a partial function with fixed parameters
    partial_process = partial(process_chunk, time_col=time_col, ref_cols=ref_cols, interval_minutes=interval_minutes)
    
    # Create a multiprocessing pool and apply the function to each chunk
    results = []
    with mp.Pool(processes=n_processes) as pool:
        for result in tqdm(pool.imap(partial_process, chunks), total=len(chunks), desc="Processing chunks"):
            results.append(result)
    
    # Combine the results
    combined_results = pd.concat(results)
    
    # Return the result series re-indexed to match the original DataFrame
    return combined_results.reindex(df.index)


def transactions_last_minute(df: pd.DataFrame, time_col: str) -> pd.Series:
    """
    Count the number of transactions that occurred in the last minute 
    before each transaction in the specified time column.

    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame containing the transaction timestamps.
    time_col : str
        The name of the column representing the time of each transaction/event.

    Returns:
    --------
    pd.Series
        A Pandas Series containing the count of transactions that occurred in the last minute 
        before each transaction timestamp. The first transaction will have a count of 0.
    """
    
    # Ensure the specified column is in datetime64 format
    df[time_col] = pd.to_datetime(df[time_col])

    # Initialize an empty list to store the counts
    counts = []

    # Iterate over each transaction's timestamp with a progress bar
    for timestamp in tqdm(df[time_col], desc="Counting transactions in last minute"):
        # Define the time range for the last minute before the current timestamp
        one_minute_ago = timestamp - pd.Timedelta(minutes=1)
        
        # Count the number of transactions in the last minute
        count = df[(df[time_col] > one_minute_ago) & (df[time_col] <= timestamp)].shape[0]
        
        # Append the count to the list
        counts.append(count)

    # Return a Pandas Series with the counts
    return pd.Series(counts, index=df.index)

def calculate_time_for_n_transactions(df: pd.DataFrame, time_column: str, n: int) -> pd.Series:
    """
    Calculate the time difference in seconds between every n-th transaction.
    
    This function calculates the time difference between the current transaction
    and the (n-1)-th previous transaction in the dataframe, returning a Pandas Series
    with the results without modifying the original dataframe.

    Parameters:
    ----------
    df : pd.DataFrame
        The dataframe containing the transaction data.
        
    time_column : str
        The column name representing the datetime values in the dataframe (must be in `datetime64` format).
    
    n : int
        The number of transactions to calculate the time difference for.

    Returns:
    -------
    pd.Series
        A Pandas Series containing the time difference in seconds between each transaction
        and the (n-1)-th previous transaction. For the first (n-1) transactions, the value will be 0.
    """
    
    # Ensure the dataframe is sorted by the time column
    df_sorted = df.sort_values(by=[time_column], ascending=True)
    
    # Calculate the time difference for every n-th transaction
    time_difference = df_sorted[time_column] - df_sorted[time_column].shift(n-1)
    
    # Convert the time difference to seconds
    time_difference = time_difference.dt.total_seconds()
    
    # Fill NaN values resulting from the first n-1 transactions with 0
    time_difference.fillna(0, inplace=True)
    
    return time_difference