import sqlite3
import pandas as pd

# Save DataFrames to SQLite databases
def save_to_sqlite(df: pd.DataFrame, db_path: str, table_name: str):
    """
    Save a DataFrame to a SQLite database.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame to be saved.
    db_path : str
        The path to the SQLite database file.
    table_name : str
        The name of the table in the database.

    Returns
    -------
    None
    """
    with sqlite3.connect(db_path) as conn:
        # Replace the table if it already exists
        df.to_sql(table_name, conn, if_exists='replace', index=False)

def query_data(db_path: str, table_name: str) -> pd.DataFrame:
    """
    Function to query data from a specified table in the given database.

    Parameters:
    db_path (str): The full path to the database file.
    table_name (str): The name of the table to query.

    Returns:
    pd.DataFrame: The queried data as a DataFrame.
    """
    query = f"SELECT * FROM {table_name}"
    
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(query, conn)
    
    return df