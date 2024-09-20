import pandas as pd


df_dtypes = {
    'id': 'category', 'acquiring_bank_name': 'category', 'approval_code': 'category',
    'authorization_transaction_date': 'datetime64[ns]', 'authorized_amount': 'float32',
    'avs_code': 'category', 'bank_merchant_id': 'category', 'bank_reconciliation_id': 'category',
    'business_id': 'category', 'capture_amount': 'float32', 'card_brand': 'category',
    'card_expiration_month': 'int8', 'card_expiration_year': 'int16', 'card_holder_name': 'category',
    'card_type': 'category', 'cavv': 'category', 'cc_version': 'category', 'charge_type': 'category',
    'client_id': 'category', 'client_type': 'category', 'country': 'category', 'country_code': 'category',
    'created': 'datetime64[ns]', 'credit_card_payment_channel': 'category', 'credit_card_processor': 'category',
    'credit_card_token_id': 'category', 'currency': 'category', 'cvn_code': 'category', 'eci': 'category',
    'external_id': 'category', 'auth_failure_reason': 'category','charge_failure_reason': 'category', 'fee_amount': 'float32', 'is_blocked_by_fraud': 'bool',
    'is_switcher': 'bool', 'is_t4': 'bool', 'issuing_bank_name': 'category', 'merchant_id': 'category',
    'refund_status': 'category', 'requester_email': 'category', 'reversed_amount': 'float32',
    'settlement_status': 'category', 'settlement_updated': 'datetime64[ns]', 'should_authenticate_credit_card': 'bool',
    'should_settle_directly': 'bool', 'status': 'category', 'total_refund_amount': 'float32',
    'total_refund_fee_amount': 'float32', 'transaction_channel': 'category', 'ucaf': 'category',
    'updated': 'datetime64[ns]', 'user_id': 'category', 'use_reward': 'category', 'dt': 'datetime64[ns]',
    'amount': 'float32', 'authentication_type': 'category', 'card_bank': 'category', 'commerce_indicator': 'category',
    'credit_card_enrollment_info': 'category', 'cybersource_merchant_id': 'category', 'eci_raw': 'category', 'ip_address': 'category',
    'is_enrolled': 'bool', 'cof_type': 'category', 'auth_status': 'category', 'descriptor': 'category', 'charge_failure_reason': 'category',
    'mid_label': 'category', 'masked_card_number': 'string'
    
}

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
def update_dtypes(df: pd.DataFrame, dtypes_dict: dict=df_dtypes) -> pd.DataFrame:
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
                    df[column] = df[column].replace({1: True, 0: False, 'true': True, 'false': False, 'True': True, 'False': False}).astype('boolean')
                elif dtype == 'category':
                    df[column] = df[column].astype('category')
                elif dtype == 'string':
                    df[column] = df[column].astype('string')  # Change to string type
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

