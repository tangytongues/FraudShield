import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(sample_size=50000):
    """
    Load and preprocess the PaySim dataset with optional sampling.
    
    Parameters:
        sample_size (int): Number of rows to sample from the dataset
    
    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Load the data
    # To handle large file size, we'll use chunks and sample
    chunk_size = min(sample_size * 2, 500000)  # Make chunk size at least 2x sample size but cap at 500k
    
    # Load a single chunk and sample from it
    df = pd.read_csv('../input/paysim1/PS_20174392719_1491204439457_log.csv', 
                     nrows=chunk_size)
    
    # Sample the data
    if len(df) > sample_size:
        df = df.sample(sample_size, random_state=42)
    
    # Basic preprocessing steps
    df = basic_preprocessing(df)
    
    return df

def basic_preprocessing(df):
    """
    Apply basic preprocessing steps to the dataframe.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Extract the customer and merchant type from the name
    df['customer_type'] = df['nameOrig'].str[0]
    df['dest_type'] = df['nameDest'].str[0]
    
    # Calculate transaction balance differences
    df['orig_balance_diff'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['dest_balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Create a flag for transactions where the origin account was emptied
    df['is_orig_acct_emptied'] = (df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)
    
    # Create a flag for transactions where the amount exceeds the original balance
    df['amount_exceeds_balance'] = df['amount'] > df['oldbalanceOrg']
    
    # Encode transaction type
    le = LabelEncoder()
    df['type_encoded'] = le.fit_transform(df['type'])
    
    return df

def prepare_features(df, target='isFraud'):
    """
    Prepare features for modeling.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
        target (str): Target variable name
        
    Returns:
        tuple: X (features), y (target), feature_names (list of feature names)
    """
    # Select numerical features
    numerical_features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest', 'orig_balance_diff', 
        'dest_balance_diff', 'type_encoded'
    ]
    
    # Add binary features
    binary_features = ['is_orig_acct_emptied', 'amount_exceeds_balance']
    
    # Combine all features
    feature_names = numerical_features + binary_features
    
    # Prepare X and y
    X = df[feature_names]
    y = df[target]
    
    return X, y, feature_names

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler.
    
    Parameters:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: Scaled X_train and X_test, and the scaler object
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler
