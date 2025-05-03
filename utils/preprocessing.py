import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_data(sample_size=50000):
    """
    Load and preprocess the PaySim dataset with optional sampling.
    
    Parameters:
        sample_size (int): Number of rows to sample from the dataset
    
    Returns:
        pd.DataFrame: Processed dataframe
    """
    # Define paths
    data_dir = os.path.join(os.getcwd(), 'data')
    data_file = os.path.join(data_dir, 'financial_fraud_data.csv')
    
    # Check if the data file exists
    if not os.path.exists(data_file):
        # Create synthetic data that resembles the PaySim dataset structure
        np.random.seed(42)  # For reproducibility
        
        # Number of transactions to generate
        n_transactions = max(sample_size * 3, 100000)
        
        # Generate transaction types with appropriate distribution
        transaction_types = np.random.choice(
            ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'],
            size=n_transactions,
            p=[0.2, 0.3, 0.05, 0.2, 0.25]  # Approximate distribution from original data
        )
        
        # Generate transaction amounts (lognormal distribution)
        amounts = np.random.lognormal(mean=8, sigma=1.5, size=n_transactions)
        
        # Generate origin account balances
        old_balance_orig = np.random.lognormal(mean=9, sigma=2, size=n_transactions)
        
        # Set new balance orig based on transaction type and amount
        new_balance_orig = np.zeros_like(old_balance_orig)
        for i in range(n_transactions):
            if transaction_types[i] in ['CASH_OUT', 'TRANSFER', 'PAYMENT', 'DEBIT']:
                # Deduct amount from balance, but don't go below 0
                new_balance_orig[i] = max(0, old_balance_orig[i] - amounts[i])
            elif transaction_types[i] == 'CASH_IN':
                # Add amount to balance
                new_balance_orig[i] = old_balance_orig[i] + amounts[i]
        
        # Generate destination balances
        old_balance_dest = np.random.lognormal(mean=8.5, sigma=2, size=n_transactions)
        
        # Set new destination balance based on transaction type and amount
        new_balance_dest = np.zeros_like(old_balance_dest)
        for i in range(n_transactions):
            if transaction_types[i] in ['CASH_IN', 'TRANSFER']:
                # Add amount to balance
                new_balance_dest[i] = old_balance_dest[i] + amounts[i]
            elif transaction_types[i] == 'CASH_OUT':
                # Deduct amount from balance, but don't go below 0
                new_balance_dest[i] = max(0, old_balance_dest[i] - amounts[i])
            else:
                # PAYMENT or DEBIT - balance doesn't change
                new_balance_dest[i] = old_balance_dest[i]
        
        # Generate origin and destination accounts
        name_orig = np.array([f'C{np.random.randint(1000000000, 9999999999)}' for _ in range(n_transactions)])
        name_dest = np.array([f'C{np.random.randint(1000000000, 9999999999)}' if np.random.random() < 0.7 
                              else f'M{np.random.randint(1000000000, 9999999999)}' for _ in range(n_transactions)])
        
        # Generate fraud labels
        # Fraud conditions based on PaySim patterns:
        # 1. Mostly in TRANSFER and CASH_OUT transactions
        # 2. Often involves emptying accounts
        # 3. Transaction amount matches original balance
        is_fraud = np.zeros(n_transactions, dtype=int)
        
        for i in range(n_transactions):
            # Only TRANSFER and CASH_OUT can be fraudulent
            if transaction_types[i] in ['TRANSFER', 'CASH_OUT']:
                # Account emptied condition
                account_emptied = (old_balance_orig[i] > 0 and new_balance_orig[i] == 0)
                
                # Amount matches balance condition (within 5%)
                amount_matches_balance = (abs(amounts[i] - old_balance_orig[i]) / old_balance_orig[i] < 0.05) if old_balance_orig[i] > 0 else False
                
                # Randomly mark as fraud based on conditions
                if (account_emptied or amount_matches_balance) and np.random.random() < 0.2:
                    is_fraud[i] = 1
        
        # Flagged fraud (attempts to transfer more than 200,000)
        is_flagged_fraud = np.zeros(n_transactions, dtype=int)
        for i in range(n_transactions):
            if transaction_types[i] == 'TRANSFER' and amounts[i] > 200000:
                is_flagged_fraud[i] = 1
        
        # Create a dataframe
        df = pd.DataFrame({
            'step': np.random.randint(1, 744, size=n_transactions),  # 744 hours in a month
            'type': transaction_types,
            'amount': amounts,
            'nameOrig': name_orig,
            'oldbalanceOrg': old_balance_orig,
            'newbalanceOrig': new_balance_orig,
            'nameDest': name_dest,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'isFraud': is_fraud,
            'isFlaggedFraud': is_flagged_fraud
        })
        
        # Save the synthetic data
        os.makedirs(data_dir, exist_ok=True)
        df.to_csv(data_file, index=False)
        print(f"Generated synthetic financial fraud data and saved to {data_file}")
    else:
        # Load the existing data file
        print(f"Loading existing data from {data_file}")
        df = pd.read_csv(data_file)
    
    # Sample the data to the requested size
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
