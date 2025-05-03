import pandas as pd
import numpy as np
import pickle
import io
import base64
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_test_data_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Parameters:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        test_size (float): Proportion of the dataset to include in the test split
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_model(X_train, y_train, model_type='random_forest', params=None):
    """
    Train a machine learning model.
    
    Parameters:
        X_train (array): Training features
        y_train (array): Training target
        model_type (str): Type of model to train
        params (dict): Model parameters
        
    Returns:
        model: Trained model
    """
    if model_type == 'logistic_regression':
        model = LogisticRegression(
            C=params.get('C', 1.0),
            max_iter=params.get('max_iter', 100),
            random_state=42
        )
    elif model_type == 'decision_tree':
        model = DecisionTreeClassifier(
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            random_state=42
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', None),
            min_samples_split=params.get('min_samples_split', 2),
            random_state=42
        )
    elif model_type == 'svm':
        model = SVC(
            C=params.get('C', 1.0),
            kernel=params.get('kernel', 'rbf'),
            probability=True,
            random_state=42
        )
    elif model_type == 'knn':
        model = KNeighborsClassifier(
            n_neighbors=params.get('n_neighbors', 5),
            weights=params.get('weights', 'uniform')
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate a trained model.
    
    Parameters:
        model: Trained model
        X_test (array): Testing features
        y_test (array): Testing target
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X_test)
    
    # For models that support probability prediction
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except:
        y_proba = y_pred
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred)
    }
    
    return metrics

def save_model(model, model_name='fraud_detection_model'):
    """
    Save the model to a binary file and return as a download link.
    
    Parameters:
        model: Trained model
        model_name (str): Name of the model
        
    Returns:
        str: Download link in HTML
    """
    # Save model to a pickle file
    model_pickle = pickle.dumps(model)
    
    # Convert to base64
    b64 = base64.b64encode(model_pickle).decode()
    
    # Generate download link
    href = f'<a href="data:file/pickle;base64,{b64}" download="{model_name}.pkl">Download {model_name}</a>'
    
    return href

def predict_transaction(model, transaction_data, feature_names, scaler=None):
    """
    Make a prediction for a single transaction.
    
    Parameters:
        model: Trained model
        transaction_data (dict): Transaction data
        feature_names (list): List of feature names
        scaler: Fitted scaler object (optional)
        
    Returns:
        tuple: prediction (0 or 1), probability
    """
    # Convert transaction data to a dataframe
    df = pd.DataFrame([transaction_data])
    
    # Extract features
    X = df[feature_names]
    
    # Scale features if a scaler is provided
    if scaler is not None:
        X = scaler.transform(X)
    
    # Make prediction
    prediction = model.predict(X)[0]
    
    # Get probability if available
    try:
        probability = model.predict_proba(X)[0, 1]
    except:
        probability = float(prediction)
    
    return prediction, probability
