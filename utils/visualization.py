import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc

def plot_transaction_distribution(df):
    """
    Plot the distribution of transaction types.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        fig: Plotly figure object
    """
    # Count transactions by type
    type_counts = df['type'].value_counts().reset_index()
    type_counts.columns = ['Transaction Type', 'Count']
    
    # Create the bar chart
    fig = px.bar(
        type_counts, 
        x='Transaction Type', 
        y='Count', 
        color='Transaction Type',
        title='Distribution of Transaction Types',
        labels={'Count': 'Number of Transactions'},
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Transaction Type',
        yaxis_title='Count',
        legend_title='Transaction Type',
        height=500
    )
    
    return fig

def plot_fraud_distribution(df):
    """
    Plot the distribution of fraudulent vs. non-fraudulent transactions.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        fig: Plotly figure object
    """
    # Count fraudulent transactions
    fraud_counts = df['isFraud'].value_counts().reset_index()
    fraud_counts.columns = ['Is Fraud', 'Count']
    fraud_counts['Is Fraud'] = fraud_counts['Is Fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    # Create the pie chart
    fig = px.pie(
        fraud_counts, 
        values='Count', 
        names='Is Fraud',
        title='Distribution of Fraudulent vs. Non-Fraudulent Transactions',
        color='Is Fraud',
        color_discrete_map={'Legitimate': '#3366CC', 'Fraudulent': '#DC3912'},
        hole=0.4
    )
    
    # Update layout
    fig.update_layout(
        legend_title='Transaction Type',
        height=500
    )
    
    return fig

def plot_fraud_by_type(df):
    """
    Plot the distribution of fraudulent transactions by type.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        fig: Plotly figure object
    """
    # Group by type and fraud
    fraud_by_type = df.groupby(['type', 'isFraud']).size().unstack(fill_value=0)
    fraud_by_type.columns = ['Legitimate', 'Fraudulent']
    fraud_by_type = fraud_by_type.reset_index()
    
    # Calculate percentage of fraudulent transactions
    fraud_by_type['Fraud Percentage'] = (fraud_by_type['Fraudulent'] / 
                                         (fraud_by_type['Legitimate'] + fraud_by_type['Fraudulent'])) * 100
    
    # Create the bar chart
    fig = px.bar(
        fraud_by_type, 
        x='type', 
        y=['Legitimate', 'Fraudulent'],
        title='Distribution of Fraudulent Transactions by Type',
        labels={'value': 'Number of Transactions', 'type': 'Transaction Type', 'variable': 'Transaction Status'},
        barmode='group',
        color_discrete_map={'Legitimate': '#3366CC', 'Fraudulent': '#DC3912'}
    )
    
    # Add a secondary y-axis for percentage
    fig.add_trace(
        go.Scatter(
            x=fraud_by_type['type'],
            y=fraud_by_type['Fraud Percentage'],
            name='Fraud Percentage',
            yaxis='y2',
            line=dict(color='#FF9900', width=3)
        )
    )
    
    # Update layout for dual y-axis
    fig.update_layout(
        xaxis_title='Transaction Type',
        yaxis_title='Number of Transactions',
        legend_title='Transaction Status',
        height=600,
        yaxis2=dict(
            title='Fraud Percentage (%)',
            titlefont=dict(color='#FF9900'),
            tickfont=dict(color='#FF9900'),
            anchor='x',
            overlaying='y',
            side='right'
        )
    )
    
    return fig

def plot_amount_distribution(df):
    """
    Plot the distribution of transaction amounts by fraud status.
    
    Parameters:
        df (pd.DataFrame): Input dataframe
    
    Returns:
        fig: Plotly figure object
    """
    # Separate fraudulent and legitimate transactions
    df_fraud = df[df['isFraud'] == 1]
    df_legit = df[df['isFraud'] == 0].sample(min(len(df_fraud) * 2, len(df_legit)), random_state=42)
    
    # Create histogram
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=df_legit['amount'],
        name='Legitimate',
        opacity=0.75,
        marker=dict(color='#3366CC')
    ))
    
    fig.add_trace(go.Histogram(
        x=df_fraud['amount'],
        name='Fraudulent',
        opacity=0.75,
        marker=dict(color='#DC3912')
    ))
    
    # Update layout
    fig.update_layout(
        title_text='Distribution of Transaction Amounts by Fraud Status',
        xaxis_title_text='Transaction Amount',
        yaxis_title_text='Count',
        bargap=0.2,
        bargroupgap=0.1,
        height=500
    )
    
    return fig

def plot_confusion_matrix(y_true, y_pred):
    """
    Plot confusion matrix for model evaluation.
    
    Parameters:
        y_true (array): True labels
        y_pred (array): Predicted labels
    
    Returns:
        fig: Matplotlib figure object
    """
    cm = confusion_matrix(y_true, y_pred)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the confusion matrix
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        ax=ax,
        cbar=False,
        annot_kws={"size": 16}
    )
    
    # Set labels and title
    ax.set_xlabel('Predicted', fontsize=14)
    ax.set_ylabel('Actual', fontsize=14)
    ax.set_title('Confusion Matrix', fontsize=16)
    
    # Set tick labels
    ax.set_xticklabels(['Legitimate', 'Fraudulent'])
    ax.set_yticklabels(['Legitimate', 'Fraudulent'])
    
    plt.tight_layout()
    
    return fig

def plot_roc_curve(y_true, y_proba):
    """
    Plot ROC curve for model evaluation.
    
    Parameters:
        y_true (array): True labels
        y_proba (array): Predicted probabilities
    
    Returns:
        fig: Matplotlib figure object
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the ROC curve
    ax.plot(
        fpr, 
        tpr, 
        color='darkorange',
        lw=2, 
        label=f'ROC curve (area = {roc_auc:.2f})'
    )
    
    # Plot the random guess line
    ax.plot(
        [0, 1], 
        [0, 1], 
        color='navy', 
        lw=2, 
        linestyle='--'
    )
    
    # Set labels and title
    ax.set_xlabel('False Positive Rate', fontsize=14)
    ax.set_ylabel('True Positive Rate', fontsize=14)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    
    # Set limits and legend
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower right", fontsize=12)
    
    plt.tight_layout()
    
    return fig

def plot_precision_recall_curve(y_true, y_proba):
    """
    Plot precision-recall curve for model evaluation.
    
    Parameters:
        y_true (array): True labels
        y_proba (array): Predicted probabilities
    
    Returns:
        fig: Matplotlib figure object
    """
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = auc(recall, precision)
    
    # Create a figure
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the precision-recall curve
    ax.plot(
        recall, 
        precision, 
        color='darkgreen',
        lw=2, 
        label=f'PR curve (area = {pr_auc:.2f})'
    )
    
    # Set labels and title
    ax.set_xlabel('Recall', fontsize=14)
    ax.set_ylabel('Precision', fontsize=14)
    ax.set_title('Precision-Recall Curve', fontsize=16)
    
    # Set limits and legend
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="lower left", fontsize=12)
    
    plt.tight_layout()
    
    return fig

def plot_feature_importance(model, feature_names):
    """
    Plot feature importance for a trained model.
    
    Parameters:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
    
    Returns:
        fig: Plotly figure object
    """
    # Get feature importances
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        raise ValueError("Model doesn't have feature_importances_ or coef_ attribute")
    
    # Create dataframe
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance
    feature_importance = feature_importance.sort_values('Importance', ascending=False)
    
    # Create the bar chart
    fig = px.bar(
        feature_importance, 
        x='Importance', 
        y='Feature',
        orientation='h',
        title='Feature Importance',
        color='Importance',
        color_continuous_scale='Viridis'
    )
    
    # Update layout
    fig.update_layout(
        xaxis_title='Importance',
        yaxis_title='Feature',
        height=500
    )
    
    return fig
