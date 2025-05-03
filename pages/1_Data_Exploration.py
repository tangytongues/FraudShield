import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from utils.visualization import (
    plot_transaction_distribution, 
    plot_fraud_distribution,
    plot_fraud_by_type,
    plot_amount_distribution
)

st.set_page_config(
    page_title="Data Exploration",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Exploration")

# Check if data is loaded
if 'data' not in st.session_state:
    st.error("Please load the data first from the home page!")
    st.stop()

# Get the data
df = st.session_state['data']

# Header image
st.image("https://images.unsplash.com/photo-1666875758412-5957b60d7969", 
         caption="Fraud detection visualization")

# Transaction Data Overview
st.subheader("Transaction Data Overview")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Transactions", f"{len(df):,}")
with col2:
    st.metric("Fraudulent Transactions", f"{df['isFraud'].sum():,}")
with col3:
    fraud_pct = (df['isFraud'].sum() / len(df)) * 100
    st.metric("Fraud Percentage", f"{fraud_pct:.2f}%")

# Display transaction types
st.subheader("Transaction Types")
type_counts = df['type'].value_counts()
fig = plot_transaction_distribution(df)
st.plotly_chart(fig, use_container_width=True)

# Distribution of fraudulent vs non-fraudulent transactions
st.subheader("Fraud Distribution")
col1, col2 = st.columns(2)

with col1:
    fig = plot_fraud_distribution(df)
    st.plotly_chart(fig)

with col2:
    fig = plot_fraud_by_type(df)
    st.plotly_chart(fig)

# Transaction amount distribution
st.subheader("Transaction Amount Analysis")
col1, col2 = st.columns(2)

with col1:
    # Plot distribution of transaction amounts
    fig = plot_amount_distribution(df)
    st.plotly_chart(fig)

with col2:
    # Calculate average amount by transaction type and fraud status
    avg_amount = df.groupby(['type', 'isFraud'])['amount'].mean().reset_index()
    avg_amount.columns = ['Transaction Type', 'Is Fraud', 'Average Amount']
    avg_amount['Is Fraud'] = avg_amount['Is Fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    # Create bar chart
    fig = px.bar(
        avg_amount, 
        x='Transaction Type', 
        y='Average Amount', 
        color='Is Fraud',
        barmode='group',
        title='Average Transaction Amount by Type and Fraud Status',
        color_discrete_map={'Legitimate': '#3366CC', 'Fraudulent': '#DC3912'}
    )
    
    st.plotly_chart(fig)

# Correlation analysis
st.subheader("Correlation Analysis")

# Select only numerical columns
numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
correlation = df[numerical_cols].corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(correlation, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
plt.title('Correlation Matrix of Numerical Features', fontsize=15)
st.pyplot(fig)

# Time series analysis
st.subheader("Transactions Over Time")

# Group by step and count transactions
time_series = df.groupby('step').agg({
    'isFraud': ['count', 'sum']
}).reset_index()
time_series.columns = ['Step', 'Total Transactions', 'Fraudulent Transactions']

# Calculate percentage
time_series['Fraud Percentage'] = (time_series['Fraudulent Transactions'] / 
                                  time_series['Total Transactions']) * 100

# Create dual y-axis plot
fig = go.Figure()

# Add bars for total transactions
fig.add_trace(
    go.Bar(
        x=time_series['Step'],
        y=time_series['Total Transactions'],
        name='Total Transactions',
        marker_color='#3366CC'
    )
)

# Add line for fraudulent transactions
fig.add_trace(
    go.Scatter(
        x=time_series['Step'],
        y=time_series['Fraudulent Transactions'],
        name='Fraudulent Transactions',
        yaxis='y2',
        line=dict(color='#DC3912', width=2)
    )
)

# Add line for fraud percentage
fig.add_trace(
    go.Scatter(
        x=time_series['Step'],
        y=time_series['Fraud Percentage'],
        name='Fraud Percentage (%)',
        yaxis='y3',
        line=dict(color='#FF9900', width=2, dash='dash')
    )
)

# Update layout for triple y-axis
fig.update_layout(
    title='Transaction Activity Over Time (Step = 1 Hour)',
    xaxis=dict(title='Time Step'),
    yaxis=dict(title='Total Transactions', side='left'),
    yaxis2=dict(
        title='Fraudulent Transactions',
        overlaying='y',
        side='right',
        showgrid=False
    ),
    yaxis3=dict(
        title='Fraud Percentage (%)',
        overlaying='y',
        anchor='free',
        position=0.95,
        side='right',
        showgrid=False
    ),
    legend=dict(x=0.01, y=0.99),
    height=600
)

st.plotly_chart(fig, use_container_width=True)

# Exploratory observations
st.subheader("Key Observations")

st.markdown("""
Based on the exploratory data analysis, we can make the following observations:

1. **Fraud Prevalence**: Only a small percentage of transactions are fraudulent, making this an imbalanced classification problem.

2. **Transaction Types**: Fraudulent transactions are primarily concentrated in TRANSFER and CASH_OUT transaction types.

3. **Transaction Amounts**: Fraudulent transactions tend to have different amount distributions compared to legitimate ones.

4. **Account Balances**: There appears to be patterns in how account balances change in fraudulent transactions.

5. **Time Patterns**: There might be temporal patterns in fraudulent activity.

These observations will guide our feature engineering process to create effective predictors for the machine learning models.
""")

# Save important information to session state
st.session_state['fraud_types'] = df[df['isFraud'] == 1]['type'].unique().tolist()
