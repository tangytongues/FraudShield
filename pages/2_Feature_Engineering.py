import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils.preprocessing import prepare_features

st.set_page_config(
    page_title="Feature Engineering",
    page_icon="🛠️",
    layout="wide"
)

st.title("🛠️ Feature Engineering")

# Check if data is loaded
if 'data' not in st.session_state:
    st.error("Please load the data first from the home page!")
    st.stop()

# Get the data
df = st.session_state['data']

# Header image
st.image("https://images.unsplash.com/photo-1451187580459-43490279c0fa", 
         caption="Feature engineering for fraud detection")

# Introduction to Feature Engineering
st.markdown("""
## Feature Engineering for Fraud Detection

Feature engineering is a critical step in building effective fraud detection models. 
We'll create new features based on transaction patterns and behavior to help the model identify suspicious activities.
""")

# Display original features
st.subheader("Original Features")
st.dataframe(df.head())

# Show feature descriptions
st.subheader("Original Feature Descriptions")
original_features = {
    'step': 'Time step (1 step = 1 hour)',
    'type': 'Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)',
    'amount': 'Transaction amount',
    'nameOrig': 'Customer who initiated the transaction',
    'oldbalanceOrg': 'Customer balance before the transaction',
    'newbalanceOrig': 'Customer balance after the transaction',
    'nameDest': 'Recipient of the transaction',
    'oldbalanceDest': 'Recipient balance before the transaction',
    'newbalanceDest': 'Recipient balance after the transaction',
    'isFraud': 'Fraudulent transaction flag (target variable)',
    'isFlaggedFraud': 'System-flagged fraudulent transaction'
}

feature_df = pd.DataFrame({
    'Feature': original_features.keys(),
    'Description': original_features.values()
})
st.table(feature_df)

# Engineered Features
st.subheader("Engineered Features")

st.markdown("""
Based on our exploratory analysis, we will create the following new features:

1. **Customer Type**: Extract customer type from name (C for customer, M for merchant)
2. **Destination Type**: Extract destination type from name
3. **Origin Balance Difference**: The difference between new and old balance for origin account
4. **Destination Balance Difference**: The difference between new and old balance for destination account
5. **Is Account Emptied**: Flag if origin account was emptied in the transaction
6. **Amount Exceeds Balance**: Flag if transaction amount exceeds origin account balance
7. **Type Encoded**: Numerical encoding of transaction type
""")

# Create the engineered features
# Extract the customer and merchant type from the name
df['customer_type'] = df['nameOrig'].str[0]
df['dest_type'] = df['nameDest'].str[0]

# Calculate transaction balance differences
df['orig_balance_diff'] = df['newbalanceOrig'] - df['oldbalanceOrg']
df['dest_balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']

# Create a flag for transactions where the origin account was emptied
df['is_orig_acct_emptied'] = ((df['oldbalanceOrg'] > 0) & (df['newbalanceOrig'] == 0)).astype(int)

# Create a flag for transactions where the amount exceeds the original balance
df['amount_exceeds_balance'] = (df['amount'] > df['oldbalanceOrg']).astype(int)

# Encode transaction type 
transaction_types = {'CASH_IN': 0, 'CASH_OUT': 1, 'DEBIT': 2, 'PAYMENT': 3, 'TRANSFER': 4}
df['type_encoded'] = df['type'].map(transaction_types)

# Display the engineered features
st.dataframe(df[['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                'oldbalanceDest', 'newbalanceDest',
                'customer_type', 'dest_type', 'orig_balance_diff', 
                'dest_balance_diff', 'is_orig_acct_emptied', 
                'amount_exceeds_balance', 'type_encoded', 'isFraud']].head(10))

# Feature Visualization
st.subheader("Feature Visualization")

col1, col2 = st.columns(2)

with col1:
    # Distribution of account emptied flag by fraud status
    emptied_by_fraud = df.groupby(['is_orig_acct_emptied', 'isFraud']).size().reset_index()
    emptied_by_fraud.columns = ['Account Emptied', 'Is Fraud', 'Count']
    emptied_by_fraud['Account Emptied'] = emptied_by_fraud['Account Emptied'].map({0: 'Not Emptied', 1: 'Emptied'})
    emptied_by_fraud['Is Fraud'] = emptied_by_fraud['Is Fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig = px.bar(
        emptied_by_fraud, 
        x='Account Emptied', 
        y='Count', 
        color='Is Fraud',
        title='Account Emptied Flag by Fraud Status',
        color_discrete_map={'Legitimate': '#3366CC', 'Fraudulent': '#DC3912'},
        barmode='group'
    )
    
    st.plotly_chart(fig)

with col2:
    # Distribution of amount exceeds balance flag by fraud status
    exceeds_by_fraud = df.groupby(['amount_exceeds_balance', 'isFraud']).size().reset_index()
    exceeds_by_fraud.columns = ['Amount Exceeds Balance', 'Is Fraud', 'Count']
    exceeds_by_fraud['Amount Exceeds Balance'] = exceeds_by_fraud['Amount Exceeds Balance'].map({0: 'Within Balance', 1: 'Exceeds Balance'})
    exceeds_by_fraud['Is Fraud'] = exceeds_by_fraud['Is Fraud'].map({0: 'Legitimate', 1: 'Fraudulent'})
    
    fig = px.bar(
        exceeds_by_fraud, 
        x='Amount Exceeds Balance', 
        y='Count', 
        color='Is Fraud',
        title='Amount Exceeds Balance Flag by Fraud Status',
        color_discrete_map={'Legitimate': '#3366CC', 'Fraudulent': '#DC3912'},
        barmode='group'
    )
    
    st.plotly_chart(fig)

# Balance difference visualization
st.subheader("Balance Differences Analysis")

col1, col2 = st.columns(2)

with col1:
    # Create sample to avoid overcrowding
    df_sample = df.sample(min(10000, len(df)), random_state=42)
    
    # Origin balance difference
    fig = px.scatter(
        df_sample, 
        x='amount', 
        y='orig_balance_diff',
        color='isFraud',
        color_discrete_map={0: '#3366CC', 1: '#DC3912'},
        title='Transaction Amount vs. Origin Balance Difference',
        labels={'amount': 'Transaction Amount', 'orig_balance_diff': 'Origin Balance Difference', 'isFraud': 'Is Fraud'},
        opacity=0.7
    )
    
    # Add reference line
    fig.add_shape(
        type="line",
        line=dict(dash="dash", color="gray"),
        x0=df_sample['amount'].min(),
        y0=-df_sample['amount'].min(),
        x1=df_sample['amount'].max(),
        y1=-df_sample['amount'].max()
    )
    
    st.plotly_chart(fig)

with col2:
    # Destination balance difference
    fig = px.scatter(
        df_sample, 
        x='amount', 
        y='dest_balance_diff',
        color='isFraud',
        color_discrete_map={0: '#3366CC', 1: '#DC3912'},
        title='Transaction Amount vs. Destination Balance Difference',
        labels={'amount': 'Transaction Amount', 'dest_balance_diff': 'Destination Balance Difference', 'isFraud': 'Is Fraud'},
        opacity=0.7
    )
    
    # Add reference line
    fig.add_shape(
        type="line",
        line=dict(dash="dash", color="gray"),
        x0=df_sample['amount'].min(),
        y0=df_sample['amount'].min(),
        x1=df_sample['amount'].max(),
        y1=df_sample['amount'].max()
    )
    
    st.plotly_chart(fig)

# Feature Importance Analysis
st.subheader("Feature Analysis for Fraud Detection")

# Prepare features for analysis
X, y, feature_names = prepare_features(df)

# Calculate the mean values of features for fraudulent and legitimate transactions
fraud_means = X[y == 1].mean()
legit_means = X[y == 0].mean()

# Calculate the difference
difference = fraud_means - legit_means

# Create a dataframe for the difference
diff_df = pd.DataFrame({
    'Feature': feature_names,
    'Difference': difference.values
})

# Sort by absolute difference
diff_df['Abs_Difference'] = diff_df['Difference'].abs()
diff_df = diff_df.sort_values('Abs_Difference', ascending=False)

# Plot the differences
fig = px.bar(
    diff_df, 
    x='Feature', 
    y='Difference',
    title='Difference in Feature Means (Fraud - Legitimate)',
    color='Difference',
    color_continuous_scale='RdBu_r'
)

st.plotly_chart(fig, use_container_width=True)

# Feature Selection
st.subheader("Feature Selection for Modeling")

st.markdown("""
Based on our analysis, we will select the following features for our model:

1. **amount**: Transaction amount
2. **oldbalanceOrg**: Initial balance before the transaction
3. **newbalanceOrig**: New balance after the transaction
4. **oldbalanceDest**: Initial balance recipient before the transaction
5. **newbalanceDest**: New balance recipient after the transaction
6. **orig_balance_diff**: The difference between new and old balance for origin account
7. **dest_balance_diff**: The difference between new and old balance for destination account
8. **type_encoded**: Numerical encoding of transaction type
9. **is_orig_acct_emptied**: Flag if origin account was emptied in the transaction
10. **amount_exceeds_balance**: Flag if transaction amount exceeds origin account balance

These features capture the key patterns we observed in fraudulent transactions.
""")

# Save the processed dataframe and feature names to session state
st.session_state['processed_data'] = df
st.session_state['feature_names'] = feature_names
