import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import os
import pickle
from utils.preprocessing import load_data

# Set page configuration
st.set_page_config(
    page_title="Financial Fraud Detection",
    page_icon="🕵️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main title
st.title("💵 Financial Fraud Detection Dashboard 🕵️‍♂️")

# Display images in a 2x2 grid
col1, col2 = st.columns(2)

with col1:
    st.image("https://images.unsplash.com/photo-1693919653649-27492e78899d", 
             caption="Visualization of fraud detection patterns")
    
with col2:
    st.image("https://images.unsplash.com/photo-1454165804606-c3d57bc86b40", 
             caption="Financial security and monitoring")

# Introduction
st.markdown("""
## About This Project
This application helps detect fraudulent transactions in financial data using machine learning. 
It demonstrates the complete data science workflow from data exploration to model deployment.

### Dataset Overview
The PaySim dataset contains synthetic financial transactions generated to simulate mobile money transactions.
It includes various transaction types such as CASH_IN, CASH_OUT, DEBIT, PAYMENT, and TRANSFER.

### Features
- **Data Exploration**: Analyze transaction patterns and distributions
- **Feature Engineering**: Create meaningful features from raw transaction data
- **Model Building**: Train machine learning models to detect fraud
- **Model Evaluation**: Assess model performance with various metrics
- **Prediction**: Make predictions on new transaction data
""")

st.markdown("---")

# Data sampling explanation
st.subheader("🔄 Sampling the Dataset")
st.markdown("""
The original PaySim dataset is quite large (over 6 million records). 
For better performance in this web application, we're using a sample of the data.

You can adjust the sample size below:
""")

# Sample size selection
sample_size = st.slider("Select sample size:", 
                        min_value=1000, 
                        max_value=100000, 
                        value=50000, 
                        step=1000)

# Load the data when user clicks the button
if st.button("Load Sample Data"):
    with st.spinner('Loading and processing data...'):
        df = load_data(sample_size)
        
        # Save the dataframe to session state for use in other pages
        st.session_state['data'] = df
        
        # Display basic statistics
        st.subheader("Data Overview")
        st.write(f"Loaded {len(df)} records")
        
        # Show head of the dataframe
        st.dataframe(df.head())
        
        # Show basic statistics
        st.subheader("Basic Statistics")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Transactions", f"{len(df):,}")
            st.metric("Fraudulent Transactions", f"{df['isFraud'].sum():,}")
            
        with col2:
            fraud_percentage = (df['isFraud'].sum() / len(df)) * 100
            st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
            st.metric("Transaction Types", f"{df['type'].nunique():,}")
        
        st.success("Data loaded successfully! Navigate to the pages in the sidebar to explore further.")

# Information about next steps
st.markdown("---")
st.subheader("📊 Navigate the Analysis")
st.markdown("""
Use the sidebar to navigate through different sections of the analysis:

1. **Data Exploration**: Visualize and understand the transaction data
2. **Feature Engineering**: Create and select relevant features
3. **Model Building**: Train machine learning models
4. **Model Evaluation**: Evaluate model performance
5. **Prediction**: Make predictions on new data
""")

# Additional image at the bottom
st.image("https://images.unsplash.com/photo-1647885208231-da09bde736a2", 
         caption="Advanced fraud detection visualization")

# Footer
st.markdown("---")
st.markdown("### 🔒 Financial Fraud Detection Project | Created with Streamlit")
