import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from utils.model_utils import predict_transaction
import json
import base64

st.set_page_config(
    page_title="Fraud Prediction",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Fraud Prediction Tool")

# Check if model is trained
if 'model' not in st.session_state:
    st.error("Please train a model first on the Model Building page!")
    st.stop()

# Get model and data from session state
model = st.session_state['model']
scaler = st.session_state['scaler']
feature_names = st.session_state['feature_names']
orig_data = st.session_state['data']

# Header image
st.image("https://images.unsplash.com/photo-1454165804606-c3d57bc86b40", 
         caption="Financial security monitoring and prediction")

# Introduction
st.markdown("""
## Predict Fraudulent Transactions

Use this tool to predict whether a transaction is fraudulent or legitimate.
You can either:
1. Input a single transaction manually
2. Upload a CSV file of transactions
3. Select a random transaction from the dataset
""")

# Create tabs for different input methods
tabs = st.tabs(["Single Transaction", "Batch Prediction", "Random Sample"])

# Tab 1: Single Transaction
with tabs[0]:
    st.subheader("Enter Transaction Details")

    # Create form for transaction input
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Basic transaction details
            transaction_type = st.selectbox(
                "Transaction Type", 
                ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"],
                index=4
            )
            
            amount = st.number_input(
                "Transaction Amount", 
                min_value=0.0, 
                max_value=1000000.0, 
                value=100000.0,
                step=1000.0
            )
            
            # Origin account details
            st.markdown("### Origin Account")
            old_balance_orig = st.number_input(
                "Original Balance (Origin)", 
                min_value=0.0, 
                max_value=1000000.0, 
                value=100000.0,
                step=1000.0
            )
            
            new_balance_orig = st.number_input(
                "New Balance (Origin)", 
                min_value=0.0, 
                max_value=1000000.0, 
                value=0.0,
                step=1000.0
            )
        
        with col2:
            # Destination account details
            st.markdown("### Destination Account")
            old_balance_dest = st.number_input(
                "Original Balance (Destination)", 
                min_value=0.0, 
                max_value=1000000.0, 
                value=0.0,
                step=1000.0
            )
            
            new_balance_dest = st.number_input(
                "New Balance (Destination)", 
                min_value=0.0, 
                max_value=1000000.0, 
                value=0.0,
                step=1000.0
            )
        
        # Submit button
        predict_button = st.form_submit_button("Predict")
    
    # Process the prediction when form is submitted
    if predict_button:
        # Map transaction type to encoded value
        type_mapping = {
            'CASH_IN': 0, 
            'CASH_OUT': 1, 
            'DEBIT': 2, 
            'PAYMENT': 3, 
            'TRANSFER': 4
        }
        type_encoded = type_mapping[transaction_type]
        
        # Calculate additional features
        orig_balance_diff = new_balance_orig - old_balance_orig
        dest_balance_diff = new_balance_dest - old_balance_dest
        is_orig_acct_emptied = 1 if (old_balance_orig > 0 and new_balance_orig == 0) else 0
        amount_exceeds_balance = 1 if amount > old_balance_orig else 0
        
        # Create transaction dictionary
        transaction = {
            'amount': amount,
            'oldbalanceOrg': old_balance_orig,
            'newbalanceOrig': new_balance_orig,
            'oldbalanceDest': old_balance_dest,
            'newbalanceDest': new_balance_dest,
            'orig_balance_diff': orig_balance_diff,
            'dest_balance_diff': dest_balance_diff,
            'type_encoded': type_encoded,
            'is_orig_acct_emptied': is_orig_acct_emptied,
            'amount_exceeds_balance': amount_exceeds_balance
        }
        
        # Make prediction
        prediction, probability = predict_transaction(model, transaction, feature_names, scaler)
        
        # Display prediction result
        st.markdown("---")
        st.subheader("Prediction Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 1:
                st.error("⚠️ Fraudulent Transaction Detected")
            else:
                st.success("✅ Transaction Appears Legitimate")
                
            st.metric("Fraud Probability", f"{probability:.4f}")
        
        with col2:
            # Create gauge chart for probability
            fig = px.pie(
                values=[probability, 1-probability],
                names=['Fraud Risk', 'Safe'],
                hole=0.7,
                color_discrete_sequence=['#F44336', '#4CAF50'],
                title="Fraud Risk Assessment"
            )
            
            # Add annotation in the center
            fig.update_layout(
                annotations=[dict(
                    text=f"{probability:.1%}",
                    x=0.5, y=0.5,
                    font_size=24,
                    showarrow=False
                )]
            )
            
            st.plotly_chart(fig)
        
        # Transaction summary
        st.subheader("Transaction Summary")
        
        summary_df = pd.DataFrame({
            'Feature': ['Transaction Type', 'Amount', 'Origin Account Emptied', 'Amount Exceeds Balance',
                       'Origin Balance Before', 'Origin Balance After', 'Destination Balance Before', 'Destination Balance After'],
            'Value': [transaction_type, f"${amount:,.2f}", 
                     "Yes" if is_orig_acct_emptied else "No", 
                     "Yes" if amount_exceeds_balance else "No",
                     f"${old_balance_orig:,.2f}", f"${new_balance_orig:,.2f}", 
                     f"${old_balance_dest:,.2f}", f"${new_balance_dest:,.2f}"]
        })
        
        st.table(summary_df)
        
        # Risk factors
        st.subheader("Risk Factors Analysis")
        
        risk_factors = []
        if transaction_type in ['CASH_OUT', 'TRANSFER']:
            risk_factors.append("Transaction type is commonly associated with fraud")
        
        if is_orig_acct_emptied:
            risk_factors.append("Origin account was emptied in this transaction")
        
        if amount_exceeds_balance:
            risk_factors.append("Transaction amount exceeds original balance")
        
        if old_balance_orig > 0 and abs(orig_balance_diff + amount) < 0.1:
            risk_factors.append("Transaction amount matches balance reduction")
        
        if dest_balance_diff == 0 and old_balance_dest > 0:
            risk_factors.append("Destination balance didn't change despite transaction")
        
        if not risk_factors:
            risk_factors.append("No specific risk factors identified")
        
        for factor in risk_factors:
            if factor == "No specific risk factors identified":
                st.info(f"• {factor}")
            else:
                st.warning(f"• {factor}")

# Tab 2: Batch Prediction
with tabs[1]:
    st.subheader("Upload Transactions for Batch Prediction")
    
    st.markdown("""
    Upload a CSV file with transaction data to make predictions on multiple transactions.
    
    The CSV should include the following columns:
    - amount: Transaction amount
    - type: Transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
    - oldbalanceOrg: Original balance of origin account
    - newbalanceOrig: New balance of origin account
    - oldbalanceDest: Original balance of destination account
    - newbalanceDest: New balance of destination account
    """)
    
    # Sample CSV template
    sample_data = {
        'type': ['TRANSFER', 'CASH_OUT', 'PAYMENT'], 
        'amount': [10000, 5000, 3000],
        'oldbalanceOrg': [10000, 5000, 3000], 
        'newbalanceOrig': [0, 0, 0],
        'oldbalanceDest': [0, 0, 0], 
        'newbalanceDest': [0, 0, 0]
    }
    
    sample_df = pd.DataFrame(sample_data)
    
    # Create download link for sample template
    def convert_df_to_csv_download_link(df, filename, link_text):
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
        return href
    
    st.markdown(convert_df_to_csv_download_link(sample_df, "transaction_template.csv", 
                                               "Download Sample CSV Template"), 
                unsafe_allow_html=True)
    
    # File uploader
    uploaded_file = st.file_uploader("Upload transaction CSV", type="csv")
    
    if uploaded_file is not None:
        # Load the uploaded data
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("Data Preview:")
            st.dataframe(batch_df.head())
            
            # Check if required columns exist
            required_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
                            'oldbalanceDest', 'newbalanceDest']
            
            missing_cols = [col for col in required_cols if col not in batch_df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                # Preprocess batch data
                batch_df = batch_df.copy()
                
                # Calculate additional features
                batch_df['orig_balance_diff'] = batch_df['newbalanceOrig'] - batch_df['oldbalanceOrg']
                batch_df['dest_balance_diff'] = batch_df['newbalanceDest'] - batch_df['oldbalanceDest']
                batch_df['is_orig_acct_emptied'] = ((batch_df['oldbalanceOrg'] > 0) & 
                                                    (batch_df['newbalanceOrig'] == 0)).astype(int)
                batch_df['amount_exceeds_balance'] = (batch_df['amount'] > batch_df['oldbalanceOrg']).astype(int)
                
                # Encode transaction type
                type_mapping = {
                    'CASH_IN': 0, 
                    'CASH_OUT': 1, 
                    'DEBIT': 2, 
                    'PAYMENT': 3, 
                    'TRANSFER': 4
                }
                batch_df['type_encoded'] = batch_df['type'].map(type_mapping)
                
                # Make predictions
                if st.button("Run Batch Prediction"):
                    with st.spinner("Processing batch predictions..."):
                        # Prepare features
                        X_batch = batch_df[feature_names]
                        
                        # Scale features
                        X_batch_scaled = scaler.transform(X_batch)
                        
                        # Predict
                        batch_df['fraud_prediction'] = model.predict(X_batch_scaled)
                        
                        try:
                            batch_df['fraud_probability'] = model.predict_proba(X_batch_scaled)[:, 1]
                        except:
                            batch_df['fraud_probability'] = batch_df['fraud_prediction']
                        
                        # Display results
                        st.subheader("Prediction Results")
                        
                        # Results summary
                        frauds_count = batch_df['fraud_prediction'].sum()
                        total_count = len(batch_df)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Total Transactions", total_count)
                            st.metric("Fraudulent Transactions", frauds_count)
                        
                        with col2:
                            fraud_percentage = (frauds_count / total_count) * 100
                            st.metric("Fraud Percentage", f"{fraud_percentage:.2f}%")
                            
                            avg_probability = batch_df['fraud_probability'].mean() * 100
                            st.metric("Average Fraud Probability", f"{avg_probability:.2f}%")
                        
                        # Create visualization
                        fig = px.histogram(
                            batch_df, 
                            x='fraud_probability',
                            color='fraud_prediction',
                            labels={'fraud_probability': 'Fraud Probability', 'fraud_prediction': 'Fraud Detected'},
                            color_discrete_map={0: '#4CAF50', 1: '#F44336'},
                            title='Distribution of Fraud Probabilities',
                            nbins=20
                        )
                        
                        st.plotly_chart(fig)
                        
                        # Show the results table
                        results_df = batch_df.copy()
                        results_df['fraud_prediction'] = results_df['fraud_prediction'].map({0: 'Legitimate', 1: 'Fraudulent'})
                        results_df['fraud_probability'] = results_df['fraud_probability'].apply(lambda x: f"{x:.4f}")
                        
                        st.dataframe(results_df)
                        
                        # Create download link for results
                        st.subheader("Download Results")
                        results_link = convert_df_to_csv_download_link(results_df, "fraud_prediction_results.csv", 
                                                                      "Download Results CSV")
                        st.markdown(results_link, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Tab 3: Random Sample
with tabs[2]:
    st.subheader("Select Random Transaction from Dataset")
    
    st.markdown("""
    Select a random transaction from the dataset to test the model's prediction.
    You can choose to see a known fraudulent transaction or a legitimate one.
    """)
    
    # Transaction type selection
    transaction_category = st.radio(
        "Transaction Category",
        ["Any Transaction", "Known Fraudulent", "Known Legitimate"]
    )
    
    # Filter data based on selection
    if transaction_category == "Known Fraudulent":
        filtered_data = orig_data[orig_data['isFraud'] == 1]
        if len(filtered_data) == 0:
            st.warning("No fraudulent transactions in the dataset!")
            st.stop()
    elif transaction_category == "Known Legitimate":
        filtered_data = orig_data[orig_data['isFraud'] == 0]
    else:
        filtered_data = orig_data
    
    # Button to select random transaction
    if st.button("Select Random Transaction"):
        # Get a random transaction
        random_index = np.random.randint(0, len(filtered_data))
        transaction = filtered_data.iloc[random_index]
        
        # Display transaction details
        st.subheader("Transaction Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Basic Information")
            st.markdown(f"**Transaction Type:** {transaction['type']}")
            st.markdown(f"**Amount:** ${transaction['amount']:,.2f}")
            st.markdown(f"**Origin Account:** {transaction['nameOrig']}")
            st.markdown(f"**Destination Account:** {transaction['nameDest']}")
            
        with col2:
            st.markdown("### Account Balances")
            st.markdown(f"**Origin Before:** ${transaction['oldbalanceOrg']:,.2f}")
            st.markdown(f"**Origin After:** ${transaction['newbalanceOrig']:,.2f}")
            st.markdown(f"**Destination Before:** ${transaction['oldbalanceDest']:,.2f}")
            st.markdown(f"**Destination After:** ${transaction['newbalanceDest']:,.2f}")
        
        # Prepare transaction for prediction
        pred_transaction = {
            'amount': transaction['amount'],
            'oldbalanceOrg': transaction['oldbalanceOrg'],
            'newbalanceOrig': transaction['newbalanceOrig'],
            'oldbalanceDest': transaction['oldbalanceDest'],
            'newbalanceDest': transaction['newbalanceDest'],
            'orig_balance_diff': transaction['orig_balance_diff'],
            'dest_balance_diff': transaction['dest_balance_diff'],
            'type_encoded': transaction['type_encoded'],
            'is_orig_acct_emptied': transaction['is_orig_acct_emptied'],
            'amount_exceeds_balance': transaction['amount_exceeds_balance']
        }
        
        # Make prediction
        prediction, probability = predict_transaction(model, pred_transaction, feature_names, scaler)
        
        # Display actual status and prediction
        st.markdown("---")
        st.subheader("Actual vs Predicted")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### Actual Status")
            if transaction['isFraud'] == 1:
                st.error("⚠️ Fraudulent Transaction")
            else:
                st.success("✅ Legitimate Transaction")
            
        with col2:
            st.markdown("### Model Prediction")
            if prediction == 1:
                st.error("⚠️ Predicted as Fraud")
            else:
                st.success("✅ Predicted as Legitimate")
            
            st.metric("Fraud Probability", f"{probability:.4f}")
            
        with col3:
            st.markdown("### Prediction Result")
            if prediction == transaction['isFraud']:
                st.success("✓ Correct Prediction")
            else:
                st.error("✗ Incorrect Prediction")
                
            if prediction == 1 and transaction['isFraud'] == 0:
                st.warning("False Positive")
            elif prediction == 0 and transaction['isFraud'] == 1:
                st.warning("False Negative")
        
        # Transaction visualization
        st.subheader("Transaction Visualization")
        
        # Create Sankey diagram
        labels = ["Origin", "Transaction", "Destination"]
        source = [0, 1]
        target = [1, 2]
        values = [transaction['amount'], transaction['amount']]
        
        # Add colors based on fraud status
        color = '#F44336' if transaction['isFraud'] == 1 else '#4CAF50'
        
        fig = px.chord(
            names=labels,
            source=source,
            target=target,
            values=values,
            color_discrete_sequence=[color],
            title=f"{transaction['type']} Transaction Flow: ${transaction['amount']:,.2f}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature importance for this prediction
        if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
            st.subheader("Local Feature Importance")
            st.markdown("The importance of each feature for this specific prediction:")
            
            # Get feature values
            feature_values = pd.DataFrame({
                'Feature': feature_names,
                'Value': [pred_transaction[f] for f in feature_names]
            })
            
            # Get feature importance from model
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
                
            # Create dataframe with local importance
            feature_values['Importance'] = importance
            feature_values['Local_Impact'] = feature_values['Value'] * feature_values['Importance']
            feature_values = feature_values.sort_values('Local_Impact', ascending=False)
            
            # Plot
            fig = px.bar(
                feature_values, 
                x='Local_Impact', 
                y='Feature',
                orientation='h',
                title='Feature Impact on This Prediction',
                color='Local_Impact',
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig)
