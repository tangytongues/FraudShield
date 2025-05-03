import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, auc
)
from utils.visualization import (
    plot_confusion_matrix, plot_roc_curve, 
    plot_precision_recall_curve, plot_feature_importance
)
from utils.model_utils import evaluate_model, save_model
import pickle
import io
import base64

st.set_page_config(
    page_title="Model Evaluation",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Model Evaluation")

# Check if model is trained
if 'model' not in st.session_state:
    st.error("Please train a model first on the Model Building page!")
    st.stop()

# Get model and data from session state
model = st.session_state['model']
X_train = st.session_state['X_train']
X_test = st.session_state['X_test']
y_train = st.session_state['y_train']
y_test = st.session_state['y_test']
X_train_scaled = st.session_state['X_train_scaled']
X_test_scaled = st.session_state['X_test_scaled']
feature_names = st.session_state['feature_names']
model_type = st.session_state['model_type']
model_params = st.session_state['model_params']

# Header image
st.image("https://images.unsplash.com/photo-1647885208231-da09bde736a2", 
         caption="Evaluating model performance for fraud detection")

# Model Evaluation Section
st.markdown("""
## Evaluating Fraud Detection Model Performance

In this section, we'll assess how well our trained model performs in detecting fraudulent transactions.
We'll examine various metrics and visualizations to understand its strengths and limitations.
""")

# Evaluate the model
with st.spinner('Evaluating model performance...'):
    metrics = evaluate_model(model, X_test_scaled, y_test)
    
    # Make predictions for visualization
    y_pred = model.predict(X_test_scaled)
    
    try:
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    except:
        y_proba = y_pred

# Performance Metrics
st.subheader("Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{metrics['accuracy']:.4f}")
with col2:
    st.metric("Precision", f"{metrics['precision']:.4f}")
with col3:
    st.metric("Recall", f"{metrics['recall']:.4f}")
with col4:
    st.metric("F1 Score", f"{metrics['f1']:.4f}")

st.metric("ROC AUC Score", f"{metrics['roc_auc']:.4f}")

# Classification Report
st.subheader("Classification Report")
st.text(metrics['classification_report'])

# Visualizations
st.subheader("Performance Visualizations")

col1, col2 = st.columns(2)

with col1:
    st.write("Confusion Matrix")
    fig = plot_confusion_matrix(y_test, y_pred)
    st.pyplot(fig)

with col2:
    st.write("ROC Curve")
    fig = plot_roc_curve(y_test, y_proba)
    st.pyplot(fig)

st.subheader("Precision-Recall Analysis")
col1, col2 = st.columns(2)

with col1:
    st.write("Precision-Recall Curve")
    fig = plot_precision_recall_curve(y_test, y_proba)
    st.pyplot(fig)

with col2:
    # Calculate precision and recall at different thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    
    # Create dataframe for thresholds
    threshold_df = pd.DataFrame({
        'Threshold': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    })
    
    # Calculate metrics for each threshold
    for t in threshold_df['Threshold']:
        y_pred_t = (y_proba >= t).astype(int)
        threshold_df.loc[threshold_df['Threshold'] == t, 'Precision'] = precision_score(y_test, y_pred_t)
        threshold_df.loc[threshold_df['Threshold'] == t, 'Recall'] = recall_score(y_test, y_pred_t)
        threshold_df.loc[threshold_df['Threshold'] == t, 'F1 Score'] = f1_score(y_test, y_pred_t)
    
    st.write("Metrics at Different Thresholds")
    st.dataframe(threshold_df.style.highlight_max(axis=0, subset=['Precision', 'Recall', 'F1 Score']))

# Feature Importance
if hasattr(model, 'feature_importances_') or hasattr(model, 'coef_'):
    st.subheader("Feature Importance")
    fig = plot_feature_importance(model, feature_names)
    st.plotly_chart(fig, use_container_width=True)

# Error Analysis
st.subheader("Error Analysis")

# Get predictions and actual values
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred,
    'Probability': y_proba
})

# Calculate error types
predictions_df['Error_Type'] = 'Correct'
predictions_df.loc[(predictions_df['Actual'] == 1) & (predictions_df['Predicted'] == 0), 'Error_Type'] = 'False Negative'
predictions_df.loc[(predictions_df['Actual'] == 0) & (predictions_df['Predicted'] == 1), 'Error_Type'] = 'False Positive'

# Add the original data features
error_analysis_df = pd.concat([X_test.reset_index(drop=True), predictions_df], axis=1)

# Show error distribution
error_counts = predictions_df['Error_Type'].value_counts()
col1, col2 = st.columns(2)

with col1:
    # Create pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(
        error_counts, 
        labels=error_counts.index, 
        autopct='%1.1f%%',
        colors=['#4CAF50', '#F44336', '#FFC107'],
        startangle=90
    )
    ax.axis('equal')
    plt.title('Distribution of Prediction Results', fontsize=16)
    st.pyplot(fig)

with col2:
    # Show incorrect prediction examples
    st.write("Examples of Incorrect Predictions")
    
    # Filter for incorrect predictions
    incorrect_df = error_analysis_df[error_analysis_df['Error_Type'] != 'Correct']
    
    if not incorrect_df.empty:
        st.dataframe(incorrect_df.head(10))
    else:
        st.write("No incorrect predictions in the sample!")

# Model Download
st.subheader("Download Trained Model")
st.markdown("""
You can download the trained model for use in other applications or for further analysis.
The model is saved in pickle format.
""")

# Save model to download
model_download = save_model(model, f"fraud_detection_{model_type}_model")
st.markdown(model_download, unsafe_allow_html=True)

# Generate a report
st.subheader("Download Performance Report")

report = f"""
# Fraud Detection Model Performance Report

## Model Information
- Model Type: {model_type}
- Parameters: {model_params}

## Dataset Information
- Training samples: {X_train.shape[0]}
- Testing samples: {X_test.shape[0]}
- Features: {X_train.shape[1]}
- Feature names: {feature_names}

## Performance Metrics
- Accuracy: {metrics['accuracy']:.4f}
- Precision: {metrics['precision']:.4f}
- Recall: {metrics['recall']:.4f}
- F1 Score: {metrics['f1']:.4f}
- ROC AUC: {metrics['roc_auc']:.4f}

## Classification Report
{metrics['classification_report']}

## Error Analysis
- Correct Predictions: {error_counts.get('Correct', 0)}
- False Positives: {error_counts.get('False Positive', 0)}
- False Negatives: {error_counts.get('False Negative', 0)}
"""

# Create a download link for the report
def create_download_link(content, filename, link_text):
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

report_download = create_download_link(report, "fraud_detection_report.md", "Download Report")
st.markdown(report_download, unsafe_allow_html=True)

# Guidance on interpreting results
st.subheader("Interpreting Results")
st.markdown("""
### Key Considerations in Fraud Detection Models:

1. **Precision vs. Recall Trade-off**:
   - **Precision**: The percentage of transactions flagged as fraud that are actually fraudulent
   - **Recall**: The percentage of actual fraudulent transactions that were correctly identified
   - Adjust threshold based on business priorities (minimizing false positives vs. catching more fraud)

2. **Cost Analysis**:
   - False Negatives: Missed fraudulent transactions (direct financial loss)
   - False Positives: Legitimate transactions incorrectly flagged (customer friction, review costs)

3. **Model Limitations**:
   - The model is only as good as the data it was trained on
   - Fraudsters adapt their techniques over time (model drift)
   - Consider implementing continuous monitoring and retraining

4. **Next Steps**:
   - Try the Prediction page to test the model on new transaction data
   - Consider ensemble models for improved performance
   - Implement threshold tuning based on business objectives
""")
