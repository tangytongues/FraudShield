import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from utils.preprocessing import prepare_features, scale_features
from utils.model_utils import train_test_data_split, train_model
import time

st.set_page_config(
    page_title="Model Building",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Building")

# Check if data is loaded
if 'data' not in st.session_state or 'feature_names' not in st.session_state:
    st.error("Please complete the Feature Engineering step first!")
    st.stop()

# Get the data
df = st.session_state['processed_data']
feature_names = st.session_state['feature_names']

# Define models
models = {
    'logistic_regression': 'Logistic Regression',
    'decision_tree': 'Decision Tree',
    'random_forest': 'Random Forest',
    'svm': 'Support Vector Machine',
    'knn': 'K-Nearest Neighbors'
}

# Model parameters
model_params = {
    'logistic_regression': {
        'C': st.sidebar.slider('C (Regularization)', 0.01, 10.0, 1.0, key='lr_c'),
        'max_iter': st.sidebar.slider('Max Iterations', 100, 1000, 100, 100, key='lr_max_iter')
    },
    'decision_tree': {
        'max_depth': st.sidebar.slider('Max Depth', 3, 20, 10, key='dt_max_depth'),
        'min_samples_split': st.sidebar.slider('Min Samples Split', 2, 20, 2, key='dt_min_samples')
    },
    'random_forest': {
        'n_estimators': st.sidebar.slider('Number of Trees', 10, 200, 100, 10, key='rf_n_estimators'),
        'max_depth': st.sidebar.slider('Max Depth', 3, 20, 10, key='rf_max_depth'),
        'min_samples_split': st.sidebar.slider('Min Samples Split', 2, 20, 2, key='rf_min_samples')
    },
    'svm': {
        'C': st.sidebar.slider('C (SVM)', 0.01, 10.0, 1.0, key='svm_c'),
        'kernel': st.sidebar.selectbox('Kernel', ['linear', 'rbf', 'poly'], key='svm_kernel')
    },
    'knn': {
        'n_neighbors': st.sidebar.slider('Number of Neighbors', 1, 20, 5, key='knn_n_neighbors'),
        'weights': st.sidebar.selectbox('Weights', ['uniform', 'distance'], key='knn_weights')
    }
}

# Header image
st.image("https://images.unsplash.com/photo-1666875758412-5957b60d7969", 
         caption="Machine learning for fraud detection")

# Model Building Section
st.markdown("""
## Building Fraud Detection Models

In this section, we'll train different machine learning models to detect fraudulent transactions.
You can select the model type and adjust parameters to optimize performance.
""")

# Prepare data for modeling
X, y, _ = prepare_features(df)

# Split the data
test_size = st.slider('Test Set Size (%)', 10, 40, 20, key='test_size_slider') / 100
X_train, X_test, y_train, y_test = train_test_data_split(X, y, test_size=test_size)

# Scale the features
X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

# Model selection
st.subheader("Model Selection")
selected_model = st.selectbox('Select a model', list(models.keys()), format_func=lambda x: models[x], key='model_selection')

# Display model parameters
st.subheader("Model Parameters")
st.write(model_params[selected_model])

# Train the model when user clicks the button
if st.button('Train Model', key='train_model_button'):
    with st.spinner('Training model... This may take a moment.'):
        # Record start time
        start_time = time.time()
        
        # Train the model
        model = train_model(
            X_train_scaled, 
            y_train, 
            model_type=selected_model, 
            params=model_params[selected_model]
        )
        
        # Record end time
        end_time = time.time()
        training_time = end_time - start_time
        
        # Save model, data, and parameters to session state
        st.session_state['model'] = model
        st.session_state['X_train'] = X_train
        st.session_state['X_test'] = X_test
        st.session_state['y_train'] = y_train
        st.session_state['y_test'] = y_test
        st.session_state['X_train_scaled'] = X_train_scaled
        st.session_state['X_test_scaled'] = X_test_scaled
        st.session_state['scaler'] = scaler
        st.session_state['model_type'] = selected_model
        st.session_state['model_params'] = model_params[selected_model]
        
        # Display success message
        st.success(f'Model trained successfully in {training_time:.2f} seconds!')
        
        # Show model details
        st.subheader("Model Details")
        
        st.markdown(f"""
        **Model Type:** {models[selected_model]}
        
        **Parameters:**
        """)
        
        for param, value in model_params[selected_model].items():
            st.markdown(f"- **{param}:** {value}")
        
        st.markdown(f"""
        **Training Data:**
        - Training samples: {X_train.shape[0]}
        - Testing samples: {X_test.shape[0]}
        - Features: {X_train.shape[1]}
        
        **Class Distribution:**
        - Training set: {y_train.value_counts().to_dict()}
        - Testing set: {y_test.value_counts().to_dict()}
        """)

# Display model description
st.subheader("About the Selected Model")

model_descriptions = {
    'logistic_regression': """
    **Logistic Regression** is a statistical model that uses a logistic function to model a binary dependent variable. 
    
    **Advantages for Fraud Detection:**
    - Provides probability scores that can be used for threshold-based decisions
    - Highly interpretable, with coefficients indicating feature importance
    - Works well with linearly separable data
    - Computationally efficient for large datasets
    
    **Parameters:**
    - C: Inverse of regularization strength. Smaller values specify stronger regularization.
    - max_iter: Maximum number of iterations for the solver to converge.
    """,
    
    'decision_tree': """
    **Decision Tree** is a non-parametric supervised learning method used for classification and regression.
    
    **Advantages for Fraud Detection:**
    - Can capture non-linear relationships between features
    - Handles feature interactions automatically
    - Provides clear decision rules that can be audited
    - No need for feature scaling
    
    **Parameters:**
    - max_depth: The maximum depth of the tree to prevent overfitting.
    - min_samples_split: The minimum number of samples required to split an internal node.
    """,
    
    'random_forest': """
    **Random Forest** is an ensemble learning method that operates by constructing multiple decision trees.
    
    **Advantages for Fraud Detection:**
    - Reduces overfitting compared to individual decision trees
    - Handles high-dimensional data well
    - Provides feature importance measures
    - Good performance with imbalanced data
    
    **Parameters:**
    - n_estimators: The number of trees in the forest.
    - max_depth: The maximum depth of each tree to prevent overfitting.
    - min_samples_split: The minimum number of samples required to split an internal node.
    """,
    
    'svm': """
    **Support Vector Machine (SVM)** is a supervised learning model that finds the optimal hyperplane that maximizes the margin between classes.
    
    **Advantages for Fraud Detection:**
    - Effective in high-dimensional spaces
    - Works well when there's a clear margin of separation
    - Versatile through different kernel functions
    - Memory efficient as it uses a subset of training points (support vectors)
    
    **Parameters:**
    - C: Regularization parameter. Smaller values specify stronger regularization.
    - kernel: Kernel type to be used in the algorithm (linear, rbf, poly).
    """,
    
    'knn': """
    **K-Nearest Neighbors (KNN)** is a non-parametric method used for classification and regression.
    
    **Advantages for Fraud Detection:**
    - Simple implementation
    - Adapts easily as new training data becomes available
    - No assumptions about the underlying data distribution
    - Works well for multi-class problems
    
    **Parameters:**
    - n_neighbors: Number of neighbors to use for classification.
    - weights: Weight function used in prediction ('uniform' or 'distance').
    """
}

st.markdown(model_descriptions[selected_model])

# Next steps
st.markdown("""
## Next Steps

After training your model, go to the **Model Evaluation** page to assess its performance.
You can visualize various metrics and see how well the model identifies fraudulent transactions.
""")
