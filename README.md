# Fraud Detection System

This is a Streamlit-based fraud detection application that processes financial transaction data, builds machine learning models, and provides interactive visualizations for identifying fraudulent activities.

## Setup Instructions

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Installation

1. Clone or download this repository to your local machine

2. Open a terminal or command prompt and navigate to the project directory

3. Install the required packages:

```bash
pip install streamlit pandas numpy matplotlib seaborn plotly scikit-learn pillow
```

### Running the Application

1. Navigate to the project directory in your terminal

2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. The application will start and open in your default web browser. If it doesn't open automatically, visit http://localhost:8501 in your browser.

## Application Structure

The application is structured as follows:

- **app.py**: Main entry point of the application
- **utils/**: Utility modules for data processing and modeling
  - **preprocessing.py**: Functions for data loading and preprocessing
  - **visualization.py**: Visualization utilities
  - **model_utils.py**: Machine learning model utilities
- **pages/**: Individual pages of the Streamlit application
  - **1_Data_Exploration.py**: Data exploration and visualization
  - **2_Feature_Engineering.py**: Feature creation and selection
  - **3_Model_Building.py**: Model training interface
  - **4_Model_Evaluation.py**: Model evaluation and metrics
  - **5_Prediction.py**: Making predictions with trained models
- **data/**: Directory for storing data files

## Using the Application

1. Start with the Data Exploration page to understand the dataset
2. Use Feature Engineering to prepare data for modeling
3. Build and train models on the Model Building page
4. Evaluate model performance on the Model Evaluation page
5. Make predictions on new data using the Prediction page

## Customization

You can modify the application by:

- Editing the utility functions in the `utils/` directory
- Adding new visualization functions in `utils/visualization.py`
- Creating new machine learning models in `utils/model_utils.py`
- Adding new pages to the application by creating new files in the `pages/` directory
