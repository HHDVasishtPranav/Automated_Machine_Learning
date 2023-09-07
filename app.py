#importing libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import streamlit as st
import ydata_profiling as ydp
import streamlit_pandas_profiling as st_profile
from streamlit_pandas_profiling import st_profile_report

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.naive_bayes import GaussianNB


def perform_ML(df, target_col, task):
    # Handle categorical features (assuming they are already encoded as integers)
    categorical_features = []  # Add the column names of categorical features here

    # Separate features (X) and target (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Handle missing values (e.g., replace missing values with the mean for numerical features)
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = [col for col in categorical_features if col in X.columns]

    imputer_numeric = SimpleImputer(strategy='mean')
    X[numeric_features] = imputer_numeric.fit_transform(X[numeric_features])

    # Encode categorical features (if not already encoded)
    label_encoders = {}
    for feature in categorical_features:
        le = LabelEncoder()
        X[feature] = le.fit_transform(X[feature])
        label_encoders[feature] = le

    # Standardize numerical features (optional, depending on the choice of models)
    scaler = StandardScaler()
    X[numeric_features] = scaler.fit_transform(X[numeric_features])

    # Split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define metrics for regression and classification
    regression_metrics = {
        'Mean Squared Error': mean_squared_error,
        'Mean Absolute Error': mean_absolute_error,
        'R-squared (R2)': r2_score
    }

    classification_metrics = {
        'Accuracy': accuracy_score,
        'F1 Score': f1_score
    }

    # Define models for regression and classification
    models = {
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(),
        'Random Forest': RandomForestRegressor(),
        'Support Vector Regressor': SVR(),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Neural Network': MLPRegressor()
    } if task == 'regression' else {
        'Logistic Regression': LogisticRegression(max_iter=10000),
        'Decision Tree Classifier': DecisionTreeClassifier(),
        'Random Forest Classifier': RandomForestClassifier(),
        'Support Vector Classifier': SVC(),
        'K-Nearest Neighbors Classifier': KNeighborsClassifier(),
        'Neural Network Classifier': MLPClassifier()
    }

    # Create an empty DataFrame to store results
    results_df = pd.DataFrame(columns=models.keys())

    # Iterate through models
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Initialize a dictionary to store metric values
        metric_values = {}
        
        if task == 'regression':
            for metric_name, metric_func in regression_metrics.items():
                score = metric_func(y_test, y_pred)
                metric_values[metric_name] = score
        elif task == 'classification':
            for metric_name, metric_func in classification_metrics.items():
                score = metric_func(y_test, y_pred)
                metric_values[metric_name] = score
        
        # Add the metric values to the DataFrame
        results_df[model_name] = pd.Series(metric_values)

    # Transpose the DataFrame to have models as rows and metrics as columns
    results_df = results_df.transpose()

    # Display the results DataFrame
    print(results_df)
    st.dataframe(results_df)




with st.sidebar:
    st.title("AutoML")
    navOptions = st.radio("Navigation", ["Home", "EDA", "ML", "Download"])
    st.info("An automated machine learning tool that performs EDA and provides a downloadable model for the uploaded dataset.\n\n~UHHDVasishtPranav😶‍🌫️")

if os.path.exists("uploadedFile.csv"):
    df_main=pd.read_csv("uploadedFile.csv",index_col=None)

if navOptions == "Home":
    #to upload the files
    st.title("Upload your dataset")
    uploadedFile = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploadedFile is not None:
        df_main = pd.read_csv(uploadedFile,index_col=None)
        st.dataframe(df_main)
        df_main.to_csv("uploadedFile.csv", index=False)

if navOptions == "EDA":
    #doing EDA with ydata_profiling(pandas_profiling)
    st.title("EDA")
    profReport= df_main.profile_report()
    st_profile_report(profReport)

if navOptions == "ML":
    st.title("ML models")
    #to select the target column
    target_col = st.selectbox("Select the target column", df_main.columns)
    #to select the reg or classif
    reg_or_classif = st.radio("Regression or Classification", ["Regression", "Classification"])
    if reg_or_classif == "Regression":
        perform_ML(df_main, target_col, task="regression")
    elif reg_or_classif == "Classification":
        perform_ML(df_main, target_col, task="classification")


if navOptions == "Download":
    pass

