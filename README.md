# Automated Machine Learning (AutoML) Tool

Welcome to the Automated Machine Learning (AutoML) tool! This tool is designed to streamline the process of Exploratory Data Analysis (EDA) and model training for your dataset. It offers a user-friendly interface for uploading your data, performing EDA, and training machine learning models for both regression and classification tasks.

## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Machine Learning (ML) Models](#machine-learning-ml-models)
- [Acknowledgements](#acknowledgements)

## Introduction<a name="introduction"></a>

This AutoML tool is built using Python and various libraries, including Streamlit, ydata_profiling, Streamlit Pandas Profiling, scikit-learn (sklearn), and more. It automates several key steps in the data analysis and machine learning process, making it easier to gain insights from your data and train predictive models.

## Getting Started<a name="getting-started"></a>

To use this tool, follow these simple steps:

1. **Upload Your Dataset**:
   - Click on the "Upload your dataset" option in the sidebar.
   - Choose a CSV file containing your dataset.
   - The tool will load your data for analysis.

2. **Exploratory Data Analysis (EDA)**:
   - Select the "EDA" option in the sidebar.
   - Explore your dataset using ydata_profiling (Pandas Profiling).
   - Gain insights into data distribution, missing values, and more.

3. **Machine Learning Models**:
   - Choose the "ML models" option in the sidebar.
   - Select the target column for prediction.
   - Specify whether you want to perform regression or classification.

4. **Model Training and Evaluation**:
   - The tool will automatically preprocess your data, split it into training and testing sets, and train multiple machine learning models.
   - You'll see a table with evaluation metrics for each model, depending on your task (regression or classification).
   - The best-performing model will be saved for download.

## Exploratory Data Analysis (EDA)<a name="exploratory-data-analysis-eda"></a>

The EDA section of this tool leverages the power of ydata_profiling (Pandas Profiling) to provide you with a comprehensive report on your dataset. You can explore statistics, data types, missing values, and visualizations to better understand your data.

## Machine Learning (ML) Models<a name="machine-learning-ml-models"></a>

In the ML models section, you can train and evaluate various machine learning models based on your chosen task (regression or classification). The tool performs the following steps:

- Handles categorical features and missing values.
- Encodes categorical features.
- Standardizes numerical features (optional).
- Splits the data into training and testing sets.
- Trains multiple models and evaluates their performance.

You can download the best-performing model for your task.

## Acknowledgements<a name="acknowledgements"></a>

This AutoML tool was developed with inspiration from Nicholas Renotte's YouTube video and relies on various libraries and packages, including Streamlit, ydata_profiling, Streamlit Pandas Profiling, joblib, scikit-learn, pandas, numpy, and more.

Feel free to explore your data and train machine learning models effortlessly with this AutoML tool! If you encounter any issues or have suggestions for improvements, please don't hesitate to reach out.