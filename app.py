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
    pass

if navOptions == "Download":
    with open("best_model.pkl", "rb") as f:
        st.download_button("download as file",f,"trained_bestModel.plk")

