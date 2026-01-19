# AI POWERED MEDICAL DIAGNOSIS SYSTEM:
This project is an AI-based medical diagnosis system that predicts the likelihood of diabetes based on patient health data. The system uses a machine learning model trained on the Pima Indians Diabetes dataset and provides predictions through a user-friendly web application built with Streamlit.
# Features:
>Predicts diabetes based on patient symptoms and health metrics

>Provides a simple web interface using Streamlit

>Trained using Decision Tree Classifier

>Displays predictions in an easy-to-understand format (Low Risk / High Risk)

>Model can be extended to other diseases in the future

# Dataset:
* dataset: Used:Pima Indians Diabetes Dataset
* source: Kaggle/UCI Machine Learning Repository
* features Included:
>pregnencies
>Glucose
>Blood Pressure
>Skin Thickness
>Insulin
>BMI
>Diabetes Pedigree Function
>Age
*  Target :Outcome(0=no Diabetes,1=Diabetes)

# Requirements:
* Python 3.9
* libraries:
>pandas
>numpy
>scikit-learn
>streamlit
>joblib

# Job Structure:

medical_diagnosis_system/
|
|-data/
|    |---diabetes.csv
|
|-model/
|    |----trained_model.pkl
|-train_model.py
|-app.py
|-requirements.txt
|-README.md

# Steps:
1. py -3.9 train_model.py

2. py -3.9 -m streamlit run app.py --server
headless true

# How it works:
>The user enters patient health metrics in the app

>The app sends the input to the trained ML model

>The model predicts 0 (Low Risk) or 1 (High Risk) of diabetes

>The result is displayed on the web interface


# Author:
developed by shridhar