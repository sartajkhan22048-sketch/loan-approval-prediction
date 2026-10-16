# Loan Approval Prediction using Machine Learning

## Project Overview

This project predicts whether a loan application is likely to be approved or rejected using machine learning techniques. It analyzes applicant information and uses a trained Random Forest model to make predictions through a Streamlit web application.

## Project Objective

The main objective is to build a machine learning model that predicts loan approval based on applicant financial and credit-related information.

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook
- Streamlit
- Git and GitHub

## Machine Learning Models

The following models were implemented and evaluated:

1. Logistic Regression
2. Naive Bayes
3. Support Vector Machine (SVM)
4. Random Forest Classifier

**Final Model:** Random Forest Classifier, used in the deployed Streamlit application.

## Data Preprocessing

The following preprocessing and feature engineering steps were performed:

- Exploratory Data Analysis (EDA)
- Missing value handling
- Categorical data encoding using One-Hot Encoding and Label Encoding
- Feature engineering using squared features for Credit Score and DTI Ratio
- Train-test split
- Feature scaling using StandardScaler

## Model Training and Evaluation

The dataset was divided into training and testing sets to train and evaluate the machine learning models.

The Random Forest model achieved **91% accuracy** in the current notebook evaluation.

## Web Application

A web application was developed using Streamlit. Users can enter applicant information and receive a loan approval prediction.

**Live Demo:**  
https://loan-approval-prediction-fkztfeaj2nrqnfe737gp7k.streamlit.app/

## Project Files

- `app.py` — Streamlit web application
- `loan_model.pkl` — Saved machine learning model and related preprocessing artifacts
- `credit_wise.ipynb` — Jupyter Notebook containing data analysis, preprocessing, model training, and evaluation
- `loan_approval_data.csv` — Dataset used for the project
- `requirements.txt` — Python dependencies required to run the application

## How to Run the Project Locally

1. Clone the repository:

   ```bash
   git clone https://github.com/sartajkhan22048-sketch/loan-approval-prediction.git
   ```

2. Open the project folder:

   ```bash
   cd loan-approval-prediction
   ```

3. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Streamlit application:

   ```bash
   streamlit run app.py
   ```

## Key Learnings

- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Training and evaluating machine learning models
- Using Scikit-learn for classification
- Building interactive applications with Streamlit
- Deploying a machine learning project online using Streamlit Community Cloud
- Managing project files using GitHub

## Author

**Sartaj Khan**

GitHub: https://github.com/sartajkhan22048-sketch
