import streamlit as st

import pandas as pd
import numpy as np

intro = """
The objective of this report is to investigate customer churn for Telco. Telco is one of the largest telecommunication service providers. As part of its value system to provide quality customer experience, it wants to investigate reasons for customer churn. Currently, a simple churn prediction model is being used but Telco has acquired the services of IESEG Consultants to propose an efficient churn prediction model.
In this report, IESEG Consultants study the company data and its existing model. The available information is then used to build more accurate models. Interpretation techniques are also outlined in the report to help Telco seamlessly implement the proposed models.
"""

scenario = """
Currently, Telco is using a linear model to predict churn. This method is used for its ease of interpretability. The insights provided by the coefficients of the linear model are used to continuously enhance the customer experience and take preventative measures for potential churn in the future. Additionally, customers with high churn probabilities receive personalized retention offers (e.g. discounts, free minutes).
"""
f_sel = """
The first step to building a model with high accuracy is to select variables wisely. Variables can have correlations between themselves and if this is not taken into account the results can be biased.
This report uses two main methods for feature selection - Correlation Matrix and Feature Importance. A correlation matrix is a table showing correlation coefficients between sets of variables. Each variable in the table is correlated with each of the other values in the table. This shows which pairs have the highest correlation. Feature importance refers to techniques that assign a score to input features based on how useful they are at predicting a target variable. They improve the efficiency and effectiveness of a predictive model on the problem.
"""


def run():

    st.image("Images/logo.png", width=150)
    st.title("Customer Churn Predictions")
    st.subheader("Introduction")
    st.write(intro)
    
    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("Current Scenario")
    st.write(scenario)

    st.markdown("""<hr style="height:10px;border:none;color:#333;background-color:#333;" /> """, unsafe_allow_html=True)
    st.subheader("Feature Selection")
    st.write(f_sel)
    
    
        