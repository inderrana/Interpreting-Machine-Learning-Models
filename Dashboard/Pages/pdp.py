import streamlit as st

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


train = pd.read_csv("./Data/churn_train.csv")
test = pd.read_csv("./Data/churn_test.csv")


#Encode yes and no with 1 and 0
yes_no_columns = ['international_plan','voice_mail_plan','churn']
for col in yes_no_columns:
    train[col].replace({'yes': 1,'no': 0},inplace=True)

yes_no_columns_test = ['international_plan','voice_mail_plan']
for col in yes_no_columns_test:
    test[col].replace({'yes': 1,'no': 0},inplace=True)    


#Drop state and area code columns
train.drop('state',axis='columns',inplace=True)
train.drop('area_code',axis='columns',inplace=True)

#Scale columns 
cols_to_scale = ['account_length','total_night_charge']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train[cols_to_scale] = scaler.fit_transform(train[cols_to_scale])
test[cols_to_scale] = scaler.fit_transform(test[cols_to_scale])

#Split target
X = train.drop('churn',axis='columns')
y = train['churn']

#Split target
X = train.drop('churn',axis='columns')
y = train['churn']

#Create train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)

# Logistic Regression
logreg = LogisticRegression().fit(X_train, y_train)

#Desision Tree
tree = DecisionTreeClassifier(max_depth=5, random_state=42)

# fit model
tree = tree.fit(X_train, y_train)

#Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train,y_train)

def plot_pdp_lr():
    fig1, ax1 = plt.subplots(figsize=(15, 6))       
    PartialDependenceDisplay.from_estimator(estimator=logreg, X=X_train, features=[1, 4,16, (1,3),(1,4),(1,16)], target=1, ax=ax1)
    fig1.tight_layout(pad=2.0)
    return fig1

def plot_pdp_dt():
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    PartialDependenceDisplay.from_estimator(estimator=tree, X=X_train, features=[1, 4,16, (1,3),(1,4),(1,16)], target=1, ax=ax1)
    fig1.tight_layout(pad=2.0)
    return fig1

def plot_pdp_rf():
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    PartialDependenceDisplay.from_estimator(estimator=rf, X=X_train, features=[1, 4,16, (1,3),(1,4),(1,16)], target=1, ax=ax1)
    fig1.tight_layout(pad=2.0)
    return fig1

def run():
    st.title("Partial Dependency Plots")
    st.write("Logistic Regression")
    st.pyplot(plot_pdp_lr())
    st.write("Decision Tree")
    st.pyplot(plot_pdp_dt())
    st.write("Random Forest")
    st.pyplot(plot_pdp_rf())
