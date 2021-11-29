import streamlit as st
import streamlit.components.v1 as components

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
import shap
shap.initjs()

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

@st.cache
# Exact Explainer (.predict) LR
def run_exact_lr():
    global explainer_lr, shap_values_lr
    explainer_lr = shap.Explainer(logreg.predict, X_train)
    shap_values_lr = explainer_lr(X_test)

@st.cache
# Kernel Explainer (.predict_proba) LR
def run_kernel_lr():
    global explainer_lr, shap_values_lr
    explainer_lr = shap.KernelExplainer(logreg.predict_proba, X_train)
    shap_values_lr = explainer_lr.shap_values(X_test)

@st.cache
# Exact Explainer (.predict) DT
def run_exact_dt():
    global explainer_dt, shap_values_dt
    explainer_dt = explainer = shap.Explainer(tree.predict, X_train)
    shap_values_dt = explainer_dt(X_test)

@st.cache
# Kernel Explainer (.predict_proba) DT
def run_kernel_dt():
    global explainer_dt, shap_values_dt
    explainer_dt = shap.KernelExplainer(tree.predict_proba, X_train)
    shap_values_dt = explainer_dt.shap_values(X_test)

@st.cache
# Exact Explainer (.predict) RF
def run_exact_rf():
    global explainer_rf, shap_values_rf
    explainer_rf = explainer = shap.Explainer(rf.predict, X_train)
    shap_values_rf = explainer_rf(X_test)

@st.cache
# Kernel Explainer (.predict_proba) RF
def run_kernel_rf():
    global explainer_rf, shap_values_rf
    explainer_rf = shap.KernelExplainer(rf.predict_proba, X_train)
    shap_values_rf = explainer_rf.shap_values(X_test)


#run_exact_lr()
#run_exact_dt()
#run_exact_rf()

@st.cache
# helper to plot js plots
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

@st.cache
# force plot (requires js helper)
def plot_force_lr():
    fig = shap.force_plot(explainer_lr.expected_value[1], shap_values_lr[1], X_test)
    return fig

# 
def plot_bar_lr():
    fig = plt.figure()
    shap.plots.bar(shap_values_lr, show=False)
    return fig

def plot_beeswarm_lr():
    fig = plt.figure()
    shap.plots.beeswarm(shap_values_lr)
    return fig

def plot_cohorts_lr():
    fig = plt.figure()
    shap.plots.bar(shap_values_lr.cohorts(2).abs.mean(0))
    return fig

def plot_wtr_lr():
    fig = plt.figure()
    shap.plots.waterfall(shap_values_lr[10], max_display=14)
    return fig

#DT

def plot_force_dt():
    fig = shap.force_plot(explainer_dt.expected_value[1], shap_values_dt[1], X_test)
    return fig

def plot_bar_dt():
    fig = plt.figure()
    shap.plots.bar(shap_values_dt, show=False)
    return fig

def plot_beeswarm_dt():
    fig = plt.figure()
    shap.plots.beeswarm(shap_values_dt)
    return fig

def plot_cohorts_dt():
    fig = plt.figure()
    shap.plots.bar(shap_values_dt.cohorts(2).abs.mean(0))
    return fig

def plot_wtr_dt():
    fig = plt.figure()
    shap.plots.waterfall(shap_values_dt[10], max_display=14)
    return fig

#RF

def plot_force_rf():
    fig = shap.force_plot(explainer_rf.expected_value[1], shap_values_rf[1], X_test)
    return fig

def plot_bar_rf():
    fig = plt.figure()
    shap.plots.bar(shap_values_rf, show=False)
    return fig

def plot_beeswarm_rf():
    fig = plt.figure()
    shap.plots.beeswarm(shap_values_rf)
    return fig

def plot_cohorts_rf():
    fig = plt.figure()
    shap.plots.bar(shap_values_rf.cohorts(2).abs.mean(0))
    return fig

def plot_wtr_rf():
    fig = plt.figure()
    shap.plots.waterfall(shap_values_rf[10], max_display=14)
    return fig


def run():
    st.title("SHAP")

    selectbox = st.sidebar.selectbox(
        "Select SHAP Explainer type:",
        ("ExactExplainer", "KernelExplainer"))

    if selectbox =="ExactExplainer":
        run_exact_lr()
        run_exact_dt()
        run_exact_rf()
        
        c1,c2,c3 = st.columns(3)
        with c1:
            st.write("Logistic Regression")
            st.pyplot(plot_bar_lr())
            st.pyplot(plot_beeswarm_lr())
            st.pyplot(plot_wtr_lr())

        #DT
        with c2:
            st.write("Decision Tree")
            st.pyplot(plot_bar_dt())
            st.pyplot(plot_beeswarm_dt())
            st.pyplot(plot_wtr_dt())

        #DT
        with c3:
            st.write("Random Forest")
            st.pyplot(plot_bar_rf())
            st.pyplot(plot_beeswarm_rf())
            st.pyplot(plot_wtr_rf())
        
    elif selectbox =="KernelExplainer":
        run_kernel_lr()
        run_kernel_dt()
        # js plot
        st_shap(plot_force_lr(), 400)
        st_shap(plot_force_dt(), 400)