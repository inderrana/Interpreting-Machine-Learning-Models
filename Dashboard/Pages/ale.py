import streamlit as st

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import PyALE

def ale(target=None, print_meanres=False, **kwargs):
    if target is not None:
        class clf():
            def __init__(self, classifier):
                self.classifier = classifier
            def predict(self, X):
                return(self.classifier.predict_proba(X)[:, target])
        clf_dummy = clf(kwargs["model"])
        kwargs["model"] = clf_dummy
    if (print_meanres & len(kwargs["feature"])==1):
        mean_response = np.mean(kwargs["model"].predict(kwargs["X"]), axis=0)
        print(f"Mean response: {mean_response:.5f}")
    return PyALE.ale(**kwargs)

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

#Create train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=5)


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


def plot_ale1d():
    fig, axs = plt.subplots(1,2,figsize=(15, 6), sharey=True)
    fig.tight_layout(pad=3)

    ale_acc_lth_lr = ale(
        X=X_train,
        model=logreg,
        feature=["international_plan"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[0], print_meanres=True)

    ale_cx_svc_call_lr = ale(
        X=X_train,
        model=logreg,
        feature=["number_customer_service_calls"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[1], print_meanres=True)
    return fig

def plot_ale2d():
    fig, axs = plt.subplots(1,2,figsize=(15, 6), sharey=True)
    ale_al_nvm = ale(
        X=X_train,
        model=tree,
        feature=["international_plan", "number_vmail_messages"], 
        contour=True,
        target=1,
        fig=fig,
        ax=axs[0])

    ale_al_nvm = ale(
        X=X_train,
        model=tree,
        feature=["account_length", "total_day_minutes"], 
        contour=True,
        target=1,
        fig=fig,
        ax=axs[1])
    return fig

def plot_ale1d_rf():
    fig, axs = plt.subplots(1,2,figsize=(15, 6), sharey=True)
    fig.tight_layout(pad=3)

    ale_acc_lth_rf = ale(
        X=X_train,
        model=logreg,
        feature=["account_length"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[0], print_meanres=True)

    ale_cx_svc_call_rf = ale(
        X=X_train,
        model=logreg,
        feature=["number_vmail_messages"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[1], print_meanres=True)
    return fig

def plot_ale2d_rf():
    fig, axs = plt.subplots(1,2,figsize=(15, 6), sharey=True)
    ale_al_nvm = ale(
        X=X_train,
        model=rf,
        feature=["account_length", "number_vmail_messages"], 
        contour=True,
        target=1,
        fig=fig,
        ax=axs[0])

    ale_al_nvm = ale(
        X=X_train,
        model=rf,
        feature=["account_length", "number_customer_service_calls"], 
        contour=True,
        target=1,
        fig=fig,
        ax=axs[1])
    return fig


def plot_ale1d_dt():
    fig, axs = plt.subplots(1,2,figsize=(15, 6), sharey=True)
    fig.tight_layout(pad=3)

    ale_acc_lth_dt = ale(
        X=X_train,
        model=tree,
        feature=["account_length"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[0], print_meanres=True)

    ale_cx_svc_call_dt = ale(
        X=X_train,
        model=tree,
        feature=["number_customer_service_calls"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[1], print_meanres=True)
    return fig

def plot_ale2d_dt():
    fig, axs = plt.subplots(1,2,figsize=(15, 6), sharey=True)
    ale_al_nvm = ale(
        X=X_train,
        model=tree,
        feature=["account_length", "number_vmail_messages"], 
        contour=True,
        target=1,
        fig=fig,
        ax=axs[0])

    ale_al_nvm = ale(
        X=X_train,
        model=tree,
        feature=["account_length", "number_customer_service_calls"], 
        contour=True,
        target=1,
        fig=fig,
        ax=axs[1])
    return fig


def run():
    st.title("Accumulated Local Effects (ALE) Plot")
    st.write("Logistic Regression")
    st.pyplot(plot_ale1d())
    st.pyplot(plot_ale2d())

    st.write("Randon Forest")
    st.pyplot(plot_ale1d_rf())
    st.pyplot(plot_ale2d_rf())
    
    st.write("Desision Tree")
    st.pyplot(plot_ale1d_dt())
    st.pyplot(plot_ale2d_dt())

