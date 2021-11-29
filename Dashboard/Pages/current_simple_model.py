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

# load titanic data
train = pd.read_csv("MyProject/Data/titanic.csv")

# drop rows containing NA values
train = train.dropna(axis=0).reset_index(drop=True)

# encode gender variable 
le_sex = LabelEncoder()
train["Sex"] = le_sex.fit_transform(train["Sex"])

# select features
features = ["Pclass", "Age", "Sex", "Fare"]

# split in train and test data
X, y = train[features], train["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4, stratify=y, random_state=42)

# Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(32, 16, 16), batch_size=32, early_stopping=False, random_state=42)
mlp = mlp.fit(X_train, y_train)



def plot_ale1d():
    fig, axs = plt.subplots(1,2,figsize=(15, 6), sharey=True)
    fig.tight_layout(pad=3)

    ale_petal_length = ale(
        X=X_train,
        model=mlp,
        feature=["Age"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[0], print_meanres=True)

    ale_setal_length = ale(
        X=X_train,
        model=mlp,
        feature=["Fare"],
        include_CI=True,
        target=1,
        fig=fig,
        ax=axs[1])

    
    return fig

def plot_ale2d():
    fig, ax = plt.subplots(figsize=(2, 2))
    ale_2d = ale(
        X=X_train,
        model=mlp,
        feature=["Fare","Age"],
        include_CI=True,
        target=1, 
        fig=fig,
        ax=ax)
    return fig




def run():
    st.title("ALE")
    st.pyplot(plot_ale1d())

    _,col,_ = st.columns(3)
    col.pyplot(plot_ale2d())


