import streamlit as st
import streamlit.components.v1 as components

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay
from sklearn.tree import DecisionTreeClassifier, plot_tree
from keras.models import model_from_json
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import PyALE
import shap
shap.initjs()


# adapt PyALE.ale function to incorporate classification models (target in PyALE, can learn if want to learn programming)
# reference: https://htmlpreview.github.io/?https://github.com/DanaJomar/PyALE/blob/master/examples/ALE%20plots%20for%20classification%20models.html
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

# load data
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

variables = ['account_length','international_plan', 
       'voice_mail_plan', 'total_day_charge',
        'number_customer_service_calls']


#shap_df = test[variables][:500]

# define model
rf_adj = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf_adj.fit(X_train[variables],y_train)

#Random Forest
pred_train = rf_adj.predict_proba(X_train[variables])
pred_test = rf_adj.predict_proba(X_test[variables])

# evaluate predictions
acc_train = accuracy_score(y_train, np.argmax(pred_train, axis=1))
acc_test = accuracy_score(y_test, np.argmax(pred_test, axis=1))

#print(f"Train:\tACC={acc_train:.4f}")
#print(f"Test:\tACC={acc_test:.4f}")

eval_dict_rf = {
    "Random Forest":{
        "Acc_train":acc_train,
        "Acc_test":acc_test
    }
}

def plot_eval_rf():
    return pd.DataFrame(eval_dict_rf)    


#Predict on test
rf_pred_c = rf_adj.predict(test[variables])
rf_predicted_dataset  = test

#add predictions to test dataset
test["churn"] = rf_pred_c



#Load shap explainer
# @st.cache
# def run_kernel_rf():
#     global explainer_rf, shap_values_rf
#     shap_df = test[variables][:500]
#     explainer_rf = shap.KernelExplainer(rf_adj.predict_proba, X_train[variables])
#     shap_values_rf = explainer_rf.shap_values(shap_df)



# @st.cache
# def plot_cohorts_rf():
#     fig = plt.figure()
#     shap.force_plot(explainer_rf.expected_value[1], shap_values_rf[1], shap_df)
#     return fig    

def plot_ale2d():
    fig, axs = plt.subplots(1,2,figsize=(15, 8), sharey=True)
    ale_al_nvm = ale(
        X=test[variables],
        model=rf_adj,
        feature=["account_length", "voice_mail_plan"], 
        contour=True,
        fig=fig,
        ax=axs[0])

    ale_al_nvm = ale(
        X=test[variables],
        model=rf_adj,
        feature=["international_plan", "account_length"], 
        contour=True,
        fig=fig,
        ax=axs[1])
    return fig


def plotpdp():
    fig, ax = plt.subplots(figsize=(15,8))
    PartialDependenceDisplay.from_estimator(estimator=rf_adj, X=test[variables], features=[1,2,3,(1,2), (1,3),(0,4)],ax=ax)
    fig.tight_layout(pad=2.0)
    return fig


def run():
    st.title("Interpretation of final selected model")
    st.dataframe(plot_eval_rf())

    c1,c2 = st.columns(2)
    with c1:
        st.write("ALE Plot")
        st.pyplot(plot_ale2d())

    with c2:
        st.write("Partial Dependency Plots")
        st.pyplot(plotpdp())         


