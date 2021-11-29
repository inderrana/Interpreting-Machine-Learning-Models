from numpy.lib.twodim_base import mask_indices
import streamlit as st

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error
import shap

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

@st.cache
def rf_feature_imp():
    rf = RandomForestClassifier(n_estimators=500, max_depth=5, random_state=42)
    rf.fit(X_train,y_train)
    rf_feat_imp = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=["Feature importance"])
    return rf_feat_imp

@st.cache
def cor_mtx():
    corr = X_train.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots()
    sns.heatmap(corr, mask=mask, cmap=plt.cm.bwr, center=0, annot=True, fmt='.2f', square=True)
    
    #fig, ax = plt.subplots()
    #fig.set_size_inches(14, 8)
    #sns.heatmap(corr, mask=mask, cmap=plt.cm.bwr, center=0, annot=True, fmt='.2f', square=True)
    #for item in ax.get_yticklabels():
    #    item.set_rotation(0)
    return fig

logreg = LogisticRegression().fit(X_train, y_train)
X100 = shap.utils.sample(X_test, 500)
explainer_lr = shap.Explainer(logreg.predict, X100)
shap_values_lr = explainer_lr(X100)

def shp_hmp():
    fig = plt.figure()
    shap.plots.heatmap(shap_values_lr[:100])
    return fig
    

def run():
    st.title("Corelation")
    c1,c2 = st.columns((3,3))
    with c1:
        st.write("Test")
        mask = np.triu(np.ones_like(X_train.corr(), dtype=bool))
        fig, ax = plt.subplots()
        fig.set_size_inches(14, 8)
        sns.heatmap(X_train.corr(), mask=mask, cmap=plt.cm.bwr, center=0, annot=True, fmt='.2f', square=True)
        st.write(fig)
        

    with c2:
        fig = px.imshow(X_train.corr())
        st.plotly_chart(fig)

    with c1:
        st.write("Feature Importance")
        st.table(rf_feature_imp())
        #c1.pyplot(shp_hmp())




