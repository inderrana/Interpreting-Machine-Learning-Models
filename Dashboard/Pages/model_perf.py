import streamlit as st

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import PartialDependenceDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, r2_score, mean_absolute_error
import PyALE
from keras.models import model_from_json
from tensorflow.keras.models import load_model


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

#Logreg
logreg = LogisticRegression().fit(X_train, y_train)
# predict
pred_train = logreg.predict(X_train)
pred_test = logreg.predict(X_test)

# evaluate predictions
acc_train_lr = accuracy_score(y_train, np.round(pred_train))
acc_test_lr = accuracy_score(y_test, np.round(pred_test))


#Decision Tree
# define model
tree = DecisionTreeClassifier(max_depth=5, random_state=42)
# fit model
tree = tree.fit(X_train, y_train)

# predict probabilities
pred_train = tree.predict_proba(X_train)
pred_test = tree.predict_proba(X_test)

# evaluate predictions
acc_train_dt = accuracy_score(y_train, np.argmax(pred_train, axis=1))
acc_test_dt = accuracy_score(y_test, np.argmax(pred_test, axis=1))


# Random Forest
rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
rf.fit(X_train,y_train)
# predict probabilities
pred_train = rf.predict_proba(X_train)
pred_test = rf.predict_proba(X_test)
# evaluate predictions
acc_train_rf = accuracy_score(y_train, np.argmax(pred_train, axis=1))
acc_test_rf = accuracy_score(y_test, np.argmax(pred_test, axis=1))

# Neural Network
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

loaded_model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

test_nn_s = loaded_model.evaluate(X_test, y_test)
test_nn = test_nn_s[1]*100

train_nn_s = loaded_model.evaluate(X_train, y_train)
train_nn = train_nn_s[1]*100



eval_dict_lr = {
    "Log Regression":{
        "Acc_train":acc_train_lr,
        "Acc_test":acc_test_lr
    }
}

eval_dict_dt = {
    "Decision Tree":{
        "Acc_train":acc_train_dt,
        "Acc_test":acc_test_dt
    }
}

eval_dict_rf = {
    "Random Forest":{
        "Acc_train":acc_train_rf,
        "Acc_test":acc_test_rf
    }
}

eval_dict_nn = {
    "Neural Network":{
        "Acc_train":train_nn,
        "Acc_test":test_nn
    }
}

def plot_eval_lr():
    return pd.DataFrame(eval_dict_lr)

def plot_eval_dt():
    return pd.DataFrame(eval_dict_dt)    

def plot_eval_rf():
    return pd.DataFrame(eval_dict_rf)    

def plot_eval_nn():
    return pd.DataFrame(eval_dict_nn)    

def run():

    st.image("Images/logo.png", width=150)
    st.title("Model Performances without feature selection")
    
    c1,c2, c3, c4 = st.columns((2,2,2,2))
    with c1:
        st.dataframe(plot_eval_lr())
    
    with c2:
        st.dataframe(plot_eval_dt())
    
    with c3:
        st.dataframe(plot_eval_rf())

    with c4:
        st.dataframe(plot_eval_nn())


    # with c1:
        # with st.echo():
        #     st.write("Logistic Regression")
        #     st.dataframe(plot_eval())

        