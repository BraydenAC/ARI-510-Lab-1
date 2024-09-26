import sklearn
import sklearn.datasets
import sklearn.neighbors
from numpy.random import random_sample
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.linear_model import SGDClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import numpy as np
import pandas as pd

#functions to change all non-numerical entries to numerical ones
def sex_tonum(entry):
    if entry == 'Male':
        return 1
    else:
        return 0
def origin_tonum(entry):
    switcher = {
        'VA Long Beach': 0,
        'Cleveland': 1,
        'Switzerland': 2,
        'Hungary': 3
    }
    return switcher.get(entry, None)
def cp_tonum(entry):
    switcher = {
        'typical angina': 0,
        'asymptomatic': 1,
        'non-anginal': 2,
        'atypical angina': 3
    }
    return switcher.get(entry, None)
def fbs_tonum(entry):
    if entry:
        return 1
    else:
        return 0
def restecg_tonum(entry):
    switcher = {
        'st-t_abnormality': 0,
        'normal': 1,
        'lv hypertrophy': 2,
    }
    return switcher.get(entry, None)
def exang_tonum(entry):
    if entry:
        return 1
    else:
        return 0
def slope_tonum(entry):
    switcher = {
        'flat': 0,
        'downsloping': 1,
        'upsloping': 2,
    }
    return switcher.get(entry, None)
def thal_tonum(entry):
    switcher = {
        'normal': 0,
        'reversable defect': 1,
        'fixed defect': 2,
    }
    return switcher.get(entry, None)

# Loading dataset into project
csv_file = 'heart_disease_uci.csv'
csv_dataset = pd.read_csv(csv_file)

#Apply numerical converting functions
csv_dataset['sex'] = csv_dataset['sex'].apply(sex_tonum)
csv_dataset['dataset'] = csv_dataset['dataset'].apply(origin_tonum)
csv_dataset['cp'] = csv_dataset['cp'].apply(cp_tonum)
csv_dataset['fbs'] = csv_dataset['fbs'].apply(fbs_tonum)
csv_dataset['restecg'] = csv_dataset['restecg'].apply(restecg_tonum)
csv_dataset['exang'] = csv_dataset['exang'].apply(exang_tonum)
csv_dataset['slope'] = csv_dataset['slope'].apply(slope_tonum)
csv_dataset['thal'] = csv_dataset['thal'].apply(thal_tonum)

#Mean Imputation
mean_value = csv_dataset.mean()
csv_dataset = csv_dataset.fillna(mean_value)

#Seperate into features and targets
pre_X, pre_y = csv_dataset.drop(['id', 'num'], axis=1).to_numpy(), csv_dataset['num'].to_numpy()
smote = SMOTE(random_state=42)
X, y = smote.fit_resample(pre_X, pre_y)

#Matching num classifiers to their corresponding meanings
target_names = ["No heart disease", "Mild heart disease", "Medium heart disease", "Severe heart disease", "Very severe heart disease"]

#split the dataset
X_Train_Dev, X_Test, y_Train_Dev, y_Test= train_test_split(X, y, test_size=0.1, random_state=42)
X_Train, X_Dev, y_Train, y_Dev = train_test_split(X_Train_Dev, y_Train_Dev, test_size=1/9, random_state=42)

# #Model 1: KNeighborsClassifier
Model_1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=4, weights='distance', algorithm='ball_tree', leaf_size=30, n_jobs=-1)
Model_1.fit(X_Train, y_Train)

# #Model 2: Stochastic Gradient Descent
Model_2 = SGDClassifier(loss="hinge", penalty="l1", max_iter=150, shuffle=True, random_state=70)
Model_2.fit(X_Train, y_Train)

# #Model 3: Decision Tree
Model_3 = tree.DecisionTreeClassifier(class_weight='balanced', splitter='best', min_samples_split=4, max_features=0.6, random_state=65)
Model_3.fit(X_Train, y_Train)

# #Make Predictions
# Model_1_Predictions = Model_1.predict(X_Dev)
# Model_2_Predictions = Model_2.predict(X_Dev)
# Model_3_Predictions = Model_3.predict(X_Dev)
Model_1_Predictions = Model_1.predict(X_Test)
Model_2_Predictions = Model_2.predict(X_Test)
Model_3_Predictions = Model_3.predict(X_Test)

# # #Display Results
# print(f"Model 1: {accuracy_score(y_Dev, Model_1_Predictions)}")
# print(f"Model 2: {accuracy_score(y_Dev, Model_2_Predictions)}")
# print(f"Model 3: {accuracy_score(y_Dev, Model_3_Predictions)}")
print(f"Model 1: {accuracy_score(y_Test, Model_1_Predictions)}")
print(f"Model 2: {accuracy_score(y_Test, Model_2_Predictions)}")
print(f"Model 3: {accuracy_score(y_Test, Model_3_Predictions)}")
# print("Model 1")
# print(classification_report(y_Dev, Model_1_Predictions))
# print("\nModel 2")
# print(classification_report(y_Dev, Model_2_Predictions))
# print("\nModel 3")
# print(classification_report(y_Dev, Model_3_Predictions))
print("Model 1")
print(classification_report(y_Test, Model_1_Predictions))
print("\nModel 2")
print(classification_report(y_Test, Model_2_Predictions))
print("\nModel 3")
print(classification_report(y_Test, Model_3_Predictions))