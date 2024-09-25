import sklearn
import sklearn.datasets
import sklearn.neighbors
from sklearn.neighbors import KNeighborsClassifier
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
csv_dataset['restecg'] = csv_dataset['restecg'].apply(cp_tonum)
csv_dataset['exang'] = csv_dataset['exang'].apply(exang_tonum)
csv_dataset['slope'] = csv_dataset['slope'].apply(slope_tonum)
csv_dataset['thal'] = csv_dataset['thal'].apply(thal_tonum)


#Seperate into features and targets
X, y = csv_dataset.drop(['id', 'num'], axis=1).to_numpy(), csv_dataset['num'].to_numpy()

#Matching num classifiers to their corresponding meanings
target_names = ["No heart disease", "Mild heart disease", "Medium heart disease", "Severe heart disease", "Very severe heart disease"]

#split the dataset
X_Train_Dev, X_Test, y_Train_Dev, y_Test= train_test_split(X, y, test_size=0.1, random_state=42)
X_Train, X_Dev, y_Train, y_Dev = train_test_split(X_Train_Dev, y_Train_Dev, test_size=1/9, random_state=42)

#Model 1: Nearest Neighbor
Model_1 = sklearn.neighbors.KNeighborsClassifier(n_neighbors=1)
Model_1.fit(X_Train, y_Train)

#Model 2: Stochastic Gradient Descent


#Model 3:


#Make Predictions
Model_1_Predictions = Model_1.predict(X_Dev)

# #Display Results
print(f"Model 1: {accuracy_score(X_Dev, Model_1_Predictions)} was the accuracy")



print(X_Train)
print(y_Train)
print()
print(X_Dev)
print(y_Dev)
print()
print(X_Test)
print(y_Test)