import sklearn
import sklearn.datasets

import numpy as np
import pandas as pd

# Loading dataset into project
csv_file = '/home/brayden/Desktop/School/ARI 510/ARI-510-Lab-1/Lab1/heart_disease_uci.csv'
csv_dataset = pd.read_csv(csv_file)
X, y = csv_dataset.drop(['id', 'num'], axis=1).to_numpy(), csv_dataset['num'].to_numpy()

#Matching num classifiers to their corresponding meanings
target_names = ["No heart disease", "Mild heart disease", "Medium heart disease", "Severe heart disease", "Very severe heart disease"]

print(X)
print()
print(y)