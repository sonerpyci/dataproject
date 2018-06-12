# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 22:06:37 2018

@author: soner
"""


from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

import readData

# Bu kısım word_count vs gibi featureları -1 ile 1 arasındaki değerlere oturtur. Sklearn dökümantasyonunda highly recommended for performance
def normalize(df):
    result = df.copy()
    for feature_name in df.columns:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


features = ["subject","word_count"]

df = pd.read_csv("set2.csv")

df["Ironic"] = pd.to_numeric(df['Ironic'])
#normalize işlemi yapılıyor.
df = normalize(df.iloc[:,0:6])

train, test = train_test_split(df, test_size = 0.2,random_state=2)

X_train=train.iloc[:,1:6]#verileri ayırdık
y_train=train.iloc[:,0]

X_test=test.iloc[:,1:6]
y_test=test.iloc[:,0]


clf = svm.SVC()
clf.fit(X_train, y_train) #modeli eğittik

y_predict = clf.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print("SVM")
print(cm)
score = clf.score(X_test, y_test)
print("Score : " , score)

