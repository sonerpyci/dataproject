# -*- coding: utf-8 -*-
"""
Created on Mon May 28 08:56:54 2018

@author: soner
"""

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
import readData




features = ["subject", "word_count", "vowel_count", "exclamation_count", "interrogation_count"]  #decision tree yaklaşımının doğru çalışması için çok sayıda feature ile desteklenmesi gereklidir
                        # fakat bizim veri setimiz buna müsait olmadığı için tek feautre ile devam ediyoruz.


df = pd.read_csv("set2.csv")
#x_train, x_test, y_train, y_test = train_test_split(df, df.Ironic, test_size=0.2)
df = np.array(df)


train, test = train_test_split(df, test_size = 0.2,random_state=2)

X_train=train[:,1:6]
y_train=train[:,0]

X_test=test[:,1:6]
y_test=test[:,0]



neural=MLPClassifier(max_iter=40000,random_state=8,hidden_layer_sizes=[80,80])
neural.fit(X_train,y_train)

y_predict = neural.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print("Neural")
print(cm)
score = neural.score(X_test, y_test)
print("Score : " , score)

import test_model

#print(test_model.test_model(neural,))
