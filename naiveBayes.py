# -*- coding: utf-8 -*-
"""
Created on Mon May 28 08:56:54 2018

@author: soner
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix

import readData


df = pd.read_csv("set2.csv")

df = np.array(df)

#16 bin verinin yüzde 20 si test kalanı train olarak ayrıldı.
train, test = train_test_split(df, test_size = 0.2,random_state=2)

X_train=train[:,1:6] #train kümesinin feature sütunları x_train feature kümesine alındı.
y_train=train[:,0] #train kümesinin ironik başlığı y_train sonuç kümesine alındı

X_test=test[:,1:6] #test kümesinin feature sütunları test feature kümesine alındı.
y_test=test[:,0] #test kümesinin ironik başlığı test sonuç kümesine alındı


BernNB = BernoulliNB() 
BernNB.fit(X_train, y_train)#model eğitiliyor.
y_predict  = BernNB.predict(X_test)
cm = confusion_matrix(y_test, y_predict) #Doğruluk matrisi. true negative false pozitive vs.
print("BERNOULLI")
print(cm)
print("BernoulliNB accuracy : " + str(BernNB.score(X_test,y_test)))#train kümeleri ile model eğitilmüşti. Şimdi eğitim ile gelen yaklaşımalr test setine uygulanıp accuracy hesaplanıyor.

MultiNB = MultinomialNB()
MultiNB.fit(X_train, y_train)
y_predict = MultiNB.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print("MultinomialNB")
print(cm)
print("MultinomialNB accuracy : " + str(MultiNB.score(X_test,y_test)))


GausNB = GaussianNB()
GausNB.fit(X_train, y_train)
y_predict = GausNB.predict(X_test)
cm = confusion_matrix(y_test, y_predict)
print("GaussianNB")
print(cm)
print("GaussianNB accuracy : " + str(GausNB.score(X_test,y_test)))


import test_model

print(test_model.test_model(GausNB))
 