# -*- coding: utf-8 -*-
"""
Created on Sun May 27 02:21:06 2018

@author: soner
"""

import graphviz
import io
import pydotplus



import sklearn
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
import seaborn as sns

import readData



features = ["subject", "word_count", "vowel_count", "exclamation_count", "interrogation_count"]


df = pd.read_csv("set2.csv")

train, test = train_test_split(df, test_size = 0.2)

#features listesinde bulunan başlıklar ile train setin başlıklarının çakışan kısmı x_traine
x_train = train[features] 
y_train = train['Ironic'] 

x_test = test[features] 
y_test = test['Ironic']



c = DecisionTreeClassifier(min_samples_split=100) 
dt = c.fit(x_train, y_train) #modeli eğitiyoruz.

# eğitilen modele uygun ağacı çizen fonksiyon. 
def show_tree(tree, features, path):
    f = io.StringIO()
    sklearn.tree.export_graphviz(tree, out_file = f, feature_names=features)
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    img=plt.imread(path)
    
    plt.rcParams["figure.figsize"] = (20,20)
    plt.imshow(img)

#jupyter de çalıştırınca yukardaki grrafik ile çakışmıyor ama spyder ide ile çakısıyor üst üste binen 2 foto çıkıyor. 
#show_tree(dt, features, 'dec_tree_01.png')#yukardaki sebep nedeniyle ayrı ayrı parça parça yürütcem sunumda bu kısmı .


y_predict = dt.predict(x_test)
cm = confusion_matrix(y_test, y_predict)
print("SVM")
print(cm)

score = dt.score(x_test, y_test)
print("Score : " ,score)

import test_model

print(test_model.test_model(dt))




