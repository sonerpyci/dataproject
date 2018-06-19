import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


df = pd.read_csv("datayeni.csv")
df = df.drop('subject',axis=1)
y = np.array(df['Ironic'])
df = df.drop('text','cleanText','subject')
X = np.array(df)



X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.5)


clf = svm.SVC(kernel = 'linear', C=1)
print("\n*****SVM SCORE****\n")
scores1 = cross_val_score(clf, X_test, y_test, cv=5)
print(scores1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

scores2 = cross_val_score(clf, X_train, y_train, cv=5)

print(scores2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))



BernNB = BernoulliNB() 
print("\n*****BernoulliNB SCORE****\n")
scores1 = cross_val_score(BernNB, X_test, y_test, cv=5)
print(scores1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

scores2 = cross_val_score(BernNB, X_train, y_train, cv=5)

print(scores2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))


gNB = GaussianNB()
print("\n*****GaussianNB SCORE****\n")

scores1 = cross_val_score(gNB, X_test, y_test, cv=5)
print(scores1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

scores2 = cross_val_score(gNB, X_train, y_train, cv=5)

print(scores2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

c = DecisionTreeClassifier() 
print("\n*****DecisionTreeClassifier SCORE****\n")
scores1 = cross_val_score(c, X_test, y_test, cv=5)
print(scores1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

scores2 = cross_val_score(c, X_train, y_train, cv=5)

print(scores2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))

neural=MLPClassifier(max_iter=90000,random_state=8,hidden_layer_sizes=[80,80])
print("\n*****NEURAL NETWORK SCORE****\n")

scores1 = cross_val_score(neural, X_test, y_test, cv=5)
print(scores1)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores1.mean(), scores1.std() * 2))

scores2 = cross_val_score(neural, X_train, y_train, cv=5)

print(scores2)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores2.mean(), scores2.std() * 2))
