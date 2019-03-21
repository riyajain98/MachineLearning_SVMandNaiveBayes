# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 00:59:15 2019

@author: RIYA JAIN
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


col_name =  ['Id', 'SepalLength in cm', 'SepalWidth in cm','PetalLength in cm', 'PetalWidth in cm','Species' ]
dataset = pd.read_csv('Iris.csv')

#dropping ID column
dataset = dataset.drop('Id', axis = 1)

#encodeing categorical dependent data
dataset['Species']= LabelEncoder().fit_transform(dataset['Species'])


#Splitting dataset to traiing set and test set
x_train, x_test = train_test_split(dataset, test_size = 0.4, random_state = 0)


#Naive Bayes 
"""
classifier = GaussianNB()
y_classifier = x_train['Species']
x_classifier = x_train.drop('Species', axis = 1)
y_test = x_test.Species
x_test = x_test.drop('Species', axis = 1)
classifier.fit(x_classifier, y_classifier)

y_predict = classifier.predict(x_test)
print(np.unique(y_predict))
print(accuracy_score(y_test, y_predict))

# Naive Bayes Accuracy: 0.9333333333333333"""

#Support Vector Machines

classifier = SVC(kernel = 'linear', random_state = 0)
y_classifier = x_train['Species']
x_classifier = x_train.drop('Species', axis = 1)
y_test = x_test.Species
x_test = x_test.drop('Species', axis = 1)
classifier.fit(x_classifier, y_classifier)

			
y_predict = classifier.predict(x_test)
print(np.unique(y_predict))
print(accuracy_score(y_test, y_predict))


#SVM accuracy: 0.9666666666666667

