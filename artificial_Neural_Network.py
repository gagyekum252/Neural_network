import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset

data = pd.read_csv("Time_seriestechnic/Churn_Modelling.csv")
X = data.iloc[:, 3:13].values
y = data.iloc[:, 13].values

print(data.head())

# Encoding Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
print(X)

from sklearn.model_selection import train_test_split

# Spliting the dataset into the training set and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state =0)

print(len(data))
print(len(X_test))

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

# Classical ML
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
pred = clf.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, pred)

print(cm)

# Making the confusion Matrix
from sklearn.metrics import accuracy_score

score = accuracy_score(y_test, pred)
print(score*100)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf2 = RandomForestClassifier(n_estimators=100)
clf2.fit(X_train, y_train)
pred = clf2.predict(X_test)

# Making the confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_test, pred)
print(cm2)

# Making the confusion Matrix
from sklearn.metrics import accuracy_score

score2 = accuracy_score(y_test, pred)
print(score2*100)

# ANN Neural Networks practicals
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN 
# input_dim: input shape (shape of the data, how many features do we have in the data)
# Adding the input layer and the first hidden layer
classifier = Sequential ()
# Adding the input layer and first hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim = 10))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
y_train[0]
# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=10, epochs=200)

# Part 3 making predictions and evaluating the model
# Predicting the model
y_pred_ann = classifier.predict(X_test)
y_pred_ann = (y_pred_ann>0.5)

cm3 = confusion_matrix(y_test, y_pred_ann)

# Predicting a single new observation
"""Predict if the customer with the following informations will leave the bank:
Geography: France
Credit Score: 600
Gender: Male
Age: 40
Tenure: 3
Balance: 60000
Number of products : 2
Has Credit Card Yes
Estimated Salary: 50000
"""
new_prediction = classifier.predict(sc.transform(np.array([[600, 0, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
new_prediction = (new_prediction> 0.5)
print(new_prediction)

# making the confusion matrix
# from sklearn.metrics 
#from keras.wrappers.scikit_learn import KerasClassifier
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu', input_dim = 10))
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units = 6, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 10)

accuracies = cross_val_score (estimator= classifier, X = X_train, y = y_train, cv=10, n_jobs= -1)
mean1 = accuracies.mean()
variance = accuracies.std()

