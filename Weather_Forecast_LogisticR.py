#!/usr/bin/env python
# coding: utf-8

# # `----------------@ WEATHER_CONDITION DATA ANALYSIS @---------------------------------------! CLASSIFICATION MODEL !----------------------`

# ### `IMPORT ALL THE REQUIRED PACKAGES:`

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image  
from six import StringIO
import pydotplus,graphviz
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn import svm


# ### `IMPORT CLEANED DATASET`

# In[ ]:


Cleaned_set = pd.read_csv('Cleaned_set.csv')


# ### `SPLIT THE DATA`

# In[ ]:


x = Cleaned_set[['Temperature', 'Apparent Temperature', 'Humidity', 'Wind Speed',
       'Wind Bearing', 'Visibility', 'Pressure', 'Solar irradiance intensity',
       'Rain_OR_SNOW1', 'Condensation1']]
y = Cleaned_set[['Cloud_Condition1']]


# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.76809, random_state = 100)
print('x_train shape :', x_train.shape)
print('x_test shape :', x_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# ### `Fit the train model and predict the test set.`

# ### `LOGISTIC REGRESSION :`

# #### `Checking for Dimensions of the train and test sets`

# In[ ]:


print("X_train: {}, Y_train: {}".format(len(x_train), len(x_test)))
print("X_train: {}, Y_train: {}".format(len(y_train), len(y_test)))


# #### `Manual tuned paramters for Logistic Regression Model`

# In[ ]:


model = LogisticRegression(C=0.01, penalty='l1', solver='liblinear')
model.fit(x_train, y_train)


# #### `Accuracy`

# In[ ]:


print("Showing Performance Metrics for Logistic Regression\n")

print ("Training Accuracy: {}".format(model.score(x_train, y_train)))
predicted = model.predict(x_test)
print ("Testing Accuracy: {}".format(accuracy_score(y_test, predicted)))


# #### `Cross Validation Accuracy`

# In[ ]:


print("Cross Validation Accuracy: \n")
cv_accuracy = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
print("Accuracy using 10 folds: ")
print(cv_accuracy)


# #### `Mean and Standard Deviation`

# In[ ]:


print("Mean accuracy: {}".format(cv_accuracy.mean()))
print("Standard Deviation: {}".format(cv_accuracy.std()))


# #### `Confusion Matrix for Logistic Regression`

# In[ ]:


labels = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
cm = confusion_matrix(y_test, predicted, labels=labels)
print(cm)


# #### `Classification Report`

# In[ ]:


print('Precision, Recall and f-1 Scores for Logistic Regression\n')
print(classification_report(y_test, predicted))


# ### `NAIVE_BAYES :`

# #### `Checking for Dimensions of the train and test sets`

# In[ ]:


print("X_train: {}, Y_train: {}".format(len(x_train), len(x_test)))
print("X_train: {}, Y_train: {}".format(len(y_train), len(y_test)))


# #### `Normalize the data because in naive bayes it won't work on negetive values.`

# In[ ]:


def Normalize_Data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# In[ ]:


x = Normalize_Data(x)


# #### `Checking for Dimensions of the train and test sets`

# In[ ]:


x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.76809, random_state = 100)
print('x_train shape :', x_train.shape)
print('x_test shape :', x_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# #### `Manual tuned paramters for Naive Bayes Model`

# In[ ]:


model = MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
model.fit(x_train, y_train)


# #### `Prediction`

# In[ ]:


predicted = model.predict(x_test)


# #### `Accuracy`

# In[ ]:


print("Showing Performance Metrics for Naive Bayes Multinomial\n")
print ("Training Accuracy: {}".format(model.score(x_train, y_train)))
print ("Testing Accuracy: {}".format(accuracy_score(y_test, predicted)))


# #### `Cross Validation Accuracy`

# In[ ]:


print("Cross Validation Accuracy: \n")
cv_accuracy = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
print("Accuracy using 10 folds: ")
print(cv_accuracy)


# #### ` Mean Accuracy and Standard Deviation`

# In[ ]:


print("Mean accuracy: {}".format(cv_accuracy.mean()))
print("Standard Deviation: {}".format(cv_accuracy.std()))


# #### `Confusion Matrix`

# In[ ]:


print("Confusion Matrix for Naive Bayes Multinomial\n")
labels = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
cm = confusion_matrix(y_test, predicted, labels=labels)
print(cm)


# #### `Classification Report`

# In[ ]:


print('Precision, Recall and f-1 Scores for Naive Bayes Multinomial\n')
print(classification_report(y_test, predicted))


# #### `NAIVE BAYES MODEL PARAMETER TUNING`

# In[ ]:


model = MultinomialNB()
param_grid = {'alpha': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]}


# In[ ]:


print("Hyper Parameter Tuning Results\n")

grid = GridSearchCV(estimator=model, param_grid = param_grid,
                    cv = 5)
grid.fit(x_train, y_train)

print("Results returned by GridSearchCV\n")
print("Best estimator: ", grid.best_estimator_)
print("\n")
print("Best Accuracy Score: ", grid.best_score_)
print("\n")
print("Best parameters found at: ", grid.best_params_)


# ### `Logistic Regression Accuracy : "0.3262" on Training Set.`
# ### `Naive Bayes Accuracy : "0.3220" on Training Set.`
