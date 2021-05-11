#!/usr/bin/env python
# coding: utf-8

# # `----------------@ WEATHER_CONDITION DATA ANALYSIS @---------------------------------------! CLASSIFICATION MODEL !----------------------`

# ### `IMPORT ALL THE REQUIRED PACKAGES - EDA:`

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from IPython.display import Image  
from six import StringIO
import pydotplus,graphviz
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OrdinalEncoder
import time


# In[ ]:


Cleaned_set = pd.read_csv('Cleaned_set.csv')
Cleaned_set.head()


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

# ### `DECISION TREE :`

# #### `Manual tuned paramters for Decision Tree`

# In[ ]:


DT = DecisionTreeClassifier(max_leaf_nodes=5, random_state=0)
DT.fit(x_train, y_train)


# #### `Prediction and Classification Report`

# In[ ]:


# making predictions
y_pred_default = DT.predict(x_test)

# Printing classifier report after prediction
print(classification_report(y_test,y_pred_default))


# #### `Confusion Matrix`

# In[ ]:


print(confusion_matrix(y_test,y_pred_default))


# In[ ]:


print('accuracy_score :',accuracy_score(y_test,y_pred_default))


# #### ` Assigning Features`

# In[ ]:


features = list(Cleaned_set.drop(['Day','Cloud_Condition1','cluster'],axis=1))
features


# #### `Importing the path for Tree building`

# In[ ]:


import os
os.environ["PATH"] += os.pathsep + 'C:/Users/jagad/AppData/Local/Programs/Python/Python38-32/Lib/site-packages/graphviz/bin'


# #### `plotting tree with max_depth=3`

# In[ ]:


dot_data = StringIO()  
export_graphviz(DT, out_file=dot_data,
                feature_names=features, filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# #### `GridSearchCV to find optimal max_depth and specify number of folds for k-fold CV`

# In[ ]:


n_folds = 5

# parameters to build the model on
parameters = {'max_depth': range(1, 40)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score = True)
tree.fit(x_train, y_train)


# #### `scores of GridSearch CV`

# In[ ]:


scores = tree.cv_results_
pd.DataFrame(scores).head()


# #### `plotting accuracies with max_depth`

# In[ ]:


plt.figure()
plt.plot(scores["param_max_depth"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_max_depth"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("max_depth")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# #### `GridSearchCV to find optimal max_depth`
# #### `specify number of folds for k-fold CV`

# In[ ]:


n_folds = 5

# parameters to build the model on
parameters = {'min_samples_leaf': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score = True)
tree.fit(x_train, y_train)


# #### `scores of GridSearch CV`

# In[ ]:


scores = tree.cv_results_
pd.DataFrame(scores).head()


# #### `plotting accuracies with max_depth`

# In[ ]:


plt.figure()
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_train_score"], 
         label="training accuracy")
plt.plot(scores["param_min_samples_leaf"], 
         scores["mean_test_score"], 
         label="test accuracy")
plt.xlabel("min_samples_leaf")
plt.ylabel("Accuracy")
plt.legend()
plt.show()


# #### `GridSearchCV to find optimal max_depth`
# #### `specify number of folds for k-fold CV`

# In[ ]:


n_folds = 5

# parameters to build the model on
parameters = {'min_samples_split': range(5, 200, 20)}

# instantiate the model
dtree = DecisionTreeClassifier(criterion = "gini", 
                               random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy", return_train_score = True)
tree.fit(x_train, y_train)


# #### `scores of GridSearch CV`

# In[ ]:


scores = tree.cv_results_
pd.DataFrame(scores).head()


# #### `Create the parameter grid`

# In[ ]:


param_grid = {
    'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ['gini', "entropy"]
}

n_folds = 5

# Instantiate the grid search model
dtree = DecisionTreeClassifier()
grid_search = GridSearchCV(estimator = dtree, param_grid = param_grid, 
                          cv = n_folds, verbose = 1)

# Fit the grid search to the data
grid_search.fit(x_train,y_train)


# #### `CV Results`

# In[ ]:


cv_results = pd.DataFrame(grid_search.cv_results_)
cv_results.head(3)


# #### `printing the optimal accuracy score and hyperparameters`

# In[ ]:


print("best accuracy", grid_search.best_score_)
print(grid_search.best_estimator_)


# #### `Model with optimal hyperparameters`

# In[ ]:


clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=5, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(x_train, y_train)


# #### `Accuracy score`

# In[ ]:


clf_gini.score(x_test,y_test)


# #### `Plotting the tree`

# In[ ]:


dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# #### `Tree with max_depth = 3`

# In[ ]:


clf_gini = DecisionTreeClassifier(criterion = "gini", 
                                  random_state = 100,
                                  max_depth=3, 
                                  min_samples_leaf=50,
                                  min_samples_split=50)
clf_gini.fit(x_train, y_train)

# score
print('Score :', clf_gini.score(x_test,y_test))


# #### `Plotting tree with max_depth=3`

# In[ ]:


dot_data = StringIO()  
export_graphviz(clf_gini, out_file=dot_data,feature_names=features,filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# #### `Classification metrics`

# In[ ]:


y_pred = clf_gini.predict(x_test)
print(classification_report(y_test, y_pred))


# ### `SVM`

# #### `Normalize the data.`

# In[ ]:


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

x = NormalizeData(x)

'''Checking for Dimensions of the train and test sets'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.76809, random_state = 100)
print('x_train shape :', x_train.shape)
print('x_test shape :', x_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# #### `Support Vector Machine Model setup after parameter tuning`

# In[ ]:


model = svm.SVC(C=1, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',
  max_iter=-1, probability=False, random_state=None, shrinking=True,
  tol=0.001, verbose=False)
model.fit(x_train, y_train)


# #### `Print results of predicted to evaluate model`

# In[ ]:


print("Showing Performance Metrics for Support Vector Machine\n")

print ("Training Accuracy: {}".format(model.score(x_train, y_train)))
predicted = model.predict(x_test)
print ("Testing Accuracy: {}".format(accuracy_score(y_test, predicted)))

#### `Cross Validation Accuracy`print("Cross Validation Accuracy: \n")
cv_accuracy = cross_val_score(estimator=model, X=x_train, y=y_train, cv=10)
print("Accuracy using 10 folds: ")
print(cv_accuracy)#### ` Mean Accuracy and Standard Deviation`print("Mean accuracy: {}".format(cv_accuracy.mean()))
print("Standard Deviation: {}".format(cv_accuracy.std()))
# #### `Confusion Matrix`

# In[ ]:


print("Confusion Matrix for Support Vector Machine\n")
labels = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
cm = confusion_matrix(y_test, predicted, labels=labels)
print(cm)


# #### `Classification metrics`

# In[ ]:


print('Precision, Recall and f-1 Scores for Support Vector Machine\n')
print(classification_report(y_test, predicted))


# #### `Support Vector Machine model parameter tuning`

# In[ ]:


model = svm.SVC()

param_grid = [{'kernel': ['rbf'],
               'gamma': [1e-4, 0.01, 0.1],
               'C': [0.01, 1]}]


# #### `Finding optimum parameters through GridSearchCV`

# In[ ]:


print("Hyper Parameter Tuning Results\n")

grid = GridSearchCV(estimator=model, param_grid = param_grid,
                    cv = 5)
grid.fit(x_train, y_train)

print("\n")
print("Results returned by GridSearchCV\n")
print("Best estimator: ", grid.best_estimator_)
print("\n")
print("Best score: ", grid.best_score_)
print("\n")
print("Best parameters found: ", grid.best_params_)


# ### `KNN`

# #### `Normalize the data.`

# In[ ]:


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

x = NormalizeData(x)

'''Checking for Dimensions of the train and test sets'''

x_train, x_test, y_train, y_test = train_test_split(x, y, train_size = 0.76809, random_state = 100)
print('x_train shape :', x_train.shape)
print('x_test shape :', x_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)


# #### `Fit the Algorithm to the training sets`

# In[ ]:


knn = KNeighborsClassifier(n_neighbors = 13)
mod = knn.fit(x_train, y_train)


# #### `Predict the test data`

# In[ ]:


predictions = mod.predict(x_test)
predictions


# #### `Cross Validation Accuracy`

# In[ ]:


print("Cross Validation Accuracy: \n")
cv_accuracy = cross_val_score(estimator=mod, X=x_train, y=y_train, cv=10)
print("Accuracy using 10 folds: ")
print(cv_accuracy)


# #### ` Mean Accuracy and Standard Deviation`

# In[ ]:


print("Mean accuracy: {}".format(cv_accuracy.mean()))
print("Standard Deviation: {}".format(cv_accuracy.std()))


# #### `Confusion Matrix`

# In[ ]:


print("Confusion Matrix for Support Vector Machine\n")
labels = [0, 1, 2, 3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
cm = confusion_matrix(y_test, predictions, labels=labels)
print(cm)


# #### `Classification metrics`

# In[ ]:


print('Precision, Recall and f-1 Scores for Support Vector Machine\n')
print(classification_report(y_test, predictions))


# ### ` After Building the all type of Classification models and thier tuning models found that the Accuracy Scores as follows`

# #### `Logistic Regression Accuracy : "0.3262" on Training Set.`
# #### `Naive Bayes Accuracy : "0.3220" on Training Set.`

# #### `Decision Tree Accuracy : "0.3244" on Training Set.`
# #### `SVM Accuracy : "0.308" on Training Set.`
# #### `KNN Accuracy : "0.2833" on Training Set.`

# ### ` Finally We selected the "Decision Tree Model" because of it's having highest compared to others (i.e, There is a three decimal difference didn't considered)`

# ## `Decision Tree Prediction on Test Data`

# In[ ]:


test = pd.read_csv('test_CloudCondition.csv')
test = test.rename(columns = {"Temperature (C)":"Temperature", "Apparent Temperature (C)" : "Apparent Temperature", "Wind Speed (km/h)" : "Wind Speed", "Wind Bearing (degrees)" : "Wind Bearing", "Visibility (km)" : "Visibility", "Pressure (millibars)" : "Pressure"})
test1 = test


# In[ ]:


ord_enc = OrdinalEncoder()
test1[['Rain_OR_SNOW1', 'Condensation1']] = ord_enc.fit_transform(test1[['Rain_OR_SNOW', 'Condensation']])
test1[['Rain_OR_SNOW1', 'Condensation1']].head(11)


# In[ ]:


test1 = test1.drop(['Day','Rain_OR_SNOW', 'Condensation'], axis = 1)


# In[ ]:


test1.head()


# In[ ]:


y_pred = clf_gini.predict(test1)
print(y_pred)


# In[ ]:


test['Cloud_Condition'] = y_pred
test.head()


# In[ ]:


test['Cloud_Condition'] = test['Cloud_Condition'].replace([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],
                                                          ['Breezy', 'Breezy and Dry', 'Breezy and Foggy', 'Breezy and Mostly Cloudy',
                                                           'Breezy and Overcast', 'Breezy and Partly Cloudy', 'Clear', 'Dangerously Windy and Partly Cloudy',
                                                           'Drizzle', 'Dry', 'Dry and Mostly Cloudy', 'Dry and Partly Cloudy', 'Foggy', 'Humid and Mostly Cloudy',
                                                           'Humid and Overcast', 'Humid and Partly Cloudy', 'Light Rain', 'Mostly Cloudy', 'Overcast', 'Partly Cloudy',
                                                           'Windy', 'Windy and Dry', 'Windy and Foggy', 'Windy and Mostly Cloudy', 'Windy and Overcast', 'Windy and Partly Cloudy'])


# In[ ]:


test.head()


# In[ ]:


ftest = test[['Day', 'Cloud_Condition']]


# In[ ]:


ftest.to_csv('Cloud_Condition.csv', index = False)

