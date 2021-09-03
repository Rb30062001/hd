# import libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import math
from sklearn.model_selection import cross_validate


# read the csv file
dataset = pd.read_csv('heart.csv')

#copy the dataset
df = dataset.copy()

# make X and Y
X = df.drop(['target'], axis=1).values
Y = df.target.values


# correleation matrix
corr_mat = df.corr()


# split based on training and test dataset

x_train, x_test, y_train, y_test =   \
       train_test_split(X,Y,test_size =0.3,random_state=1234,stratify=Y)
       

# Logistic regression

lr = LogisticRegression()
lr.fit(x_train, y_train)

y_predict = lr.predict(x_test)

train_score = lr.score(x_train, y_train)
test_score = lr.score(x_test, y_test)


# accuracy score

acc_score = accuracy_score(y_test, y_predict)


rmse = math.sqrt(mean_squared_error(y_test, y_predict))


# Cross validation

lr_cross = LogisticRegression()

cv_results_lr = cross_validate(lr_cross, X, Y, cv=10, return_train_score=True)

test_cv_avg = np.average(cv_results_lr['test_score'])
train_cv_avg = np.average(cv_results_lr['train_score'])

pickle.dump(lr, open('model.pkl','wb'))


