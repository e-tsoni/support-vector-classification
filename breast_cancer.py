# -*- coding: utf-8 -*-
"""
@author: Eftychia Tsoni

"""

#making the necessary imports
from __future__ import print_function, division
import numpy as np 
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import statistics
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn import neighbors
from sklearn.decomposition import KernelPCA

# get_score for fit the model in cross validation and get a different accuracy in every loop
def get_score (model, Xtrain, Xtest, Ytrain, Ytest):
    model.fit(Xtrain, Ytrain)
    return model.score(Xtest, Ytest)

### EXPLORING THE DATA ###

# The data comes with SKlearn. 
# We load the data and we print to see the data, the targets and the features.
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.DESCR)
# print target names
print(breast_cancer_data.target_names)
# print data features
print(breast_cancer_data.feature_names)
# print features values for all instances (569)
print(breast_cancer_data.data)

### SPLIT, STANDARDIZATION, FIT AND TEST ###

# spliting the dataset into train and test set. 
# 60% for train and 40% for the test
X=breast_cancer_data.data
Y=breast_cancer_data.target
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.40, random_state = 0)
# scale the data
scaler = StandardScaler()
# X=scaler.fit_transform(X) -- we could scale all the data first before splitting --
Xtrain = scaler.fit_transform(Xtrain)
Xtest = scaler.transform(Xtest)

# Create the model = SVC() with linear kernel and default parameters
model = SVC(kernel='linear', gamma='auto')
# fit the model and evaluate
print("Model evaluation with linear Kernel and default parameters: ")
print("\n")
t0 = datetime.now()
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))
t1 = datetime.now()
print("Time = ",t1-t0)
print("\n")

# Create the model = SVC() with rbf kernel and default parameters
# fit the model and evaluate
print("Model evaluation with rbf Kernel and default parameters: ")
print("\n")
t0 = datetime.now()
model = SVC(kernel='rbf', gamma='auto')
model.fit(Xtrain, Ytrain)
print("train score:", model.score(Xtrain, Ytrain))
print("test score:", model.score(Xtest, Ytest))
t1 = datetime.now()
print("Time = ",t1-t0)
print("\n")

### USING CROSS VALIDATION ###

# split the data using k-fold method and a loop "for'
# using again the linear classifier
model = SVC(kernel='linear', gamma='auto')
kf=StratifiedKFold(n_splits=10)
print(kf)
#create a list to store the accuracy 
score=[]

# fit the model and evaluate showing the time needed
t0 = datetime.now()
for train_index, test_index in kf.split(breast_cancer_data.data, breast_cancer_data.target):
    Xtrain, Xtest, Ytrain, Ytest = breast_cancer_data.data[train_index], breast_cancer_data.data[test_index], \
    breast_cancer_data.target[train_index],breast_cancer_data.target[test_index] 
    score.append(get_score(model, Xtrain, Xtest, Ytrain, Ytest))
t1 = datetime.now()
print("Split the data using k-fold method (k=10) and default parameters: ")
print("\n")
print(score)
print("Accuracy (mean) = ",statistics.mean(score))
print("Time = ",t1-t0)
print("\n")
# split the data using cross_val_score()
t0 = datetime.now()
scores=cross_val_score(model, X, Y, cv=10, scoring="accuracy")
t1 = datetime.now()
print("Split the data using cross_val_score()")
print("\n")
print(scores)
print("Accuracy (mean) = ",scores.mean())
print("Time = ", t1-t0)
print("\n")

### PARAMETERS TUNING ###

# parameter tuning for rbf kernel using "for" loops.
C=[0.5, 10, 100, 1000]
gammas=[0.0001, 0.2, 10, 100]
cvp_scores=[]
t0 = datetime.now()
for c in C:
    for g in gammas:
        model = SVC(kernel='rbf', C=c, gamma=g)
        print("C= ",c," gamma= ", g)
        print("--------------------------------------------------------------")
        scores_cv=cross_val_score(model, X, Y, cv=10, scoring="accuracy")
        cvp_scores.append(scores_cv.mean())
        print("mean = ", scores_cv.mean())
        print("\n")
t1 = datetime.now()
print("max= ",max(cvp_scores))
print("Time = ", t1-t0)

### DATA VISUALIZATION WITH PCA ###

# data visualization in 2D using PCA
print(Y.shape) #(569,)
print(X.shape) #(569,30)
# reshape the Y labels to concatenate it with the X data,
# so that we can create a DataFrame.
labels = np.reshape(Y,(569,1))
# concatenate the data and labels so the final shape of the array will be 569 x 31
X_new = np.concatenate([X,labels],axis=1)
print(X_new.shape) # (569,31)
# create the DataFrame
breast_cancer_df = pd.DataFrame(X_new)
print(breast_cancer_df)
# print the features
features = breast_cancer_data.feature_names
# manually add the label field to the features array
features_labels = np.append(features,'label')
# embed the column names to the breast_dataset dataframe
breast_cancer_df.columns = features_labels
# print the first few rows of the dataframe.
print(breast_cancer_df.head())
# we change the labels 0 and 1 to benign and malignant. 0='Benign', 1='Malignant'
breast_cancer_df['label'].replace(0, 'Benign',inplace=True)
breast_cancer_df['label'].replace(1, 'Malignant',inplace=True)
#print the last few rows of the breast_dataset 
breast_cancer_df.tail()
# print(X) 
# normalize data
X_scaled = breast_cancer_df.loc[:, features].values
X_scaled = StandardScaler().fit_transform(X_scaled)
print(np.mean(X_scaled),np.std(X_scaled))
print(X_scaled)
# convert the normalized features into an array format
# rename the colums (feature0, feature1, feature2,...e.t.c)
features_cols = ['feature'+str(i) for i in range(X_scaled.shape[1])]
# create the data frame with normalized values of the features
normalised_breast_df = pd.DataFrame(X_scaled,columns=features_cols) 
# print the first part of the data frame to see the resaults
print(normalised_breast_df.head())
# we will project the 30 dimentions of breast cancer data to 2 dimensional principal components
# we want to keep 2 principal components out of 30
# we call fit_transform on the aggregate data.
pca_breast_cancer = PCA(n_components=2)
principalComponents_breast = pca_breast_cancer.fit_transform(X_scaled)
print(principalComponents_breast)
print("\n")
# create a DataFrame with principal components
principal_breast_Df = pd.DataFrame(data = principalComponents_breast
             , columns = ['principal component 1', 'principal component 2'])
print(principal_breast_Df.tail())
print(principal_breast_Df.shape)
# find the explained_variance_ratio. 
# It will provide the amount of information or variance each principal component holds 
# after projecting the data to 2 dimensional subspace.
print('Explained variation per principal component: {}'.format(pca_breast_cancer.explained_variance_ratio_))
# 2D visualization of the 569 samples
plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Principal Component - 1',fontsize=18)
plt.ylabel('Principal Component - 2',fontsize=18)
plt.title("Principal Component Analysis of Breast Cancer Dataset",fontsize=18)
targets = ['Benign', 'Malignant']
colors = ['b', 'r']
for target, color in zip(targets,colors):
    indicesToKeep = breast_cancer_df['label'] == target
    plt.scatter(principal_breast_Df.loc[indicesToKeep, 'principal component 1']
               , principal_breast_Df.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)
plt.legend(targets,prop={'size': 15})




