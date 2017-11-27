#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 09:55:40 2017

@author: pesa
"""
'''
Task: Run the cell below to load our data. This piece of code is exactly the same as in the previous notebook.
'''

###%matplotlib inline###

from sklearn import datasets

import numpy as np

import matplotlib.pyplot as plt

#import k-nn classifier

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import operator


iris = datasets.load_iris()


#view a description of the dataset (uncomment next line to do so)
print(iris.DESCR)

#Set X equal to features, Y equal to the targets
X=iris.data  # measurements (datapoint)
y=iris.target  # classes to fit onto the measurements X


mySeed=1234567

#initialize random seed generator 
np.random.seed(mySeed)

#we add some random noise to our data to make the task more challenging

X=X+np.random.normal(0,0.5,X.shape)

'''
Task: The code below splits our data into two sets (a training and testing set),
    and subsequently trains a scikit-learn classifier 
    on the training data and tests on the testing data. 
To avoid complicating things, in this lab you just need to follow this setting, 
    no need to consider cross-validation. 
We are also using a fixed number of neighbours (10) and the euclidean distance. 
You can just run the cell below and make sure you understand the code 
- nothing else to do here.
'''

np.random.seed(mySeed)

# generate a random permutation of integers of the size
# of our dataset datapoints (X) and associated classes (y)
indices= np.random.permutation(X.shape[0])
bins=np.array_split(indices,5) # we  just need a training and testing set here

foldTrain=np.concatenate((bins[0],bins[1],bins[2],bins[3]), axis=0)
foldTest=bins[4]

# Set-up a knn classifier with 10 neighbours considered and euclidean distance
knn=KNeighborsClassifier(n_neighbors=10, metric='euclidean')

# Train our knn model by pairing X and y using the 'training' indeces
knn.fit(X[foldTrain],y[foldTrain])

# predict on the test indeces (return array of predictions)
y_pred=knn.predict(X[foldTest])

# print the accuracy of the predictions by comparing with the test data
print(accuracy_score(y[foldTest],y_pred))

'''
Task: Your task is now to implement your own version of k-NN, 
based on the lecture slides and the description given here. 
A suggested structure for doing this is included in the comments below, 
but feel free to start working in a different cell or in your favourite IDE. 
We are still using a simple training/test split (no cross-validation here) 
    to avoid complicating things, and thus use a fixed number of neighbours (10) 
    and the euclidean distance.
'''
## ANSWER HERE: Suggested code structure in comments below

# given a test point, your code should
# 
# - get the 'nearest neighbours' - i.e. the samples in the training set 
# - that are nearest to our test sample
# -----> done by evaluating the distance of the test sample to 
# all samples in the training set
# - assign a label to the test sample based on the 'neighbours'

##=== FUNCTION DEFINITIONS  ===##


#define distance functions: given two vectors (ndarrays), 
#this function returns the distance between them
#Write at least two distance functions, measuring the squared distance between 
#your data and the absolute value distance.
#You can implement both of these by looking at the numpy.linalg.norm method, 
#or implement your own version.  

def euclideanDistance(in1,in2):
    result=0
    for i in range (0, len(in1)):
        result += np.sqrt(in1[i] - in2[i])**2
    return result

def squaredDistance(in1,in2):
    result=0
    for i in range (0, len(in1)):
        result += (in1[i] - in2[i])**2
    return result

def absDistance(in1,in2):
    result=0
    for i in range (0, len(in1)):
        result += abs(in1[i] - in2[i])
    return result


#The "get neighbours" function returns the nearest neighbour indices in X 
#of the test point x_.  In more detail:

# Input: x_ : point in test data

#       X   : training data

#       n   : number of neighbours to return

#       T   : total number of training data

# Output: n-nearest neighbours of x_ in training data X


def getNeighbours(x_,inX,n): # where T is number of data
    
    X = np.array(inX) # make sure X is ndarray
    distances=np.zeros(X[:,0].size)
    for i in range(0, distances.size):
        distances[i] = squaredDistance(x_, X[i])
    distances=np.argsort(distances)
    return distances[:n] # indeces of n-nearest neighbours in training data



# The assign label function returns the assigned label for a test data point, 
#given the labels of nearest neighbours
# Input: nLabels : labels (classes) of nearest neighbours of a test point
# Output: the assigned label
# e.g., if we have n=1 (one neighbour), 
#then we can just return the label of the nearest neighbour
# else, we can e.g., choose the majority class

from collections import Counter
 
def assignLabel(nLabels):
    if nLabels.size == 1:
        return nLabels
        
    data = Counter(nLabels)
    mode = data.most_common(1)[0][0]
    return mode # label assigned to test point x_



##=== FUNCTION DEFINITIONS (END)  ===##


# here is some sample code for evaluating the kNN classifier you just built
# NOTE: this is just a suggested way to do this 
# - you can do it in another way if you want

correct=0;
nNumber = 10;

for i in foldTest: #for all test points

    # knn classifier
    x_=X[i] # test point x_
    y_=y[i] # true label for y_

    
    # get neighbours of x_ in training data 
    nIndeces = getNeighbours(x_, X, nNumber)
    
    # assignLabel to x_ based on neighbours
    testPrediction = assignLabel( y[nIndeces] )

    # evaluate if the assigned label is correct (equal to y_)
    if testPrediction == y_ :
        correct += 1

accuracy = correct / len(foldTest)

print(accuracy)