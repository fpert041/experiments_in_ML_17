#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:50:59 2017

@author: pesa
"""

import numpy as np
from collections import Counter

class MyKnn:
    
# =============================================================================
#   Variables
# =============================================================================
    
    training_set_X = [] # training datapoints
    training_set_y = [] # training labels (classes)
    num_neighbours = 0
    metric = ''  
    def distanceFunc():
        return 0
    
# =============================================================================
#   Define helper functions
# =============================================================================
        
#define distance functions: given two vectors (ndarrays), 
#this function returns the distance between them
#Write at least two distance functions, measuring the squared distance between 
#your data and the absolute value distance.
#You can implement both of these by looking at the numpy.linalg.norm method, 
#or implement your own version.  
        
    def euclideanDistance(self, in1,in2):
        result=0
        for i in range (0, len(in1)):
            result += (in1[i] - in2[i])**2
        return np.sqrt(result)

    def squaredDistance(self, in1,in2):
        result=0
        for i in range (0, len(in1)):
            result += (in1[i] - in2[i])**2
        return result

    def absDistance(self, in1,in2):
        result=0
        for i in range (0, len(in1)):
            result += abs(in1[i] - in2[i])
        return result


#The "get neighbours" function returns the nearest neighbour indices in X 
#of the test point x_.  In more detail:

# Input: x_ : point in test data

#       inX   : training data

#       n   : number of neighbours to return

# Output: n-nearest neighbours of x_ in training data X

    def getNeighbours(self, x_, inX, n):
        X = np.array(inX) # make sure X is ndarray
        distances=np.zeros(len(X)) # create an empty array of 0s
        for i in range(0, distances.size):
            distances[i] = self.distanceFunc(x_, X[i])
        sortedIndeces = np.argsort(distances)
        return sortedIndeces[:n] # indeces of n-nearest neighbours in training data
   
# The assign label function returns the assigned label for a test data point, 
#given the labels of nearest neighbours
# Input: nLabels : labels (classes) of nearest neighbours of a test point
# Output: the assigned label
# e.g., if we have n=1 (one neighbour), 
#then we can just return the label of the nearest neighbour
# else, we can e.g., choose the majority class
 
    def assignLabel(self, nLabels):        
        data = Counter(nLabels)
        mode = data.most_common(1)[0][0]
        return mode # label assigned to test point x_
    
# =============================================================================
#    End Helper Functions
# =============================================================================
    
    # ----- Initialisation parameters/methods -----
    
    def __init__(self, in_num_neighbours = 1, in_metric = 'euclidean') :
        self.num_neighbours = in_num_neighbours
        self.metric = in_metric
        if in_metric == 'euclidean':
            self.distanceFunc = self.squaredDistance
        else : 
            if in_metric == 'true_euclidean':
                self.distanceFunc = self.euclideanDistance
            else : 
                if in_metric == 'absolute':
                    self.distanceFunc = self.absDistance
                else:
                    print ( in_metric, 'did not match any distance metric')
                    print ( 
    'you should choose from "euclidean", "true_euclidean", and "absolute" ')
                    print ( ' defaulting to "euclidean" ')
                    self.distanceFunc = self.squaredDistance
            
        
    def fit(self, in_training_set_X, in_training_set_y):
        self.training_set_X = np.array(in_training_set_X)
        self.training_set_y = np.array(in_training_set_y)


    # ----- ML Utilities ------

    def predict(self, test_set_x):
        predictions = np.zeros(len(test_set_x), dtype=np.int8)
        
        ind = 0
        for i in test_set_x: #for all test points
            # knn classifier
            x_= i # test point x_        
    
            # get neighbours of x_ in training data 
            nIndeces = np.array(self.getNeighbours(x_, self.training_set_X, self.num_neighbours))
        
    
            # assignLabel to x_ based on neighbours
            predictions[ind] = self.assignLabel(self.training_set_y[nIndeces])
            ind += 1
            
        return predictions
    
    def accuracy_score(self, test_real_labels_y, predicted_labels):
         # evaluate if the assigned label is correct (equal to y_)
        count = 0.
        totNum = len(test_real_labels_y)
        for i in range (0, totNum):
            if test_real_labels_y[i] == predicted_labels[i]:
                count += 1.
        return count / totNum
    
