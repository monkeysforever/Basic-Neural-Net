# -*- coding: utf-8 -*-
"""
This program performs logistic regression on a dataset of images of cats and dogs and outputs a classifier which can classify an image as a dog or cat.
The code can also be reused for other datasets with some minor changes

@author: Randeep
"""

import numpy as np
from sklearn.datasets import make_moons, make_circles, make_blobs
from matplotlib import pyplot
from pandas import DataFrame

SPLIT_RATIO = 90

def sigmoid(x):
    #The function returns values ranging from 0 to 1, which helps to quantify the probability of the image being of a cat
    #The argument x is expected to be a  numpy array
    return (1/(1+np.exp(-x)))

def init_parameters(dimensions):
    w1 = np.random.randn(dimensions[1], dimensions[0])
    b1 = np.zeros((dimensions[1], 1))
    w2 = np.random.randn(dimensions[2], dimensions[1])
    b2 = np.zeros((dimensions[2], 1))
    parameters = {'w1' : w1,
                  'b1' : b1,
                  'w2' : w2,
                  'b2' : b2}
    return parameters

def forward_propagation(X, Y, parameters):    
    m = X.shape[1]
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    Z1 = np.matmul(w1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(w2, A1) + b2
    A2 = sigmoid(Z2)
    
    Cost = -(np.matmul(Y, np.log(A2).T) + np.matmul(1 - Y, np.log(1 - A2).T))/m
    Cost = np.squeeze(Cost)
    
    forward_cache = {'A1' : A1,                     
                     'A2' : A2}
    return Cost, forward_cache

def back_propagation(X, Y, forward_cache, w2):
    m = X.shape[1]    
    A1 = forward_cache['A1']    
    A2 = forward_cache['A2']
    
    dZ2 = A2 - Y
    dw2 = np.matmul(dZ2, A1.T)/m
    db2 = np.sum(dZ2, axis = 1, keepdims = True)/m
    dZ1 = np.multiply(np.matmul(w2.T, dZ2), 1 - np.power(A1, 2))
    dw1 = np.matmul(dZ1, X.T)/m
    db1 = np.sum(dZ1, axis = 1, keepdims = True)/m
    
    gradients = {'dw2' : dw2,
                 'dw1' : dw1,
                 'db2' : db2,
                 'db1' : db1}
    
    return gradients

def predict(X, parameters):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']
    
    Z1 = np.matmul(w1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.matmul(w2, A1) + b2
    A2 = sigmoid(Z2)

    predictions = (A2 > 0.5).astype(int)
    return predictions



def gradient_descent(X, Y, parameters, num_iterations, learning_rate, print_iteration):
    Costs = []
    for i in range(num_iterations):
        Cost, forward_cache =  forward_propagation(X, Y, parameters)
        gradients = back_propagation(X, Y, forward_cache, parameters['w2'])        
        parameters['w1'] -= learning_rate * gradients['dw1']
        parameters['b1'] -= learning_rate * gradients['db1']
        parameters['w2'] -= learning_rate * gradients['dw2']
        parameters['b2'] -= learning_rate * gradients['db2']
        if i != 0 and i % print_iteration == 0:
            Costs.append(Cost)        
            print ("Cost after iteration %i: %f" %(i, Cost))
    return Costs, parameters    
    

def neural_net(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate, dimensions, print_iteration = 0):
    parameters = init_parameters(dimensions)
    Costs, parameters = gradient_descent(X_train, Y_train, parameters, num_iterations, learning_rate, print_iteration)
    Y_train_predictions = predict(X_train, parameters)
    Y_test_predictions = predict(X_test, parameters)
    training_accuracy = 100 - np.mean(np.abs(Y_train - Y_train_predictions))*100
    test_accuracy = 100 - np.mean(np.abs(Y_test - Y_test_predictions))*100
    model = {'training_accuracy' : training_accuracy,
             'test_accuracy' : test_accuracy,
             'parameters' : parameters,
             'learning_rate' : learning_rate,
             'training_predictions' : Y_train_predictions,
             'test_predictions' : Y_test_predictions,
             'iterations' : num_iterations,
             'Costs' : Costs}
    return model
    
def generate_dataset(examples_count, dataset_type = 'moons'):
    #This function generates different types of data sets and divides the dataset into training and test sets
    #The datasets supported are blobs, moons and circles
    #Shape of X for test and training is (number of features, examples)
    #Shape of Y for test and training is (1, examples)    
    if dataset_type == 'blobs':
        X, y = make_blobs(n_samples=examples_count, centers=2, n_features=2)        
    elif dataset_type == 'moons':
        X, y = make_moons(n_samples=examples_count, noise=0.1)
    elif dataset_type == 'circles':
        X, y = make_circles(n_samples=examples_count, noise=0.05)
        
    X = X.T
    y = np.reshape(y, (1, y.shape[0]))   
    split_index = X.shape[1]*SPLIT_RATIO//100    
    indices = np.random.permutation(X.shape[1])
    training_idx, test_idx = indices[:split_index], indices[split_index:]    
    X_training, X_test, Y_training, Y_test = X[:, training_idx], X[:, test_idx], y[:, training_idx], y[:, test_idx]    
    return X_training, Y_training, X_test, Y_test  


def plot_dataset(X, Y):
    #This function plots the dataset        
    df = DataFrame(dict(x=X[0,:], y=X[1,:], label=Y[0, :]))
    colors = {0:'red', 1:'blue'}
    fig, ax = pyplot.subplots()    
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    pyplot.show()
    
def plot_cost(Costs):
    pyplot.plot(Costs)
    pyplot.show()

#Steps to create classifier
    
#1.Load the dataset, remember to put in right filepath and not load too many images    
X_training, Y_training, X_test, Y_test = generate_dataset(5000, 'moons')


#4.Train your model
model = neural_net(X_training, Y_training, X_test, Y_test, 5000, 0.05, [2, 4, 1], 200)

print(model['training_accuracy'])
print(model['test_accuracy'])
plot_dataset(X_training, Y_training)
plot_dataset(X_training, model['training_predictions'])
plot_cost(model['Costs'])

#Use predict functions with weigths and bias after training to make predictions
#Train and test accuracy can be improved by changing the values of hyperparameters learning_rate, num_iterations, file_count
#Try to plot the cost values with number of iterations to get an intuition about the reducing cost
#If your cost fluctuates up and down reduce learning_rate

#97.111111
#95.6
#10000 iterations
#.1 
#5000 samples

#98.888888888
#99.2
#1000
#.1
#5000

#98.733333333
99
5000
.05
5000