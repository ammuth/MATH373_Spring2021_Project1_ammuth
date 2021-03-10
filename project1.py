# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 10:37:55 2021

@author: lngsm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tqdm

         

def sigmoid(theta, X):
    return 1 / (1 + np.exp((-np.matmul(X,theta.transpose()))))

def calculate_cost(theta, X, y, lbda): # theta is dimensions n x 1, X is dimensions m x n, y is dimensions m x 1, lbda is regularization constant
    m = X.shape[0]
    h = sigmoid(theta, X)
    cost = (1/m)*(-y*np.log(h)-(1-y)*np.log(1-h)).sum() + (lbda / (2*m))*np.square(theta).sum()
    cost -= (lbda / (2*m)) * theta[0]**2 # remove contribution of theta_zero as it should not be included in cost
    return cost

def calculate_grad(theta, X, y, lbda):
    m = X.shape[0]
    h = sigmoid(theta, X)
    grad = np.matmul(X.transpose(),h - y) # vectorized implementation of gradient
    grad += (lbda/m) * theta
    grad[0] -= (lbda/m) * theta[0] # remove contribution of theta_zero as it should not be included in grad calculation
    return grad

def logistic_regression(X, y, alpha, iterations, test_X, test_y):
    theta = np.random.rand(X.shape[1]) # randomly initiates weights
    m = X.shape[0]
    costs_train = []
    costs_test = []
    for i in range(iterations):
        costs_train.append(calculate_cost(theta, X, y, 1))
        theta -= alpha * (1/m)*calculate_grad(theta, X ,y, 1)
        costs_test.append(calculate_cost(theta, test_X, test_y, 1))
    x_graph = np.arange(0,iterations,1);    
    plt.plot(x_graph,costs_train, label='train') 
    plt.plot(x_graph,costs_test, label='test')
    plt.legend()
    return theta 

def predict(theta, X, threshold):
    pred = sigmoid(theta, X)
    pred_result = (pred>=threshold).astype(int) # those above threshold = 1, 0 otherwise
    return pred_result

def normalize(X, mean, std):
    return (X-mean) / std




df = pd.read_csv('C:\\Users\\lngsm\\Documents\\math373\\DiabetesProject\\diabetes2.csv')
df.isnull().sum()
print(len(df[df['Outcome']==1]))
print(len(df[df['Outcome']==0])) # making sure data is balanced, a bit of imbalance but should be okay
train=df.sample(frac=0.75,random_state=150) #random state is a seed value
test=df.drop(train.index)

train_x = train.loc[:,train.columns != "Outcome"] # splitting dependent and independent variables
test_x = test.loc[:,test.columns != "Outcome"]
train_y = train['Outcome'].values
test_y = test['Outcome'].values
train_mean = train_x.mean(axis=0) # mean normalization
train_std = train_x.std(axis=0)
train_x = normalize(train_x,train_mean ,train_std)
test_x = normalize(test_x,train_mean ,train_std )

train_x.insert(0, 'One', 1) # adding column of ones for theta that is independent of features
test_x.insert(0, 'One', 1)      


theta = logistic_regression(train_x.values, train_y, 0.05, 500, test_x.values, test_y)

pred_y = predict(theta, test_x.values, 0.6)
result = pred_y == test_y
pred_train_y = predict(theta, train_x.values, 0.6)
result_train = pred_train_y == train_y
print(sum(result) / len(result))    
            
            
###############################################
#############PART 2############################
###############################################
            
train = pd.read_csv("C:\\Users\\lngsm\\Documents\\math373\\titanic_data\\train.csv")            
test = pd.read_csv("C:\\Users\\lngsm\\Documents\\math373\\titanic_data\\test.csv")            
  
def fill_nan(data, key, method = "mean"):
    if method == "mean":
        data[key].fillna(data["Age"].mean(), inplace = True)
    if method == "mode":
        data[key].fillna(data["Age"].mode()[0], inplace = True)
    if method == "median":
        data[key].fillna(data["Age"].median(), inplace = True)        
            
data_train_cleaned = train.copy(deep = True)
data_test_cleaned = test.copy(deep = True)

#calculate stats of our data
data_train_cleaned.describe(include = 'all')
data_test_cleaned.describe(include = 'all')

#clean data
#fill empty age
fill_nan(data_train_cleaned, "Age", "median")
fill_nan(data_test_cleaned, "Age", "median")

#fill empty embarked in train
data_train_cleaned["Embarked"].fillna(data_train_cleaned["Embarked"].mode()[0], inplace = True)

#fill empty fare in test
data_test_cleaned["Fare"].fillna(data_test_cleaned["Fare"].mean(), inplace = True)

data_train_cleaned = data_train_cleaned.drop("Cabin", axis = 1)
data_test_cleaned = data_test_cleaned.drop("Cabin", axis = 1)

data_train_cleaned = data_train_cleaned.drop(["PassengerId", "Name", "Ticket"], axis = 1)
data_test_cleaned = data_test_cleaned.drop(["PassengerId", "Name", "Ticket"], axis = 1)



#map Sex of a passenger to interger values , female : 0 , male : 1
data_train_cleaned['Sex'] = data_train_cleaned['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
data_test_cleaned['Sex'] = data_test_cleaned['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

#map embarked of a passenger to integer values S: 0, C : 1, Q : 2
data_train_cleaned['Embarked'] = data_train_cleaned['Embarked'].map({'S' : 0, 'C' : 1, 'Q': 2}).astype(int)
data_test_cleaned['Embarked'] = data_test_cleaned['Embarked'].map({'S' : 0, 'C' : 1, 'Q': 2}).astype(int)




#make a copy of our data to slice it
X_data = data_train_cleaned.copy(deep = True).values # .values converst pandas dataframe to a numpy array

#split data into train and val
X_train = X_data[:623] #70% of our training data 891 is ~ 623 values
X_val = X_data[623:] #30% of our training data is ~ 268 values

# labels are " survived " column of the dataset
Y_train = X_train[:,0]
Y_val = X_val[:,0]

#remove labels from dataset and only keep features
X_train = np.delete(X_train, 0, axis = 1)
X_val = np.delete(X_val, 0, axis = 1)

X_train = X_train.T
X_val = X_val.T

#fix our lable matrix
Y_train = Y_train.reshape((Y_train.shape[0], 1))
Y_val = Y_val.reshape((Y_val.shape[0], 1))

#sanity check
print("x train :" + str(X_train.shape))
print("x val :" + str(X_val.shape))
print("y train :" + str(Y_train.shape))
print("y val :" + str(Y_val.shape) )

def calc_stats(data):
    mu = data.mean(axis = 1, keepdims = True)
    sigma = data.std(axis = 1, keepdims = True)
    return mu, sigma

def standardize(data, mu, sigma):
    std_data = (data - mu) / sigma
    return std_data
mu, sigma = calc_stats(X_train)
X_train = standardize(X_train, mu, sigma)
X_val = standardize(X_val, mu, sigma)

#sanity check !
print(X_train.shape)
print(X_train[:5])

# initialize parameters
def initialize_parameters(dim):
    W = np.zeros((dim, 1))
    b = 0
    return W, b

#forward propagation
def forward_prop(X, W, b):
#     #sanity check
#     print("forward prop")
#     print("X shape:" + str(X.shape))
#     print("W shape:" + str(W.shape))
    Z = np.dot(W.T, X) + b
    A = sigmoid(Z)
    return A.T

#cost function
def compute_cost(Y, A):
#     #sanity check
#     print("computing cost")
#     print("Y shape:" + str(Y.shape))
#     print("A shape:" + str(A.shape))
    J = - np.sum(np.dot(Y.T, np.log(A)) + np.dot((1 - Y).T, np.log(1 - A)))/Y.shape[0]
    return J

def sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    return sig
# back prop function
def back_prop(X, Y, A):
    #sanity check
#     print("back_prop")
#     print("X shape:" + str(X.shape))
#     print("Y shape:" + str(Y.shape))
#     print("A shape:" + str(A.shape))
    dW = np.dot(X, (A-Y))/X.shape[1]
    db = np.sum(A - Y)/X.shape[1]
    
    return dW, db

#gradient descent
def gradient_descent(W, b, dW, db, learning_rate = 0.001):
    W = W - learning_rate * dW
    b = b - learning_rate * db
    return W, b

def logistic_regression(X, Y, num_iterations, learning_rate, print_cost = False, cost_graph = False):
    m = X_train.shape[1] #number of training examples
    W, b = initialize_parameters(X_train.shape[0]) #initialize learning parameters
    for i in tqdm.tqdm(range(num_iterations)):
        
        A = forward_prop(X, W, b)
        cost = compute_cost(Y, A)
        dW, db = back_prop(X, Y, A)
        W, b = gradient_descent(W, b, dW, db, learning_rate)
        
        # Record the costs
        if i % 100 == 0:
            costs.append(cost)
        # Print the cost every 100 training iterations
        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" %(i, cost))    
    if cost_graph == True:
        plt.plot(costs)
    return W, b

#make predictions !
def predict(X_val, W, b):
    predictions = forward_prop(X_val, W, b)
    
    #map predictions below 0.5 to 0 and above 0.5 to 1
    predictions[predictions > 0.5] = int(1)
    predictions[predictions < 0.5] = int(0)
    return predictions

# calculate accuracy
def test_accuracy(predictions, Y_val):
    accuracy = np.sum(predictions == Y_val)/predictions.shape[0]*100
    return accuracy

costs = [] #store cost to plot against iterations
W, b = logistic_regression(X_train, Y_train, 10000, 0.01, cost_graph = True)

#make predictions on our validation dataset
preds_train = predict(X_train, W, b)
preds_val = predict(X_val, W, b)
#calculate accuracy of our predictions
print(f"Accuracy on train data {test_accuracy(preds_train, Y_train)}%")
print(f"Accuracy on validation data {test_accuracy(preds_val, Y_val)}%")

X_test = data_test_cleaned.values

#standardize test dataset
X_test = X_test.T
X_test = standardize(X_test, mu, sigma)

#make predictions
predictions_test = predict(X_test, W, b).astype(int)

#compile into a dataframe
predictions_df = pd.DataFrame({ 'PassengerId': test["PassengerId"], 'Survived': predictions_test[:,0]})

#export dataframe as csv
predictions_df.to_csv("C:\\Users\\lngsm\\Documents\\math373\\Logistic_regression.csv", index = False)

predictions_df









