from __future__ import print_function
import pandas as pd
import numpy as np
from numpy import array, dot, transpose
from numpy.linalg import inv
import scipy.io #for reading the mat file
from sklearn.model_selection import KFold #Split the data into kfolds for cross validation
import matplotlib.pyplot as plt
import random


random.seed(10)


def read_file(filename = "./data1.mat"):
    data = scipy.io.loadmat(filename)['data'] #load the data using scipy 
    df = pd.DataFrame({'X':data[:,0],'Y':data[:,1]}) #convert the np array to pandas for convenience
    return df

def transform_data(X_data,d):
    #f(x,theta) = theta_0 + theta_1.x + theta_2.x**2 + ..... theta_d.x**d
    new_data = np.ones((X_data.shape[0],d+1)) 
    for i in range(1,d+1):
        #set the column to be equal to X^i where i is the column number
        new_data[:,i] = np.power(X_data,i).transpose()
    return new_data

def process_data(data,folds,d = 1):
  
    X = np.array(data['X'])
    Y = np.array(data['Y'])

    #transform the data for the transformed linear regression
    X = transform_data(X,d)

    train_scores = [] #average score for the particular configuration of d
    test_scores = []

    kf = KFold(n_splits = folds)
    kf.get_n_splits(X)
    for train_index,test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        N_train = X_train.shape[0]
        N_test = X_test.shape[0]

        #closed form solution
        #w = (X^TX)^-1 X^T Y

        # w = np.matmul(np.matmul(np.linalg.inv(np.matmul(X_train.transpose(),X_train)),X_train.transpose()),Y_train)
        Xt = transpose(X_train)
        product = dot(Xt,X_train)
        theInverse = inv(product)
        w = dot(dot(theInverse,Xt),Y_train)
        
        #predict for train
        # y = W^T X
        y_train_pred = dot(X_train,transpose(w))
        y_test_pred = dot(X_test,transpose(w))

        #calculate the loss
        #loss = 1/2N (Y - W^TX)^2
        train_loss = (np.sum((Y_train - y_train_pred)**2)/N_train )*0.5
        test_loss = (np.sum((Y_test - y_test_pred)**2)/N_test) * 0.5
        train_scores.append(train_loss)
        test_scores.append(test_loss)
    print(w)
    return(np.mean(train_scores),np.mean(test_scores))

def optimize_hyperparameters(data,params):
    #optimize for params
    #params  = [1,2,3,4,...] specifying the degree of the polynomial

    #placeholder for the errors for different degrees of the polynomial
    tr_errors = []
    ts_errors = []

    #we perform 10-fold crossvalidation
    for param in params:
        print("Evaluating for the degree : ",param)
        train_loss,test_loss = process_data(data,10,param)
        tr_errors.append(train_loss)
        ts_errors.append(test_loss)
    return tr_errors,ts_errors




if __name__ == "__main__":
    print("Reading file!")
    df = read_file()
    print("Finding the optimal weights!")
    
    #degress of polynomial
    params = np.arange(0,20)
    tr_errors,ts_errors = optimize_hyperparameters(df,params)
    print("Training errors : ",tr_errors)
    print("Test errors : ",ts_errors)

    #plot the relation between the training and test errors
    plt.plot(params,tr_errors,color='b',label="Train error")
    plt.plot(params,ts_errors,color='r',label="Test error")
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Error")
    plt.title("Degree of polynomial and error")
    plt.legend(loc="best")
    # plt.savefig("combined.pdf", bbox_inches='tight')
    plt.show()

    plt.plot(params,tr_errors,color='b')
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Training Error")
    plt.title("Degree of polynomial and training error")
    # plt.savefig("training_error.pdf", bbox_inches='tight')

    plt.show()

    plt.plot(params,ts_errors,color='r')
    plt.xlabel("Degree of polynomial")
    plt.ylabel("Test Error")
    plt.title("Degree of polynomial and test error")
    # plt.savefig("test_error.pdf", bbox_inches='tight')
    plt.show()



#the maximum degree polynomial that can fit the data: 9