from __future__ import print_function
import numpy as np
from numpy import array, dot, transpose
from numpy.linalg import inv
import scipy.io #for reading the mat file
from sklearn.model_selection import KFold #Split the data into kfolds for cross validation
import matplotlib.pyplot as plt
import random
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.linear_model import SGDClassifier
import sys
from io import StringIO


random.seed(10)

#utility function
def read_file(filename = "./data2.mat"):
    data = scipy.io.loadmat(filename)['data'] #load the data using scipy 
    return data

#utility function
def plot_data(data):
    #The last column is the label/target
    # X = data.iloc[:,:-1]
    Y = data[:,-1]

    #get the examples from the two classes since this is binary classification
    class_0 = data[Y == 0]
    class_1 = data[Y == 1]

    #scatter plot of input data colored with classes
    plt.scatter(class_0[:,0],class_0[:,1],s=10,label="Class 0")
    plt.scatter(class_1[:,0],class_1[:,1],s=10,label="Class 1")
    plt.legend(loc="best")
    plt.title("2-dimensional data")
    plt.savefig("input_data.pdf", bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    #read the file
    data = read_file()

    plot_data(data)
    #extract the X and Y
    # X = data.iloc[:,:-1]
    X = data[:,:-1]
    Y = data[:,-1]


    #plot the data to check if there is a linear decision boundary
    plot_data(data)


    
    #sgd classifier

    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()

    #create an object
    logisticRegr = SGDClassifier(n_iter = 150,loss = 'log',penalty = 'none',fit_intercept= True,shuffle = True,verbose = True,warm_start = True)
    result = logisticRegr.fit(X,Y)

    #get the output of the fit method while training
    sys.stdout = old_stdout
    loss_history = mystdout.getvalue()
    loss_list = []
    for line in loss_history.split('\n'):
        if(len(line.split("loss: ")) == 1):
            continue
        loss_list.append(float(line.split("loss: ")[-1]))

    #plot the empricial risk with iterations    
    plt.figure()
    plt.plot(np.arange(len(loss_list)), loss_list)
    plt.xlabel("Number of iterations")
    plt.ylabel("Loss")
    plt.title("Empirical risk vs number of iteration")
    plt.savefig("Risk_time.pdf",bbox_inches='tight')
    plt.show()


    print("The intercept is: ",logisticRegr.intercept_)
    print("The coefficients are: ",logisticRegr.coef_)

    # #plot the decision boundary
    plt.figure(figsize=(10, 6))
    plt.scatter(X[Y == 0][:, 0], X[Y == 0][:, 1], color='b', label='Class 0')
    plt.scatter(X[Y == 1][:, 0], X[Y == 1][:, 1], color='r', label='Class 1')
    plt.legend()
    x1_min, x1_max = X[:,0].min(), X[:,0].max(),
    x2_min, x2_max = X[:,1].min(), X[:,1].max(),
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max), np.linspace(x2_min, x2_max))
    grid = np.c_[xx1.ravel(), xx2.ravel()]
    probs = logisticRegr.predict_proba(grid)[:,1].reshape(xx1.shape)
    plt.contour(xx1, xx2, probs, [0.5], linewidths=1, colors='black')
    plt.savefig("decision.pdf", bbox_inches='tight')
    plt.show()