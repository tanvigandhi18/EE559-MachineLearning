# ################################################
# ## EE559 HW1, Prof. Jenkins
# ## Created by Arindam Jati
# ## Tested in Python 3.6.3, OSX El Capitan, and subsequent versions
# ################################################
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import Perceptron
from sklearn.model_selection import LearningCurveDisplay
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def Polynomial_Mapping(X,order):
    """
    Polynomial mapping function, you only have to implement the order < 4
    Args:
      X (ndarray (n,2)): data before polynomial mapping, n examples with 2 features
      order: largest exponent in the polynomial mapping
    Return:
      X_polynomial
    """
      
    if order == 1:
        # Order 1: Add bias term (x0=1) to original features
        return np.hstack((np.ones((X.shape[0], 1)), X))    #order 1= 1,x1,x2 = 3 terms
    elif order == 2:
        # Order 2: Original features, their squares, and interaction term, plus bias
        return np.hstack((np.ones((X.shape[0], 1)), X, X**2, X[:, 0:1] * X[:, 1:2]))   #order2 = 1,x1,x2,x1x2,x1^2,x2^2 = 6 terms
    elif order == 3:
        # Order 3: Includes terms up to cubic, all interactions, plus bias
        X1, X2 = X[:, 0:1], X[:, 1:2]  #order3 = (1,x1,x2,x1^2,x2^2,x1x2, x1^3,x2^3,x1x2^2,x1^2x2)
        return np.hstack((np.ones((X.shape[0], 1)), X, X**2, X1*X2, X**3, X1**2 * X2, X1 * X2**2)) #total 10 terms for a datapoint
    elif order>=4 and order<=7 or order == 10 or order == 11 or order == 15:
        poly = PolynomialFeatures(degree=order)
        X_poly = poly.fit_transform(X)   #order 4 = 15 features
        
        return X_poly

def accuracy(y_pred,y):
    error_count = 0
    for i in range(len(y)):
        if y_pred[i]!=y[i]:
            error_count+=1
    accuracy_score = 100 - ((error_count/len(y)) * 100)
        
    return accuracy_score


def plotDecBoundaries(training, label_train, perceptron, order):
    # Total number of classes
    nclass = max(np.unique(label_train))

    # Set the feature range for plotting
    max_x = np.ceil(max(training[:, 0])) + 1
    min_x = np.floor(min(training[:, 0])) - 1
    max_y = np.ceil(max(training[:, 1])) + 1
    min_y = np.floor(min(training[:, 1])) - 1

    xrange = (min_x, max_x)
    yrange = (min_y, max_y)

    # step size for how finely you want to visualize the decision boundary.
    inc = 0.005

    # Generate grid coordinates. This will be the basis of the decision
    # boundary visualization.
    (x, y) = np.meshgrid(np.arange(xrange[0], xrange[1]+inc/100, inc), np.arange(yrange[0], yrange[1]+inc/100, inc))

    # Flatten the grid so the perceptron can predict on it
    # Assuming `polynomial_features` is a fitted instance of sklearn's PolynomialFeatures
# and `order` is the degree of polynomial features you want to apply.

    # 1. Flatten the meshgrid
    grid_points = np.c_[x.ravel(), y.ravel()]

    # 2. Apply the polynomial transformation
    # Make sure `polynomial_features` is already fitted to your training data
    #transformed_grid_points = polynomial_features.transform(grid_points)
    transformed_grid_points = Polynomial_Mapping(grid_points, order)

    # 3. Predict using the perceptron
    grid_pred = perceptron.predict(transformed_grid_points)

# 4. Reshape the prediction back into the grid shape
    decisionmap = grid_pred.reshape(x.shape)

    
    # Reshape the prediction back into the grid shape
    decisionmap = grid_pred.reshape(x.shape)

    # Plot the decision map
    plt.contourf(x, y, decisionmap, alpha=0.5)
    
    # Plot the training points
    plt.scatter(training[:, 0], training[:, 1], c=label_train, edgecolor='k', s=20)
    
    plt.xlim(x.min(), x.max())
    plt.ylim(y.min(), y.max())
    plt.title("Decision Boundaries")
    plt.show()
 


