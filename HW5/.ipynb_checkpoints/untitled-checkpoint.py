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