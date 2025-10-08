import numpy as np
class LinearRegression():
    """
    A simple implementation of  Linear Regression from scratch.  

    This class uses batch gradient descent to optimize the model parameters (weights and bias) 

    Attributes
    ----------
    learning_rate : float
        Step size for updating parameters during gradient descent (default: 0.001).
    epochs : int
        Number of iterations to run gradient descent (default: 1000).
    echoe : bool
        If True, prints training progress every 100 epochs.
    tolerance: int
        tolerance range for gradient decent (default value )
    weight : float
        Learned slope parameter of the regression line.
    bias : float
        Learned intercept parameter of the regression line.

    Methods
    -------
    fit(X, y):
        Train the model on feature X and target y using gradient descent.
    predict(X):
        Predict target values for given input X using learned parameters.
    mean_squared_error(X,y):
        Compute the mean squared error of the model prediction on data
    root_mean_squared_error(X, y):
        Compute the root mean squared error of the model predictions on data.
    mean_absolute_error(X,y):
        Compute the mean absolute error of the model prediction on data
    """
    def __init__(self,learning_rate=0.001,epochs=1000,tolerance=1e-6,echoe=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance =tolerance
        self.echoe  = echoe #if set to True will print progress of model training
        # default values not to be set manually during initialization
        self.weight =  None  
        self.bias = 0
    def _gradient_descent(self,X,y):
        """Private method for gradient descent (not meant to be called directly)."""
        X,y = np.array(X),np.array(y)
        # reshaping to a 2-D column vector if 1-D to allow matrix operations work well
        if X.ndim==1:
           X= X.reshape(-1,1)
        # using the shape of X  to determine the number of weights 
        n_samples,n_features = np.shape(X)
        w = np.zeros(n_features)
        b = self.bias

        prev_loss = float("inf")
        for epoch in range(1,self.epochs+1):
            y_predict =np.dot(X,w) + b
            error = y_predict-y
            # Gradients
            w_gradient = (2/n_samples)*np.dot(X.T,(error))
            b_gradient = (2/n_samples)*np.sum(error)
            # weights and biases update
            w = w - (self.learning_rate*w_gradient)
            b = b - (self.learning_rate*b_gradient)
            loss=np.mean(error**2)
            if self.echoe  and  epoch % 100 == 0:
                # prints progress of model training every 100 iterations if echoe is True
                print(f"Epoch {epoch}/{self.epochs} | Weights mean : {np.mean(w):.4f} | Bias: {b:.4f}")
            if epoch>10 and abs(prev_loss - loss) < self.tolerance:
                # if model has done more than 10 iterations and  loss is no longer changing by at least tolerance 
                if self.echoe:
                    print(
                        f"Early stopping at epoch {epoch} | "
                        f"Weight mean: {np.mean(w):.4f} | Bias: {b:.4f} | "
                        f"Loss: {loss:.6f}"
                    )
                break
            prev_loss=loss
        return w,b
    def fit(self,X,y):
        """Trains the model on the data using (Batch)Gradient descent """
        X,y = np.array(X),np.array(y)
        self.weight,self.bias = self._gradient_descent(X,y)
    def predict(self,X):
        X = np.array(X)
        # reshaping to a 2-D column vector if 1-D to allow matrix operations work well
        if X.ndim==1:
           X= X.reshape(-1,1)
        return np.dot(X,self.weight) + self.bias
    def root_mean_squared_error(self,X,y):
        X,y = np.array(X),np.array(y)
        rmse = np.sqrt(self.mean_squared_error(X,y))
        return rmse
    def mean_squared_error(self,X,y):
        X,y = np.array(X),np.array(y)
        y_predict = self.predict(X)
        mse = np.mean((y-y_predict)**2)
        return mse
    def mean_absolute_error(self,X,y):
        X,y = np.array(X),np.array(y)
        y_predict = self.predict(X)
        mae = np.mean(abs(y-y_predict))
        return mae