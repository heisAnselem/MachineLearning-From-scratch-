import numpy as np
class LinearRegression():
    """
    A simple implementation of Univariate Linear Regression from scratch.  

    This class uses gradient descent to optimize the model parameters (weight and bias) 
    for predicting a continuous target variable from a single feature.

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
        Compute the mean_squared_error of the model prediction on data
    root_mean_squared_error(X, y):
        Compute the root mean squared error of the model predictions on data.
    """
    def __init__(self,learning_rate=0.001,epochs=1000,tolerance=1e-6,echoe=False):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.tolerance =tolerance
        self.echoe  = echoe #if set to True will print progress of model training
        # default values not to be set manually during initialization
        self.weight =  0  
        self.bias = 0
    def _gradient_descent(self,X,y):
        """Private method for gradient descent (not meant to be called directly)."""
        n = len(X)
        w = self.weight
        b = self.bias

        prev_loss = float("inf")
        for epoch in range(1,self.epochs+1):
            y_predict = w*X + b
            # Gradients
            w_gradient = (-2/n)*np.sum(X*(y-y_predict))
            b_gradient = (-2/n)*np.sum(y-y_predict)
            # weights and biases update
            w = w - (self.learning_rate*w_gradient)
            b = b - (self.learning_rate*b_gradient)
            loss=(1/n)*np.sum((y-y_predict)**2)
            if self.echoe  and  epoch % 100 == 0:
                # prints progress of model training every 100 iterations if echoe is True
                print(f"Epoch {epoch}/{self.epochs} | Weight: {w:.4f} | Bias: {b:.4f}")
            if epoch>10 and abs(prev_loss - loss) < self.tolerance:
                if self.echoe:
                    print(
                        f"Early stopping at epoch {epoch} | "
                        f"Weight: {w:.4f} | Bias: {b:.4f} | "
                        f"Loss: {loss:.6f}"
                    )
                break
            prev_loss=loss
        return w,b
    def fit(self,X,y):
        """Trains the model on the data using (Batch)Gradient descent """
        self.weight,self.bias = self._gradient_descent(X,y)
    def predict(self,X):
        return self.weight*X + self.bias
    def root_mean_squared_error(self,X,y):
        n = len(X)
        rmse = np.sqrt((1/n)*np.sum((y-self.predict(X))**2))
        return rmse
    def mean_squared_error(self,X,y):
        n = len(X)
        mse = (1/n)*np.sum((y-self.predict(X))**2)
        return mse

        
        