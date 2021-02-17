from statsmodels.tsa.arima.model import ARIMA
import pandas as pd
import matplotlib.pyplot as plt

def __init__(self, X, order, threshold):
    self.X = X
    self.order = order
    self.threshold = threshold
    self.model = ARIMA(X['value'], order=order)
    self.model_fit = None
    self.residual = None

def fit(self):
    # fit model
    self.model_fit = self.model.fit()
    self.residual = pd.DataFrame(self.model_fit.resid)

    # summary of fit model
    print(self.model_fit.summary())

def getResidual(self):
    return self.residual

def residualInfo(self):
    # plot of residuals
    self.X['residual'] = list(self.residual.loc[:,0])
    self.X.plot.scatter(x = 'timestamp', y = 'residual')
    plt.figure(figsize = (12, 6))
    plt.show()  

    # density plot of residuals
    self.residual.plot(kind = 'kde')
    plt.figure(figsize = (12, 6))
    plt.show()

    # summary stats of residuals
    print(self.residual.describe())

def getPrediction(self):
    return self.model_fit.predict()

def predictAnomaly(self):
    return abs(self.residual) > self.tgAhreshold