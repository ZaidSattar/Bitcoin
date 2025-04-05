import numpy as np
import pandas as pd
import logging
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ARIMAModel:
    """
    ARIMA Model for forecasting Bitcoin prices
    """
    
    def __init__(self, order=(5, 1, 0), seasonal_order=None):
        """
        Initialize ARIMA model
        
        Args:
            order (tuple): ARIMA order (p, d, q)
            seasonal_order (tuple, optional): Seasonal order for SARIMA
        """
        self.order = order
        self.seasonal_order = seasonal_order
        self.model = None
        self.results = None
        
    def fit(self, series):
        """
        Fit ARIMA model to the data
        
        Args:
            series (pd.Series): Time series data to fit
        """
        logger.info(f"Fitting ARIMA model with order {self.order}")
        
        try:
            if self.seasonal_order:
                self.model = SARIMAX(series, 
                                     order=self.order, 
                                     seasonal_order=self.seasonal_order,
                                     enforce_stationarity=False,
                                     enforce_invertibility=False)
                logger.info(f"Using SARIMA with seasonal_order {self.seasonal_order}")
            else:
                self.model = ARIMA(series, order=self.order)
                
            self.results = self.model.fit()
            logger.info("ARIMA model fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting ARIMA model: {e}")
            raise
    
    def predict(self, steps=1, dynamic=False):
        """
        Make predictions with the fitted model
        
        Args:
            steps (int): Number of steps to forecast
            dynamic (bool): Whether to use dynamic forecasting
            
        Returns:
            pd.Series: Predicted values
        """
        if self.results is None:
            raise ValueError("Model must be fitted before making predictions")
            
        logger.info(f"Making predictions with ARIMA for {steps} steps")
        
        try:
            forecast = self.results.forecast(steps=steps, dynamic=dynamic)
            return forecast
            
        except Exception as e:
            logger.error(f"Error in ARIMA prediction: {e}")
            raise
    
    def evaluate(self, test_data, dynamic=False):
        """
        Evaluate the model on test data
        
        Args:
            test_data (pd.Series): Test data
            dynamic (bool): Whether to use dynamic forecasting
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.results is None:
            raise ValueError("Model must be fitted before evaluation")
            
        logger.info("Evaluating ARIMA model")
        
        try:
            # Get predictions for test period
            predictions = self.results.forecast(steps=len(test_data), dynamic=dynamic)
            
            # Calculate metrics
            mse = mean_squared_error(test_data, predictions)
            rmse = math.sqrt(mse)
            mae = mean_absolute_error(test_data, predictions)
            
            # Calculate MAPE (Mean Absolute Percentage Error)
            mape = np.mean(np.abs((test_data - predictions) / test_data)) * 100
            
            logger.info(f"ARIMA evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, MAPE={mape:.4f}%")
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'mape': mape,
                'predictions': predictions,
                'actuals': test_data
            }
            
        except Exception as e:
            logger.error(f"Error in ARIMA evaluation: {e}")
            raise
    
    def plot_forecast(self, train_data, test_data, predictions, title='ARIMA Forecast'):
        """
        Plot the forecast against actual values
        
        Args:
            train_data (pd.Series): Training data
            test_data (pd.Series): Test data
            predictions (pd.Series): Predictions
            title (str): Plot title
        """
        plt.figure(figsize=(12, 6))
        
        # Plot training data
        plt.plot(train_data.index, train_data, label='Training Data')
        
        # Plot test data
        plt.plot(test_data.index, test_data, label='Actual', color='blue')
        
        # Plot predictions
        plt.plot(test_data.index, predictions, label='Predicted', color='red', linestyle='--')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True)
        
        return plt.gcf() 