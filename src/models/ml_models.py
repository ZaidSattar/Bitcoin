import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import seaborn as sns

# Try to import XGBoost, but make it optional
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. XGBoost models will be skipped.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RegressionModel:
    """
    Class for regression models to predict Bitcoin prices
    """
    
    def __init__(self, model_type='linear', **kwargs):
        """
        Initialize regression model
        
        Args:
            model_type (str): Type of regression model
                              ('linear', 'ridge', 'lasso', 'rf', 'gb', 'xgb')
            **kwargs: Additional arguments for the specific model
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = self._get_model()
        self.feature_importances_ = None
        
    def _get_model(self):
        """
        Get the regression model based on model_type
        
        Returns:
            sklearn model: Regression model
        """
        if self.model_type == 'linear':
            return LinearRegression(**self.kwargs)
        elif self.model_type == 'ridge':
            return Ridge(**self.kwargs)
        elif self.model_type == 'lasso':
            return Lasso(**self.kwargs)
        elif self.model_type == 'rf':
            return RandomForestRegressor(**self.kwargs)
        elif self.model_type == 'gb':
            return GradientBoostingRegressor(**self.kwargs)
        elif self.model_type == 'xgb':
            if XGBOOST_AVAILABLE:
                return xgb.XGBRegressor(**self.kwargs)
            else:
                raise ValueError("XGBoost is not available")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """
        Fit the regression model to the data
        
        Args:
            X (array-like): Features
            y (array-like): Target
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Fitting {self.model_type} regression model")
        self.model.fit(X, y)
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
            
        return self
    
    def predict(self, X):
        """
        Make predictions with the model
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Predictions
        """
        logger.info(f"Making predictions with {self.model_type} regression model")
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data
        
        Args:
            X (array-like): Features
            y (array-like): Actual values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {self.model_type} regression model")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"Regression evaluation: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred,
            'actuals': y
        }
    
    def plot_predictions(self, X, y, title=None):
        """
        Plot actual vs predicted values
        
        Args:
            X (array-like): Features
            y (array-like): Actual values
            title (str, optional): Plot title
            
        Returns:
            matplotlib.figure.Figure: Predictions plot
        """
        predictions = self.predict(X)
        
        plt.figure(figsize=(10, 6))
        plt.scatter(y, predictions, alpha=0.5)
        
        # Plot perfect prediction line
        min_val = min(min(y), min(predictions))
        max_val = max(max(y), max(predictions))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title(title or f'{self.model_type.upper()} Regression: Actual vs Predicted')
        plt.grid(True)
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_names, top_n=20):
        """
        Plot feature importances
        
        Args:
            feature_names (list): Names of features
            top_n (int): Number of top features to show
            
        Returns:
            matplotlib.figure.Figure: Feature importance plot
        """
        if self.feature_importances_ is None:
            logger.warning("Model doesn't have feature importances")
            return None
        
        # Create DataFrame of feature importances
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        })
        
        # Sort by importance
        importances = importances.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importances)
        plt.title(f'Top {top_n} Feature Importances ({self.model_type.upper()})')
        plt.tight_layout()
        
        return plt.gcf()


class ClassificationModel:
    """
    Class for classification models to predict Bitcoin price direction
    """
    
    def __init__(self, model_type='logistic', **kwargs):
        """
        Initialize classification model
        
        Args:
            model_type (str): Type of classification model
                              ('logistic', 'rf', 'gb', 'xgb')
            **kwargs: Additional arguments for the specific model
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = self._get_model()
        self.feature_importances_ = None
        
    def _get_model(self):
        """
        Get the classification model based on model_type
        
        Returns:
            sklearn model: Classification model
        """
        if self.model_type == 'logistic':
            return LogisticRegression(**self.kwargs)
        elif self.model_type == 'rf':
            return RandomForestClassifier(**self.kwargs)
        elif self.model_type == 'gb':
            return GradientBoostingClassifier(**self.kwargs)
        elif self.model_type == 'xgb':
            if XGBOOST_AVAILABLE:
                return xgb.XGBClassifier(**self.kwargs)
            else:
                raise ValueError("XGBoost is not available")
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X, y):
        """
        Fit the classification model to the data
        
        Args:
            X (array-like): Features
            y (array-like): Target
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Fitting {self.model_type} classification model")
        self.model.fit(X, y)
        
        # Store feature importances if available
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
            
        return self
    
    def predict(self, X):
        """
        Make predictions with the model
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Predictions
        """
        logger.info(f"Making predictions with {self.model_type} classification model")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Class probabilities
        """
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)
        else:
            logger.warning(f"{self.model_type} does not support predict_proba")
            return None
    
    def evaluate(self, X, y):
        """
        Evaluate the model on test data
        
        Args:
            X (array-like): Features
            y (array-like): Actual values
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        logger.info(f"Evaluating {self.model_type} classification model")
        
        # Make predictions
        y_pred = self.predict(X)
        
        # Get probabilities if possible
        y_prob = None
        if hasattr(self.model, 'predict_proba'):
            y_prob = self.predict_proba(X)
            
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='binary')
        recall = recall_score(y, y_pred, average='binary')
        f1 = f1_score(y, y_pred, average='binary')
        
        # ROC AUC if probabilities are available
        roc_auc = None
        if y_prob is not None:
            roc_auc = roc_auc_score(y, y_prob[:, 1])
        
        logger.info(f"Classification evaluation: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': confusion_matrix(y, y_pred),
            'classification_report': classification_report(y, y_pred),
            'predictions': y_pred,
            'probabilities': y_prob,
            'actuals': y
        }
    
    def plot_confusion_matrix(self, y_true, y_pred=None, title=None):
        """
        Plot confusion matrix
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like, optional): Predicted labels (if None, will be calculated)
            title (str, optional): Plot title
            
        Returns:
            matplotlib.figure.Figure: Confusion matrix plot
        """
        if y_pred is None:
            y_pred = self.predict(X)
            
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(title or f'{self.model_type.upper()} Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        return plt.gcf()
    
    def plot_feature_importance(self, feature_names, top_n=20):
        """
        Plot feature importances
        
        Args:
            feature_names (list): Names of features
            top_n (int): Number of top features to show
            
        Returns:
            matplotlib.figure.Figure: Feature importance plot
        """
        if self.feature_importances_ is None:
            logger.warning("Model doesn't have feature importances")
            return None
        
        # Create DataFrame of feature importances
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        })
        
        # Sort by importance
        importances = importances.sort_values('importance', ascending=False).head(top_n)
        
        # Plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=importances)
        plt.title(f'Top {top_n} Feature Importances ({self.model_type.upper()})')
        plt.tight_layout()
        
        return plt.gcf() 