import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Class for evaluating and comparing different models
    """
    
    def __init__(self):
        """
        Initialize ModelEvaluator
        """
        self.regression_results = {}
        self.classification_results = {}
        self.time_series_results = {}
        self.clustering_results = {}
        self.anomaly_results = {}
        
    def evaluate_regression_model(self, model_name, y_true, y_pred):
        """
        Evaluate regression model and store results
        
        Args:
            model_name (str): Name of the model
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        logger.info(f"Regression evaluation for {model_name}: "
                   f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
                   f"R2={metrics['r2']:.4f}, MAPE={metrics['mape']:.2f}%")
        
        self.regression_results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        return metrics
    
    def evaluate_classification_model(self, model_name, y_true, y_pred, y_proba=None):
        """
        Evaluate classification model and store results
        
        Args:
            model_name (str): Name of the model
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            y_proba (array-like, optional): Predicted probabilities
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary'),
            'recall': recall_score(y_true, y_pred, average='binary'),
            'f1': f1_score(y_true, y_pred, average='binary'),
            'confusion_matrix': confusion_matrix(y_true, y_pred)
        }
        
        # ROC AUC if probabilities are available
        if y_proba is not None and y_proba.ndim > 1:
            from sklearn.metrics import roc_auc_score
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
        
        logger.info(f"Classification evaluation for {model_name}: "
                   f"Accuracy={metrics['accuracy']:.4f}, Precision={metrics['precision']:.4f}, "
                   f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}")
        
        self.classification_results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_proba': y_proba
        }
        
        return metrics
    
    def evaluate_time_series_model(self, model_name, y_true, y_pred):
        """
        Evaluate time series model and store results
        
        Args:
            model_name (str): Name of the model
            y_true (array-like): True values
            y_pred (array-like): Predicted values
            
        Returns:
            dict: Evaluation metrics
        """
        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        logger.info(f"Time series evaluation for {model_name}: "
                   f"RMSE={metrics['rmse']:.4f}, MAE={metrics['mae']:.4f}, "
                   f"MAPE={metrics['mape']:.2f}%")
        
        self.time_series_results[model_name] = {
            'metrics': metrics,
            'y_true': y_true,
            'y_pred': y_pred
        }
        
        return metrics
    
    def store_clustering_results(self, model_name, labels, evaluation_metrics, model=None):
        """
        Store clustering model results
        
        Args:
            model_name (str): Name of the model
            labels (array-like): Cluster labels
            evaluation_metrics (dict): Evaluation metrics
            model (object, optional): Clustering model object
        """
        self.clustering_results[model_name] = {
            'labels': labels,
            'metrics': evaluation_metrics,
            'model': model
        }
        
        logger.info(f"Stored clustering results for {model_name}: {len(np.unique(labels))} clusters")
    
    def store_anomaly_results(self, model_name, anomalies, scores=None, model=None):
        """
        Store anomaly detection results
        
        Args:
            model_name (str): Name of the model
            anomalies (array-like): Anomaly labels (boolean)
            scores (array-like, optional): Anomaly scores
            model (object, optional): Anomaly detection model object
        """
        self.anomaly_results[model_name] = {
            'anomalies': anomalies,
            'scores': scores,
            'model': model,
            'num_anomalies': np.sum(anomalies),
            'anomaly_rate': np.mean(anomalies)
        }
        
        logger.info(f"Stored anomaly detection results for {model_name}: "
                   f"{np.sum(anomalies)} anomalies ({np.mean(anomalies):.2%})")
    
    def compare_regression_models(self, metric='rmse'):
        """
        Compare regression models based on a specific metric
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            pd.DataFrame: Comparison of models
        """
        if not self.regression_results:
            logger.warning("No regression models to compare")
            return None
        
        results = {}
        for model_name, model_results in self.regression_results.items():
            results[model_name] = model_results['metrics']
            
        df = pd.DataFrame(results).T
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        ax = df[metric].sort_values().plot(kind='bar')
        plt.title(f'Regression Models Comparison - {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xlabel('Model')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(df[metric].sort_values()):
            ax.text(i, v + (v * 0.01), f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        
        return df
    
    def compare_classification_models(self, metric='f1'):
        """
        Compare classification models based on a specific metric
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            pd.DataFrame: Comparison of models
        """
        if not self.classification_results:
            logger.warning("No classification models to compare")
            return None
        
        results = {}
        for model_name, model_results in self.classification_results.items():
            results[model_name] = model_results['metrics']
            
        df = pd.DataFrame(results).T
        
        # Filter numeric columns for plotting
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        plot_df = df[numeric_cols].drop('confusion_matrix', axis=1, errors='ignore')
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        ax = plot_df[metric].sort_values(ascending=False).plot(kind='bar')
        plt.title(f'Classification Models Comparison - {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xlabel('Model')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(plot_df[metric].sort_values(ascending=False)):
            ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        
        return df
    
    def compare_time_series_models(self, metric='rmse'):
        """
        Compare time series models based on a specific metric
        
        Args:
            metric (str): Metric to use for comparison
            
        Returns:
            pd.DataFrame: Comparison of models
        """
        if not self.time_series_results:
            logger.warning("No time series models to compare")
            return None
        
        results = {}
        for model_name, model_results in self.time_series_results.items():
            results[model_name] = model_results['metrics']
            
        df = pd.DataFrame(results).T
        
        # Create comparison plot
        plt.figure(figsize=(12, 6))
        ax = df[metric].sort_values().plot(kind='bar')
        plt.title(f'Time Series Models Comparison - {metric.upper()}')
        plt.ylabel(metric.upper())
        plt.xlabel('Model')
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(df[metric].sort_values()):
            ax.text(i, v + (v * 0.01), f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        
        return df
    
    def plot_regression_predictions(self, model_names=None, time_index=None, max_points=1000):
        """
        Plot regression model predictions vs. actual values
        
        Args:
            model_names (list, optional): Names of models to plot
            time_index (array-like, optional): Time index for x-axis
            max_points (int): Maximum number of points to plot
            
        Returns:
            matplotlib.figure.Figure: Predictions plot
        """
        if not self.regression_results:
            logger.warning("No regression results to plot")
            return None
        
        # Default to all models if not specified
        if model_names is None:
            model_names = list(self.regression_results.keys())
        
        plt.figure(figsize=(15, 8))
        
        # Get first model's true values
        first_model = model_names[0]
        y_true = self.regression_results[first_model]['y_true']
        
        # Sample points if too many
        if len(y_true) > max_points:
            logger.info(f"Sampling {max_points} points for plotting")
            indices = np.linspace(0, len(y_true) - 1, max_points, dtype=int)
            y_true = y_true[indices]
            time_index = time_index[indices] if time_index is not None else None
        
        # Plot actual values
        x_values = time_index if time_index is not None else np.arange(len(y_true))
        plt.plot(x_values, y_true, 'k-', linewidth=2, label='Actual')
        
        # Plot each model's predictions
        for model_name in model_names:
            if model_name in self.regression_results:
                y_pred = self.regression_results[model_name]['y_pred']
                
                # Sample points if needed
                if len(y_pred) > max_points:
                    y_pred = y_pred[indices]
                
                plt.plot(x_values, y_pred, '--', linewidth=1.5, label=f'{model_name} Predictions')
        
        plt.title('Regression Models: Actual vs. Predicted')
        plt.xlabel('Time' if time_index is not None else 'Sample Index')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_confusion_matrices(self, model_names=None, normalize=False):
        """
        Plot confusion matrices for classification models
        
        Args:
            model_names (list, optional): Names of models to plot
            normalize (bool): Whether to normalize confusion matrices
            
        Returns:
            matplotlib.figure.Figure: Confusion matrices plot
        """
        if not self.classification_results:
            logger.warning("No classification results to plot")
            return None
        
        # Default to all models if not specified
        if model_names is None:
            model_names = list(self.classification_results.keys())
        
        n_models = len(model_names)
        n_cols = min(3, n_models)
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        if n_models == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for i, model_name in enumerate(model_names):
            if i < len(axes) and model_name in self.classification_results:
                ax = axes[i]
                
                cm = self.classification_results[model_name]['metrics']['confusion_matrix']
                
                if normalize:
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    fmt = '.2f'
                else:
                    fmt = 'd'
                
                sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', ax=ax)
                ax.set_title(f'{model_name} Confusion Matrix')
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(n_models, len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        return fig
    
    def plot_roc_curves(self, model_names=None):
        """
        Plot ROC curves for classification models
        
        Args:
            model_names (list, optional): Names of models to plot
            
        Returns:
            matplotlib.figure.Figure: ROC curves plot
        """
        if not self.classification_results:
            logger.warning("No classification results to plot")
            return None
        
        # Get models with probability predictions
        valid_models = []
        for model_name, results in self.classification_results.items():
            if results['y_proba'] is not None and results['y_proba'].ndim > 1:
                valid_models.append(model_name)
        
        if not valid_models:
            logger.warning("No models with probability predictions")
            return None
        
        # Default to all valid models if not specified
        if model_names is None:
            model_names = valid_models
        else:
            model_names = [m for m in model_names if m in valid_models]
        
        plt.figure(figsize=(10, 8))
        
        for model_name in model_names:
            y_true = self.classification_results[model_name]['y_true']
            y_proba = self.classification_results[model_name]['y_proba'][:, 1]
            
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curves')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_precision_recall_curves(self, model_names=None):
        """
        Plot precision-recall curves for classification models
        
        Args:
            model_names (list, optional): Names of models to plot
            
        Returns:
            matplotlib.figure.Figure: Precision-recall curves plot
        """
        if not self.classification_results:
            logger.warning("No classification results to plot")
            return None
        
        # Get models with probability predictions
        valid_models = []
        for model_name, results in self.classification_results.items():
            if results['y_proba'] is not None and results['y_proba'].ndim > 1:
                valid_models.append(model_name)
        
        if not valid_models:
            logger.warning("No models with probability predictions")
            return None
        
        # Default to all valid models if not specified
        if model_names is None:
            model_names = valid_models
        else:
            model_names = [m for m in model_names if m in valid_models]
        
        plt.figure(figsize=(10, 8))
        
        for model_name in model_names:
            y_true = self.classification_results[model_name]['y_true']
            y_proba = self.classification_results[model_name]['y_proba'][:, 1]
            
            precision, recall, _ = precision_recall_curve(y_true, y_proba)
            pr_auc = auc(recall, precision)
            
            plt.plot(recall, precision, lw=2, label=f'{model_name} (AUC = {pr_auc:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend(loc="best")
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def compare_models_table(self, model_type='regression', metrics=None):
        """
        Create a formatted comparison table of model metrics
        
        Args:
            model_type (str): Type of models to compare ('regression', 'classification', 'time_series')
            metrics (list, optional): Metrics to include in the table
            
        Returns:
            pd.DataFrame: Formatted comparison table
        """
        if model_type == 'regression':
            results_dict = self.regression_results
            default_metrics = ['rmse', 'mae', 'r2', 'mape']
        elif model_type == 'classification':
            results_dict = self.classification_results
            default_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        elif model_type == 'time_series':
            results_dict = self.time_series_results
            default_metrics = ['rmse', 'mae', 'mape']
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return None
        
        if not results_dict:
            logger.warning(f"No {model_type} models to compare")
            return None
        
        # Use default metrics if not specified
        if metrics is None:
            metrics = default_metrics
        
        # Extract metrics for each model
        table_data = {}
        for model_name, model_results in results_dict.items():
            model_metrics = {}
            for metric in metrics:
                if metric in model_results['metrics']:
                    model_metrics[metric] = model_results['metrics'][metric]
                else:
                    model_metrics[metric] = np.nan
            table_data[model_name] = model_metrics
        
        df = pd.DataFrame(table_data).T
        
        # Format the table
        formatted_df = df.copy()
        for col in formatted_df.columns:
            if col in ['rmse', 'mae', 'mse']:
                formatted_df[col] = formatted_df[col].map('{:.4f}'.format)
            elif col in ['r2', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
                formatted_df[col] = formatted_df[col].map('{:.4f}'.format)
            elif col == 'mape':
                formatted_df[col] = formatted_df[col].map('{:.2f}%'.format)
        
        return formatted_df
    
    def get_best_model(self, model_type='regression', metric=None, higher_is_better=False):
        """
        Get the best performing model based on a specific metric
        
        Args:
            model_type (str): Type of models to compare ('regression', 'classification', 'time_series')
            metric (str, optional): Metric to use for comparison
            higher_is_better (bool): Whether higher values of the metric are better
            
        Returns:
            tuple: (best_model_name, best_score)
        """
        if model_type == 'regression':
            results_dict = self.regression_results
            default_metric = 'rmse'
            higher_is_better = False if metric is None else higher_is_better
        elif model_type == 'classification':
            results_dict = self.classification_results
            default_metric = 'f1'
            higher_is_better = True if metric is None else higher_is_better
        elif model_type == 'time_series':
            results_dict = self.time_series_results
            default_metric = 'rmse'
            higher_is_better = False if metric is None else higher_is_better
        else:
            logger.warning(f"Unknown model type: {model_type}")
            return None, None
        
        if not results_dict:
            logger.warning(f"No {model_type} models to compare")
            return None, None
        
        # Use default metric if not specified
        if metric is None:
            metric = default_metric
        
        # Extract metric for each model
        scores = {}
        for model_name, model_results in results_dict.items():
            if metric in model_results['metrics']:
                scores[model_name] = model_results['metrics'][metric]
        
        if not scores:
            logger.warning(f"No models have metric: {metric}")
            return None, None
        
        # Find best model
        if higher_is_better:
            best_model = max(scores.items(), key=lambda x: x[1])
        else:
            best_model = min(scores.items(), key=lambda x: x[1])
        
        logger.info(f"Best {model_type} model based on {metric}: {best_model[0]} with score {best_model[1]:.4f}")
        
        return best_model 