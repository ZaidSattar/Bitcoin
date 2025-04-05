import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import seaborn as sns
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClusteringModel:
    """
    Class for clustering Bitcoin market regimes
    """
    
    def __init__(self, model_type='kmeans', **kwargs):
        """
        Initialize clustering model
        
        Args:
            model_type (str): Type of clustering model
                             ('kmeans', 'hierarchical', 'dbscan')
            **kwargs: Additional arguments for the specific model
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = self._get_model()
        self.labels_ = None
        
    def _get_model(self):
        """
        Get the clustering model based on model_type
        
        Returns:
            sklearn model: Clustering model
        """
        if self.model_type == 'kmeans':
            return KMeans(**self.kwargs)
        elif self.model_type == 'hierarchical':
            return AgglomerativeClustering(**self.kwargs)
        elif self.model_type == 'dbscan':
            return DBSCAN(**self.kwargs)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X):
        """
        Fit the clustering model to the data
        
        Args:
            X (array-like): Features
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Fitting {self.model_type} clustering model")
        self.model.fit(X)
        
        # Store cluster labels
        self.labels_ = self.model.labels_
            
        return self
    
    def predict(self, X):
        """
        Predict cluster for new data
        Note: Some clustering algorithms like DBSCAN don't support predict
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Cluster labels
        """
        if hasattr(self.model, 'predict'):
            logger.info(f"Predicting clusters with {self.model_type}")
            return self.model.predict(X)
        else:
            logger.warning(f"{self.model_type} does not support predict")
            return None
    
    def evaluate(self, X):
        """
        Evaluate the clustering model using silhouette score and other metrics
        
        Args:
            X (array-like): Features
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before evaluation")
            
        logger.info(f"Evaluating {self.model_type} clustering model")
        
        # Skip evaluation if there's only one cluster or if all points are noise (-1 in DBSCAN)
        if len(np.unique(self.labels_)) <= 1 or (len(np.unique(self.labels_)) == 2 and -1 in self.labels_):
            logger.warning("Cannot evaluate clustering with only one cluster or all noise points")
            return {
                'silhouette_score': None,
                'calinski_harabasz_score': None,
                'davies_bouldin_score': None,
                'num_clusters': len(np.unique(self.labels_)),
                'cluster_sizes': pd.Series(self.labels_).value_counts().to_dict()
            }
        
        # Calculate metrics
        try:
            silhouette = silhouette_score(X, self.labels_)
        except:
            silhouette = None
            
        try:
            calinski_harabasz = calinski_harabasz_score(X, self.labels_)
        except:
            calinski_harabasz = None
            
        try:
            davies_bouldin = davies_bouldin_score(X, self.labels_)
        except:
            davies_bouldin = None
        
        logger.info(f"Clustering evaluation: Silhouette={silhouette}, CH={calinski_harabasz}, DB={davies_bouldin}")
        
        return {
            'silhouette_score': silhouette,
            'calinski_harabasz_score': calinski_harabasz,
            'davies_bouldin_score': davies_bouldin,
            'num_clusters': len(np.unique(self.labels_)),
            'cluster_sizes': pd.Series(self.labels_).value_counts().to_dict()
        }
    
    def plot_clusters_2d(self, X, pca_components=2, feature_names=None, title=None):
        """
        Plot clusters in 2D using PCA for dimensionality reduction if needed
        
        Args:
            X (array-like): Features
            pca_components (int): Number of PCA components
            feature_names (list): Names of features
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Clusters plot
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before plotting")
            
        # Apply PCA if data is high-dimensional
        if X.shape[1] > 2:
            logger.info(f"Applying PCA to reduce dimensions from {X.shape[1]} to {pca_components}")
            pca = PCA(n_components=pca_components)
            X_pca = pca.fit_transform(X)
            
            # If we want 2D plot but requested more components
            if pca_components > 2:
                X_plot = X_pca[:, :2]
            else:
                X_plot = X_pca
        else:
            X_plot = X
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Get unique labels and colors
        unique_labels = np.unique(self.labels_)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = self.labels_ == label
            plt.scatter(
                X_plot[mask, 0], X_plot[mask, 1],
                c=[color], label=f'Cluster {label}',
                alpha=0.7, edgecolors='k', s=50
            )
            
        plt.title(title or f'{self.model_type.upper()} Clustering')
        
        if X.shape[1] > 2:
            plt.xlabel('PCA Component 1')
            plt.ylabel('PCA Component 2')
        elif feature_names and len(feature_names) >= 2:
            plt.xlabel(feature_names[0])
            plt.ylabel(feature_names[1])
        else:
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')
            
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_dendrogram(self, X, title=None):
        """
        Plot dendrogram for hierarchical clustering
        
        Args:
            X (array-like): Features
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Dendrogram plot
        """
        if self.model_type != 'hierarchical':
            logger.warning("Dendrogram is only available for hierarchical clustering")
            return None
            
        # Calculate linkage
        Z = linkage(X, method='ward')
        
        plt.figure(figsize=(12, 8))
        dendrogram(Z)
        plt.title(title or 'Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample index')
        plt.ylabel('Distance')
        
        return plt.gcf()
    
    def plot_cluster_summary(self, X, original_data, columns_to_plot, n_cols=3):
        """
        Plot summary statistics for each cluster
        
        Args:
            X (array-like): Features used for clustering
            original_data (pd.DataFrame): Original data with additional columns
            columns_to_plot (list): Columns to include in summary
            n_cols (int): Number of columns in the subplot grid
            
        Returns:
            matplotlib.figure.Figure: Cluster summary plot
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before plotting")
            
        # Add cluster labels to original data
        data_with_clusters = original_data.copy()
        data_with_clusters['cluster'] = self.labels_
        
        # Get unique clusters
        unique_clusters = np.unique(self.labels_)
        n_clusters = len(unique_clusters)
        
        # Calculate n_rows for subplot grid
        n_rows = (len(columns_to_plot) + n_cols - 1) // n_cols
        
        # Create figure
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        axes = axes.flatten()
        
        # Plot each feature
        for i, col in enumerate(columns_to_plot):
            if i < len(axes):
                ax = axes[i]
                
                # Box plot by cluster
                sns.boxplot(x='cluster', y=col, data=data_with_clusters, ax=ax)
                ax.set_title(f'{col} by Cluster')
                ax.set_xlabel('Cluster')
                
        # Hide unused subplots
        for i in range(len(columns_to_plot), len(axes)):
            fig.delaxes(axes[i])
            
        plt.tight_layout()
        
        return fig
    
    def get_cluster_profiles(self, original_data, columns_to_profile=None):
        """
        Get statistical profiles for each cluster
        
        Args:
            original_data (pd.DataFrame): Original data
            columns_to_profile (list): Columns to include in profiles
            
        Returns:
            pd.DataFrame: Cluster profiles
        """
        if self.labels_ is None:
            raise ValueError("Model must be fitted before getting profiles")
            
        # Add cluster labels to original data
        data_with_clusters = original_data.copy()
        data_with_clusters['cluster'] = self.labels_
        
        # Select columns to profile
        if columns_to_profile is None:
            columns_to_profile = original_data.columns
            
        # Calculate profiles
        profiles = data_with_clusters.groupby('cluster')[columns_to_profile].agg([
            'mean', 'median', 'std', 'min', 'max', 'count'
        ])
        
        return profiles


class AnomalyDetector:
    """
    Class for anomaly detection in Bitcoin data
    """
    
    def __init__(self, model_type='isolation_forest', contamination=0.05, **kwargs):
        """
        Initialize anomaly detection model
        
        Args:
            model_type (str): Type of anomaly detection model
                             ('isolation_forest', 'local_outlier_factor', 'one_class_svm', 'zscore')
            contamination (float): Expected proportion of outliers
            **kwargs: Additional arguments for the specific model
        """
        self.model_type = model_type
        self.contamination = contamination
        self.kwargs = kwargs
        self.model = self._get_model() if model_type != 'zscore' else None
        self.anomaly_scores_ = None
        self.anomalies_ = None
        
    def _get_model(self):
        """
        Get the anomaly detection model based on model_type
        
        Returns:
            sklearn model: Anomaly detection model
        """
        if self.model_type == 'isolation_forest':
            return IsolationForest(contamination=self.contamination, **self.kwargs)
        elif self.model_type == 'local_outlier_factor':
            return LocalOutlierFactor(contamination=self.contamination, **self.kwargs)
        elif self.model_type == 'one_class_svm':
            return OneClassSVM(**self.kwargs)
        elif self.model_type == 'zscore':
            return None
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X):
        """
        Fit the anomaly detection model to the data
        
        Args:
            X (array-like): Features
            
        Returns:
            self: Fitted model
        """
        logger.info(f"Fitting {self.model_type} anomaly detection model")
        
        if self.model_type == 'zscore':
            self._fit_zscore(X)
        else:
            if hasattr(self.model, 'fit_predict'):
                self.anomalies_ = self.model.fit_predict(X)
                # Convert to boolean (True for anomalies)
                self.anomalies_ = self.anomalies_ == -1
            else:
                self.model.fit(X)
            
        return self
    
    def _fit_zscore(self, X):
        """
        Perform Z-score anomaly detection
        
        Args:
            X (array-like): Features
        """
        # Calculate Z-scores
        if isinstance(X, pd.DataFrame):
            z_scores = (X - X.mean()) / X.std()
            # For multivariate data, use mean absolute z-score across features
            if z_scores.shape[1] > 1:
                self.anomaly_scores_ = z_scores.abs().mean(axis=1).values
            else:
                self.anomaly_scores_ = z_scores.abs().values.flatten()
        else:
            # For numpy arrays
            z_scores = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
            if z_scores.ndim > 1 and z_scores.shape[1] > 1:
                self.anomaly_scores_ = np.mean(np.abs(z_scores), axis=1)
            else:
                self.anomaly_scores_ = np.abs(z_scores).flatten()
        
        # Detect anomalies based on threshold (typically Z > 3)
        threshold = 3.0
        self.anomalies_ = self.anomaly_scores_ > threshold
    
    def predict(self, X):
        """
        Predict anomalies in new data
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Anomaly labels (True for anomalies)
        """
        logger.info(f"Detecting anomalies with {self.model_type}")
        
        if self.model_type == 'zscore':
            # Calculate Z-scores for new data
            if isinstance(X, pd.DataFrame):
                z_scores = (X - X.mean()) / X.std()
                if z_scores.shape[1] > 1:
                    scores = z_scores.abs().mean(axis=1).values
                else:
                    scores = z_scores.abs().values.flatten()
            else:
                z_scores = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                if z_scores.ndim > 1 and z_scores.shape[1] > 1:
                    scores = np.mean(np.abs(z_scores), axis=1)
                else:
                    scores = np.abs(z_scores).flatten()
            
            threshold = 3.0
            return scores > threshold
        else:
            if hasattr(self.model, 'predict'):
                anomalies = self.model.predict(X)
                # Convert to boolean (True for anomalies)
                return anomalies == -1
            else:
                logger.warning(f"{self.model_type} does not support predict for new data")
                return None
    
    def score_samples(self, X):
        """
        Get anomaly scores for each sample
        
        Args:
            X (array-like): Features
            
        Returns:
            array: Anomaly scores
        """
        if self.model_type == 'zscore':
            # Use z-scores directly
            if isinstance(X, pd.DataFrame):
                z_scores = (X - X.mean()) / X.std()
                if z_scores.shape[1] > 1:
                    return z_scores.abs().mean(axis=1).values
                else:
                    return z_scores.abs().values.flatten()
            else:
                z_scores = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
                if z_scores.ndim > 1 and z_scores.shape[1] > 1:
                    return np.mean(np.abs(z_scores), axis=1)
                else:
                    return np.abs(z_scores).flatten()
        else:
            if hasattr(self.model, 'score_samples'):
                return -self.model.score_samples(X)  # Negative because higher means more anomalous
            elif hasattr(self.model, 'decision_function'):
                return -self.model.decision_function(X)
            else:
                logger.warning(f"{self.model_type} does not support anomaly scoring")
                return None
    
    def plot_anomalies(self, X, time_index=None, feature_idx=0, title=None):
        """
        Plot anomalies in the data
        
        Args:
            X (array-like): Features
            time_index (array-like): Time index for x-axis
            feature_idx (int): Index of feature to plot
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Anomalies plot
        """
        if self.anomalies_ is None:
            raise ValueError("Model must be fitted before plotting anomalies")
            
        # Extract the feature to plot
        if isinstance(X, pd.DataFrame):
            if feature_idx < len(X.columns):
                feature_name = X.columns[feature_idx]
                feature_values = X.iloc[:, feature_idx].values
            else:
                feature_name = "Feature"
                feature_values = X.iloc[:, 0].values
        else:
            feature_name = f"Feature {feature_idx}"
            if X.ndim > 1 and X.shape[1] > feature_idx:
                feature_values = X[:, feature_idx]
            else:
                feature_values = X.flatten()
        
        # Get anomaly points
        anomaly_indices = np.where(self.anomalies_)[0]
        
        # Create x-axis values
        if time_index is not None:
            x_values = time_index
            x_label = "Time"
        else:
            x_values = np.arange(len(feature_values))
            x_label = "Sample Index"
        
        # Plot
        plt.figure(figsize=(15, 6))
        plt.plot(x_values, feature_values, 'b-', label=feature_name)
        plt.scatter(
            x_values[anomaly_indices], 
            feature_values[anomaly_indices], 
            color='red', 
            s=50, 
            marker='o', 
            alpha=0.7,
            label='Anomalies'
        )
        
        plt.title(title or f'{self.model_type.upper()} Anomaly Detection')
        plt.xlabel(x_label)
        plt.ylabel(feature_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add annotation for number of anomalies
        plt.annotate(
            f"Detected {len(anomaly_indices)} anomalies ({len(anomaly_indices)/len(feature_values):.1%})",
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )
        
        return plt.gcf()
    
    def plot_anomaly_scores(self, X, threshold=None, time_index=None, title=None):
        """
        Plot anomaly scores
        
        Args:
            X (array-like): Features
            threshold (float): Anomaly threshold
            time_index (array-like): Time index for x-axis
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Anomaly scores plot
        """
        # Get anomaly scores
        scores = self.score_samples(X)
        
        if scores is None:
            logger.warning("Cannot plot anomaly scores for this model")
            return None
        
        # Create x-axis values
        if time_index is not None:
            x_values = time_index
            x_label = "Time"
        else:
            x_values = np.arange(len(scores))
            x_label = "Sample Index"
        
        # Set threshold if not provided
        if threshold is None:
            if self.model_type == 'zscore':
                threshold = 3.0
            else:
                # Use percentile based on contamination
                threshold = np.percentile(scores, 100 * (1 - self.contamination))
        
        # Plot
        plt.figure(figsize=(15, 6))
        plt.plot(x_values, scores, 'b-', label='Anomaly Score')
        
        if threshold is not None:
            plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.2f})')
            
            # Highlight anomalies
            anomaly_indices = np.where(scores > threshold)[0]
            plt.scatter(
                x_values[anomaly_indices], 
                scores[anomaly_indices], 
                color='red', 
                s=50, 
                marker='o', 
                alpha=0.7,
                label='Anomalies'
            )
        
        plt.title(title or f'{self.model_type.upper()} Anomaly Scores')
        plt.xlabel(x_label)
        plt.ylabel('Anomaly Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf() 