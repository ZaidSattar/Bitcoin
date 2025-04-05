import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.data.data_loader import DataLoader
from src.features.feature_engineering import FeatureEngineering
from src.models.time_series_models import ARIMAModel
from src.models.ml_models import RegressionModel, ClassificationModel, XGBOOST_AVAILABLE
from src.models.clustering_models import ClusteringModel, AnomalyDetector
from src.evaluation.model_evaluation import ModelEvaluator
from src.visualization.visualizer import BitcoinVisualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("bitcoin_prediction.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_output_dirs():
    """Create output directories if they don't exist"""
    os.makedirs('output/figures', exist_ok=True)
    os.makedirs('output/models', exist_ok=True)
    os.makedirs('output/data', exist_ok=True)
    logger.info("Created output directories")

def load_and_prepare_data(data_path, resample_rule='D'):
    """
    Load and prepare the Bitcoin data
    
    Args:
        data_path (str): Path to the data file
        resample_rule (str): Rule for resampling data
        
    Returns:
        tuple: (train_data, val_data, test_data, feature_data)
    """
    logger.info("Starting data loading and preparation")
    
    # Load data
    data_loader = DataLoader(data_path, resample_rule=resample_rule)
    df = data_loader.load_data()
    
    # Add returns
    df_with_returns = data_loader.get_price_returns()
    
    # Create feature engineering instance
    feature_eng = FeatureEngineering(df_with_returns)
    
    # Prepare features with technical indicators and lag features
    logger.info("Generating features")
    feature_data, scaler = feature_eng.prepare_features(
        forecast_horizon=1,
        n_lags=10,
        include_date_features=True,
        scaler_type='standard'
    )
    
    # Split data for training and testing
    train_data, val_data, test_data = data_loader.split_data(test_size=0.2, val_size=0.1)
    
    logger.info(f"Data prepared: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
    
    return train_data, val_data, test_data, feature_data, scaler

def run_time_series_models(train_data, test_data):
    """
    Run time series models (ARIMA)
    
    Args:
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        
    Returns:
        dict: Time series model results
    """
    logger.info("Running time series models")
    evaluator = ModelEvaluator()
    results = {}
    
    # ARIMA model
    logger.info("Training ARIMA model")
    arima_model = ARIMAModel(order=(5, 1, 0))
    arima_model.fit(train_data['Close'])
    
    # Make predictions
    arima_preds = arima_model.predict(steps=len(test_data))
    
    # Evaluate
    arima_eval = arima_model.evaluate(test_data['Close'])
    evaluator.evaluate_time_series_model('ARIMA', test_data['Close'], arima_preds)
    
    results['arima'] = {
        'model': arima_model,
        'predictions': arima_preds,
        'evaluation': arima_eval
    }
    
    # Compare models
    evaluator.compare_time_series_models()
    
    return results

def run_ml_models(feature_data):
    """
    Run machine learning models for regression and classification
    
    Args:
        feature_data (pd.DataFrame): Prepared feature data
        
    Returns:
        tuple: (regression_results, classification_results)
    """
    logger.info("Running machine learning models")
    evaluator = ModelEvaluator()
    
    # Prepare data for regression (predict next day price)
    X = feature_data.drop([col for col in feature_data.columns if col.startswith('target_')], axis=1)
    y_reg = feature_data['target_price_1']
    
    # Prepare data for classification (predict direction)
    y_clf = feature_data['target_direction_1']
    
    # Split data
    train_size = int(0.7 * len(X))
    val_size = int(0.1 * len(X))
    
    X_train, y_reg_train = X[:train_size], y_reg[:train_size]
    X_val, y_reg_val = X[train_size:train_size+val_size], y_reg[train_size:train_size+val_size]
    X_test, y_reg_test = X[train_size+val_size:], y_reg[train_size+val_size:]
    
    y_clf_train = y_clf[:train_size]
    y_clf_val = y_clf[train_size:train_size+val_size]
    y_clf_test = y_clf[train_size+val_size:]
    
    # Regression models
    logger.info("Training regression models")
    regression_results = {}
    
    # Linear Regression
    linear_reg = RegressionModel(model_type='linear')
    linear_reg.fit(X_train, y_reg_train)
    linear_preds = linear_reg.predict(X_test)
    linear_eval = linear_reg.evaluate(X_test, y_reg_test)
    evaluator.evaluate_regression_model('Linear Regression', y_reg_test, linear_preds)
    regression_results['linear'] = {'model': linear_reg, 'predictions': linear_preds, 'evaluation': linear_eval}
    
    # Random Forest
    rf_reg = RegressionModel(model_type='rf', n_estimators=100, random_state=42)
    rf_reg.fit(X_train, y_reg_train)
    rf_preds = rf_reg.predict(X_test)
    rf_eval = rf_reg.evaluate(X_test, y_reg_test)
    evaluator.evaluate_regression_model('Random Forest', y_reg_test, rf_preds)
    regression_results['rf'] = {'model': rf_reg, 'predictions': rf_preds, 'evaluation': rf_eval}
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        try:
            xgb_reg = RegressionModel(model_type='xgb', n_estimators=100, random_state=42)
            xgb_reg.fit(X_train, y_reg_train)
            xgb_preds = xgb_reg.predict(X_test)
            xgb_eval = xgb_reg.evaluate(X_test, y_reg_test)
            evaluator.evaluate_regression_model('XGBoost', y_reg_test, xgb_preds)
            regression_results['xgb'] = {'model': xgb_reg, 'predictions': xgb_preds, 'evaluation': xgb_eval}
        except Exception as e:
            logger.error(f"Error in XGBoost regression: {e}")
    else:
        logger.info("Skipping XGBoost regression as it's not available")
    
    # Compare regression models
    reg_comparison = evaluator.compare_regression_models()
    
    # Classification models
    logger.info("Training classification models")
    classification_results = {}
    
    # Logistic Regression
    log_clf = ClassificationModel(model_type='logistic', random_state=42, max_iter=1000)
    log_clf.fit(X_train, y_clf_train)
    log_preds = log_clf.predict(X_test)
    log_proba = log_clf.predict_proba(X_test)
    log_eval = log_clf.evaluate(X_test, y_clf_test)
    evaluator.evaluate_classification_model('Logistic Regression', y_clf_test, log_preds, log_proba)
    classification_results['logistic'] = {'model': log_clf, 'predictions': log_preds, 'evaluation': log_eval}
    
    # Random Forest
    rf_clf = ClassificationModel(model_type='rf', n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_clf_train)
    rf_preds = rf_clf.predict(X_test)
    rf_proba = rf_clf.predict_proba(X_test)
    rf_eval = rf_clf.evaluate(X_test, y_clf_test)
    evaluator.evaluate_classification_model('Random Forest', y_clf_test, rf_preds, rf_proba)
    classification_results['rf'] = {'model': rf_clf, 'predictions': rf_preds, 'evaluation': rf_eval}
    
    # XGBoost
    if XGBOOST_AVAILABLE:
        try:
            xgb_clf = ClassificationModel(model_type='xgb', n_estimators=100, random_state=42)
            xgb_clf.fit(X_train, y_clf_train)
            xgb_preds = xgb_clf.predict(X_test)
            xgb_proba = xgb_clf.predict_proba(X_test)
            xgb_eval = xgb_clf.evaluate(X_test, y_clf_test)
            evaluator.evaluate_classification_model('XGBoost', y_clf_test, xgb_preds, xgb_proba)
            classification_results['xgb'] = {'model': xgb_clf, 'predictions': xgb_preds, 'evaluation': xgb_eval}
        except Exception as e:
            logger.error(f"Error in XGBoost classification: {e}")
    else:
        logger.info("Skipping XGBoost classification as it's not available")
    
    # Compare classification models
    clf_comparison = evaluator.compare_classification_models()
    
    # Plot ROC curves
    evaluator.plot_roc_curves()
    plt.savefig('output/figures/roc_curves.png')
    
    return regression_results, classification_results

def run_clustering_and_anomaly_detection(feature_data):
    """
    Run clustering and anomaly detection
    
    Args:
        feature_data (pd.DataFrame): Prepared feature data
        
    Returns:
        tuple: (clustering_results, anomaly_results)
    """
    logger.info("Running clustering and anomaly detection")
    evaluator = ModelEvaluator()
    
    # Select features for clustering
    cluster_features = [col for col in feature_data.columns if not col.startswith('target_')]
    X_cluster = feature_data[cluster_features]
    
    # K-Means clustering
    logger.info("Running K-Means clustering")
    kmeans = ClusteringModel(model_type='kmeans', n_clusters=4, random_state=42)
    kmeans.fit(X_cluster)
    kmeans_eval = kmeans.evaluate(X_cluster)
    evaluator.store_clustering_results('K-Means', kmeans.labels_, kmeans_eval, kmeans)
    
    # Hierarchical clustering
    logger.info("Running Hierarchical clustering")
    hierarchical = ClusteringModel(model_type='hierarchical', n_clusters=4)
    hierarchical.fit(X_cluster)
    hierarchical_eval = hierarchical.evaluate(X_cluster)
    evaluator.store_clustering_results('Hierarchical', hierarchical.labels_, hierarchical_eval, hierarchical)
    
    # Anomaly detection
    logger.info("Running anomaly detection")
    # Isolation Forest
    iso_forest = AnomalyDetector(model_type='isolation_forest', contamination=0.05, random_state=42)
    iso_forest.fit(X_cluster)
    iso_forest_scores = iso_forest.score_samples(X_cluster)
    evaluator.store_anomaly_results('Isolation Forest', iso_forest.anomalies_, iso_forest_scores, iso_forest)
    
    # Local Outlier Factor
    lof = AnomalyDetector(model_type='local_outlier_factor', contamination=0.05, n_neighbors=20)
    lof.fit(X_cluster)
    evaluator.store_anomaly_results('LOF', lof.anomalies_, None, lof)
    
    # Z-score
    zscore = AnomalyDetector(model_type='zscore')
    zscore.fit(X_cluster)
    evaluator.store_anomaly_results('Z-Score', zscore.anomalies_, zscore.anomaly_scores_, zscore)
    
    return {'kmeans': kmeans, 'hierarchical': hierarchical}, {'iso_forest': iso_forest, 'lof': lof, 'zscore': zscore}

def create_visualizations(df, train_data, test_data, feature_data, time_series_results, clustering_models, anomaly_models):
    """
    Create visualizations for the results
    
    Args:
        df (pd.DataFrame): Original data
        train_data (pd.DataFrame): Training data
        test_data (pd.DataFrame): Test data
        feature_data (pd.DataFrame): Feature data
        time_series_results (dict): Time series model results
        clustering_models (dict): Clustering model results
        anomaly_models (dict): Anomaly detection model results
    """
    logger.info("Creating visualizations")
    
    viz = BitcoinVisualizer()
    
    # Price history plot
    price_fig = viz.plot_price_history(df)
    price_fig.savefig('output/figures/price_history.png')
    
    # Returns distribution - calculate returns if not in original dataframe
    if 'returns' not in df.columns:
        df['returns'] = df['Close'].pct_change()
    
    returns_fig = viz.plot_return_distribution(df['returns'].dropna())
    returns_fig.savefig('output/figures/returns_distribution.png')
    
    # Volatility plot
    volatility_fig = viz.plot_volatility(df, window=30)
    volatility_fig.savefig('output/figures/volatility.png')
    
    # Correlation heatmap
    base_features = [col for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'returns'] if col in df.columns]
    corr_fig = viz.plot_correlation_heatmap(df[base_features])
    corr_fig.savefig('output/figures/correlation_heatmap.png')
    
    # Technical indicators
    if 'RSI_3' in feature_data.columns and 'MACD' in feature_data.columns:
        indicators = [col for col in ['RSI_3', 'MACD', 'MACD_signal', 'MACD_hist', 'BB_width'] if col in feature_data.columns]
        if indicators:
            tech_fig = viz.plot_technical_indicators(feature_data, indicators)
            tech_fig.savefig('output/figures/technical_indicators.png')
    
    # Time series forecasts
    if 'arima' in time_series_results:
        arima_model = time_series_results['arima']['model']
        arima_preds = time_series_results['arima']['predictions']
        arima_fig = arima_model.plot_forecast(train_data['Close'][-100:], test_data['Close'], arima_preds)
        arima_fig.savefig('output/figures/arima_forecast.png')
    
    # Clustering visualization
    if 'kmeans' in clustering_models:
        kmeans = clustering_models['kmeans']
        # Select two features for visualization
        feature_x, feature_y = 'Close', 'Volume'
        if feature_x in feature_data.columns and feature_y in feature_data.columns:
            cluster_fig = viz.plot_clusters(feature_data, kmeans.labels_, feature_x, feature_y, 'K-Means Clustering')
            cluster_fig.savefig('output/figures/kmeans_clusters.png')
    
    # Anomaly detection visualization
    if 'iso_forest' in anomaly_models:
        iso_forest = anomaly_models['iso_forest']
        anomaly_fig = viz.plot_anomalies(df['Close'], iso_forest.anomalies_, 'Close', df.index, 'Isolation Forest Anomalies')
        anomaly_fig.savefig('output/figures/isolation_forest_anomalies.png')

def recursive_format_numpy(obj):
    """
    Recursively format numpy values in dictionaries and lists
    
    Args:
        obj: Object to format
        
    Returns:
        Formatted object
    """
    if isinstance(obj, dict):
        return {k: recursive_format_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [recursive_format_numpy(v) for v in obj]
    elif hasattr(obj, 'item') and callable(getattr(obj, 'item')):
        return obj.item()
    else:
        return obj
        
def generate_insights(feature_data, time_series_results, regression_results, classification_results, clustering_models, anomaly_models):
    """
    Generate insights from the model results
    
    Args:
        feature_data (pd.DataFrame): Feature data
        time_series_results (dict): Time series model results
        regression_results (dict): Regression model results
        classification_results (dict): Classification model results
        clustering_models (dict): Clustering model results 
        anomaly_models (dict): Anomaly detection model results
        
    Returns:
        dict: Insights from the models
    """
    logger.info("Generating insights")
    insights = {}
    
    # Time series insights
    if time_series_results:
        insights['time_series'] = {}
        
        # Compare time series models
        model_performance = {}
        for model_name, model_results in time_series_results.items():
            model_performance[model_name] = {
                'rmse': model_results['evaluation'].get('rmse', 0),
                'mae': model_results['evaluation'].get('mae', 0),
                'mape': model_results['evaluation'].get('mape', 0)
            }
        
        best_model = min(model_performance.items(), key=lambda x: x[1]['rmse'])
        insights['time_series']['best_model'] = best_model[0]
        insights['time_series']['model_performance'] = model_performance
    
    # Regression insights
    if regression_results:
        insights['regression'] = {}
        
        # Compare regression models
        model_performance = {}
        for model_name, model_results in regression_results.items():
            model_performance[model_name] = {
                'rmse': model_results['evaluation'].get('rmse', 0),
                'mae': model_results['evaluation'].get('mae', 0),
                'r2': model_results['evaluation'].get('r2', 0),
                'mape': model_results['evaluation'].get('mape', 0)
            }
        
        best_model = min(model_performance.items(), key=lambda x: x[1]['rmse'])
        insights['regression']['best_model'] = best_model[0]
        insights['regression']['model_performance'] = model_performance
    
    # Classification insights
    if classification_results:
        insights['classification'] = {}
        
        # Compare classification models
        model_performance = {}
        for model_name, model_results in classification_results.items():
            model_performance[model_name] = {
                'accuracy': model_results['evaluation'].get('accuracy', 0),
                'precision': model_results['evaluation'].get('precision', 0),
                'recall': model_results['evaluation'].get('recall', 0),
                'f1': model_results['evaluation'].get('f1', 0)
            }
        
        best_model = max(model_performance.items(), key=lambda x: x[1]['f1'])
        insights['classification']['best_model'] = best_model[0]
        insights['classification']['model_performance'] = model_performance
    
    # Clustering insights
    if clustering_models:
        insights['clustering'] = {}
        
        # Add cluster labels to feature data
        if 'kmeans' in clustering_models:
            feature_data_with_clusters = feature_data.copy()
            feature_data_with_clusters['cluster'] = clustering_models['kmeans'].labels_
            
            # Analyze cluster characteristics
            # Check if necessary columns exist before calculating
            columns_to_use = ['Close']
            if 'returns' in feature_data_with_clusters.columns:
                columns_to_use.append('returns')
            if 'volatility_14' in feature_data_with_clusters.columns:
                columns_to_use.append('volatility_14')
                
            cluster_profiles = feature_data_with_clusters.groupby('cluster')[columns_to_use].mean()
            
            # Identify dominant clusters
            cluster_sizes = feature_data_with_clusters['cluster'].value_counts().to_dict()
            
            insights['clustering']['cluster_profiles'] = cluster_profiles.to_dict()
            insights['clustering']['cluster_sizes'] = cluster_sizes
    
    # Anomaly detection insights
    if anomaly_models:
        insights['anomaly_detection'] = {}
        
        # Compare anomaly detection models
        anomaly_counts = {}
        for model_name, model_results in anomaly_models.items():
            anomaly_counts[model_name] = sum(model_results.anomalies_)
            
            # Get anomaly dates
            if hasattr(feature_data, 'index'):
                anomaly_dates = feature_data.index[model_results.anomalies_].tolist()
                insights['anomaly_detection'][f'{model_name}_dates'] = [str(date) for date in anomaly_dates]
        
        insights['anomaly_detection']['anomaly_counts'] = anomaly_counts
    
    # Save insights to file
    with open('output/insights.txt', 'w') as f:
        for category, category_insights in insights.items():
            f.write(f"{category.upper()} INSIGHTS\n")
            f.write("=" * 50 + "\n")
            for key, value in category_insights.items():
                # Format numpy values
                formatted_value = recursive_format_numpy(value)
                f.write(f"{key}: {formatted_value}\n")
            f.write("\n")
    
    return insights

def main():
    """Main function to run the entire pipeline"""
    logger.info("Starting Bitcoin price prediction pipeline")
    
    # Create output directories
    create_output_dirs()
    
    # Load and prepare data
    train_data, val_data, test_data, feature_data, scaler = load_and_prepare_data(
        data_path='btcusd_1-min_data.csv',
        resample_rule='D'  # Resample to daily data for better performance
    )
    
    # Save a sample of the processed data
    feature_data.to_csv('output/data/feature_data_sample.csv', index=True)
    
    # Run time series models
    time_series_results = run_time_series_models(train_data, test_data)
    
    # Run ML models
    regression_results, classification_results = run_ml_models(feature_data)
    
    # Run clustering and anomaly detection
    clustering_models, anomaly_models = run_clustering_and_anomaly_detection(feature_data)
    
    # Create visualizations
    create_visualizations(
        pd.concat([train_data, val_data, test_data]), 
        train_data, 
        test_data, 
        feature_data, 
        time_series_results,
        clustering_models,
        anomaly_models
    )
    
    # Generate insights
    insights = generate_insights(
        feature_data,
        time_series_results,
        regression_results,
        classification_results,
        clustering_models,
        anomaly_models
    )
    
    logger.info("Bitcoin price prediction pipeline completed successfully")
    
    # Print insights to console
    print("\n===== INSIGHTS =====")
    for category, category_insights in insights.items():
        print(f"\n{category.upper()} INSIGHTS")
        print("=" * 50)
        for key, value in category_insights.items():
            print(f"{key}: {value}")

if __name__ == "__main__":
    main() 