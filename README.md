# Bitcoin Price Prediction System

A comprehensive machine learning system for Bitcoin price forecasting, direction prediction, market regime segmentation, and anomaly detection.

## Overview

This project implements a complete machine learning pipeline for analyzing Bitcoin price data and making predictions using various models:

- **Time Series Forecasting**: ARIMA and LSTM models for price prediction
- **Regression Models**: Linear Regression, Random Forest, and XGBoost to predict price levels
- **Classification Models**: Logistic Regression, Random Forest, and XGBoost to predict price direction
- **Clustering**: K-Means and Hierarchical clustering for market regime segmentation
- **Anomaly Detection**: Isolation Forest, Local Outlier Factor, and Z-score methods to identify unusual market events

## Features

- Data loading and preprocessing from CSV files
- Feature engineering with technical indicators (moving averages, RSI, MACD, etc.)
- Time series analysis and forecasting
- Machine learning model training and evaluation
- Market regime clustering
- Anomaly detection
- Visualization of results
- Insights generation

## Project Structure

```
├── main.py                        # Main script to run the entire pipeline
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
├── src/                           # Source code directory
│   ├── data/                      # Data loading and processing
│   ├── features/                  # Feature engineering
│   ├── models/                    # Machine learning models
│   ├── evaluation/                # Model evaluation
│   └── visualization/             # Data visualization
├── output/                        # Output directory
│   ├── figures/                   # Saved visualizations
│   ├── models/                    # Saved models
│   ├── data/                      # Processed data samples
│   └── insights.txt               # Generated insights
```

## Installation

1. Clone the repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Usage

To run the full pipeline, execute:

```bash
python main.py
```

This will:
1. Load and preprocess the Bitcoin price data
2. Generate features and technical indicators
3. Train and evaluate various models
4. Perform clustering and anomaly detection
5. Create visualizations
6. Generate insights

## Data

The system uses a Bitcoin price dataset in CSV format with the following columns:
- Timestamp
- Open
- High
- Low
- Close
- Volume
- Datetime

Place the file named `btcusd_1-min_data.csv` in the root directory.

## Models

### Time Series Models
- ARIMA (Auto-Regressive Integrated Moving Average)
- LSTM (Long Short-Term Memory neural networks)

### Regression Models
- Linear Regression
- Random Forest Regressor
- XGBoost Regressor

### Classification Models
- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier

### Clustering Models
- K-Means
- Hierarchical (Agglomerative)

### Anomaly Detection Models
- Isolation Forest
- Local Outlier Factor (LOF)
- Z-score based detection

## Outputs

The system generates the following outputs:

- Visualizations saved in `output/figures/`
- Processed data samples in `output/data/`
- Insights in `output/insights.txt`
- Logs in `bitcoin_prediction.log`

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

## License

MIT 