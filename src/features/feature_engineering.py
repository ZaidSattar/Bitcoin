import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Class for creating features from Bitcoin price data
    """
    
    def __init__(self, df=None):
        """
        Initialize FeatureEngineering
        
        Args:
            df (pd.DataFrame, optional): DataFrame containing price data
        """
        self.df = df
        
    def set_data(self, df):
        """
        Set data to use for feature engineering
        
        Args:
            df (pd.DataFrame): DataFrame containing price data
        """
        self.df = df
        
    def create_technical_indicators(self):
        """
        Create technical indicators for price data
        
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        if self.df is None:
            raise ValueError("Data must be set before creating features")
            
        logger.info("Creating technical indicators")
        df = self.df.copy()
        
        # Make sure data is sorted by date
        df = df.sort_index()
        
        # Moving Averages
        for window in [7, 14, 30, 50, 200]:
            df[f'SMA_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'EMA_{window}'] = df['Close'].ewm(span=window, adjust=False).mean()
        
        # Price distance from moving averages (normalized)
        for window in [7, 14, 30, 50, 200]:
            df[f'SMA_{window}_dist'] = (df['Close'] - df[f'SMA_{window}']) / df[f'SMA_{window}']
            df[f'EMA_{window}_dist'] = (df['Close'] - df[f'EMA_{window}']) / df[f'EMA_{window}']
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # Moving Average Convergence Divergence (MACD)
        ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = ema_12 - ema_26
        df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_hist'] = df['MACD'] - df['MACD_signal']
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['BB_upper'] = sma_20 + (std_20 * 2)
        df['BB_middle'] = sma_20
        df['BB_lower'] = sma_20 - (std_20 * 2)
        df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
        df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
        
        # Average True Range (ATR) - Volatility
        high_low = df['High'] - df['Low']
        high_close = (df['High'] - df['Close'].shift()).abs()
        low_close = (df['Low'] - df['Close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['ATR_14'] = true_range.rolling(window=14).mean()
        
        # Simple stochastic oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Williams %R
        df['Williams_R'] = -100 * ((high_14 - df['Close']) / (high_14 - low_14))
        
        # On-Balance Volume (OBV)
        obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        df['OBV'] = obv
        
        logger.info(f"Technical indicators created. New shape: {df.shape}")
        
        return df
        
    def create_lagged_features(self, n_lags=5):
        """
        Create lagged features for price and returns
        
        Args:
            n_lags (int): Number of lags to create
            
        Returns:
            pd.DataFrame: DataFrame with lagged features
        """
        if self.df is None:
            raise ValueError("Data must be set before creating features")
            
        logger.info(f"Creating lagged features with {n_lags} lags")
        df = self.df.copy()
        
        # Calculate returns if they don't exist
        if 'returns' not in df.columns:
            df['returns'] = df['Close'].pct_change()
        
        # Create lagged price features
        for i in range(1, n_lags + 1):
            df[f'Close_lag_{i}'] = df['Close'].shift(i)
            df[f'returns_lag_{i}'] = df['returns'].shift(i)
            
            # Price momentum (percent change over different periods)
            df[f'momentum_{i}'] = df['Close'].pct_change(i)
        
        # Create rolling window features
        for window in [7, 14, 30]:
            df[f'returns_mean_{window}'] = df['returns'].rolling(window=window).mean()
            df[f'returns_std_{window}'] = df['returns'].rolling(window=window).std()
            df[f'returns_min_{window}'] = df['returns'].rolling(window=window).min()
            df[f'returns_max_{window}'] = df['returns'].rolling(window=window).max()
            
            # Add volatility features
            df[f'volatility_{window}'] = df['returns'].rolling(window=window).std()
            
        logger.info(f"Lagged features created. New shape: {df.shape}")
        
        return df
        
    def create_date_features(self):
        """
        Create date-based features (day of week, month, etc.)
        
        Returns:
            pd.DataFrame: DataFrame with date features
        """
        if self.df is None:
            raise ValueError("Data must be set before creating features")
            
        logger.info("Creating date features")
        df = self.df.copy()
        
        # Reset index to get datetime as column
        df = df.reset_index()
        
        # Create date features
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['day_of_month'] = df['datetime'].dt.day
        df['week_of_year'] = df['datetime'].dt.isocalendar().week
        df['month'] = df['datetime'].dt.month
        df['quarter'] = df['datetime'].dt.quarter
        df['year'] = df['datetime'].dt.year
        
        # Create cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Set datetime as index again
        df = df.set_index('datetime')
        
        logger.info(f"Date features created. New shape: {df.shape}")
        
        return df
        
    def create_target_variables(self, forecast_horizon=1):
        """
        Create target variables for different prediction tasks
        
        Args:
            forecast_horizon (int): Forecast horizon in periods
            
        Returns:
            pd.DataFrame: DataFrame with target variables
        """
        if self.df is None:
            raise ValueError("Data must be set before creating features")
            
        logger.info(f"Creating target variables with forecast horizon {forecast_horizon}")
        df = self.df.copy()
        
        # Regression targets
        df[f'target_price_{forecast_horizon}'] = df['Close'].shift(-forecast_horizon)
        df[f'target_return_{forecast_horizon}'] = df['Close'].pct_change(-forecast_horizon)
        
        # Classification targets (price direction)
        df[f'target_direction_{forecast_horizon}'] = (df[f'target_return_{forecast_horizon}'] > 0).astype(int)
        
        # Create volatility targets
        for window in [7, 14, 30]:
            df[f'future_volatility_{window}'] = df['returns'].rolling(window=window).std().shift(-window)
        
        logger.info(f"Target variables created. New shape: {df.shape}")
        
        return df
    
    def scale_features(self, df, scaler_type='standard', exclude_cols=None):
        """
        Scale numerical features in the dataframe
        
        Args:
            df (pd.DataFrame): DataFrame with features
            scaler_type (str): Type of scaling ('standard' or 'minmax')
            exclude_cols (list): Columns to exclude from scaling
            
        Returns:
            tuple: (scaled_df, scaler)
        """
        if exclude_cols is None:
            exclude_cols = []
        
        # Add target variables to excluded columns
        target_cols = [col for col in df.columns if col.startswith('target_')]
        exclude_cols = exclude_cols + target_cols
        
        # Separate features for scaling
        features_to_scale = df.columns.difference(exclude_cols)
        features_df = df[features_to_scale]
        
        # Choose scaler
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be 'standard' or 'minmax'")
        
        # Fit and transform
        scaled_features = scaler.fit_transform(features_df)
        scaled_features_df = pd.DataFrame(scaled_features, index=df.index, columns=features_to_scale)
        
        # Combine with excluded columns
        result_df = pd.concat([scaled_features_df, df[exclude_cols]], axis=1)
        
        return result_df, scaler
    
    def prepare_features(self, forecast_horizon=1, n_lags=5, include_date_features=True, 
                          scaler_type='standard', exclude_scaling=None):
        """
        Full pipeline to prepare features
        
        Args:
            forecast_horizon (int): Forecast horizon for target
            n_lags (int): Number of lagged features
            include_date_features (bool): Whether to include date features
            scaler_type (str): Type of scaling
            exclude_scaling (list): Features to exclude from scaling
            
        Returns:
            tuple: (complete_df, scaler)
        """
        if self.df is None:
            raise ValueError("Data must be set before creating features")
            
        logger.info("Starting feature preparation pipeline")
        
        # Step 1: Create technical indicators
        df_tech = self.create_technical_indicators()
        
        # Step 2: Create lagged features
        df_lagged = self.create_lagged_features(n_lags=n_lags)
        
        # Merge technical and lagged features
        # Get columns unique to df_lagged that aren't in df_tech
        lagged_only_cols = [col for col in df_lagged.columns if col not in df_tech.columns]
        df_features = pd.concat([df_tech, df_lagged[lagged_only_cols]], axis=1)
        
        # Step 3: Create date features if requested
        if include_date_features:
            df_date = self.create_date_features()
            # Get columns unique to df_date that aren't in df_features
            date_only_cols = [col for col in df_date.columns if col not in df_features.columns]
            if date_only_cols:
                df_features = pd.concat([df_features, df_date[date_only_cols]], axis=1)
        
        # Step 4: Create target variables
        df_with_targets = self.create_target_variables(forecast_horizon=forecast_horizon)
        # Get columns unique to df_with_targets that aren't in df_features
        target_only_cols = [col for col in df_with_targets.columns if col not in df_features.columns]
        if target_only_cols:
            df_features = pd.concat([df_features, df_with_targets[target_only_cols]], axis=1)
        
        # Step 5: Scale features
        df_scaled, scaler = self.scale_features(df_features, scaler_type=scaler_type, exclude_cols=exclude_scaling)
        
        # Remove rows with NaN values
        df_final = df_scaled.dropna()
        
        logger.info(f"Feature preparation complete. Final shape: {df_final.shape}")
        
        return df_final, scaler 