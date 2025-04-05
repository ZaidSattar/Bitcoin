import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataLoader:
    """
    Class for loading and preprocessing Bitcoin price data
    """
    
    def __init__(self, data_path, resample_rule=None):
        """
        Initialize DataLoader
        
        Args:
            data_path (str): Path to the Bitcoin price data CSV
            resample_rule (str, optional): Rule for resampling data (e.g., 'D' for daily)
        """
        self.data_path = Path(data_path)
        self.resample_rule = resample_rule
        self.data = None
        
    def load_data(self):
        """
        Load data from CSV file
        
        Returns:
            pd.DataFrame: Loaded and processed data
        """
        logger.info(f"Loading data from {self.data_path}")
        
        try:
            # Load data
            df = pd.read_csv(self.data_path)
            
            # Convert datetime column to pandas datetime
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Set datetime as index
            df.set_index('datetime', inplace=True)
            
            # Resample if specified
            if self.resample_rule:
                logger.info(f"Resampling data with rule: {self.resample_rule}")
                df = self._resample_data(df)
                
            # Remove rows with missing values
            df = df.dropna()
            
            self.data = df
            logger.info(f"Data loaded successfully with shape: {df.shape}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _resample_data(self, df):
        """
        Resample data based on specified rule
        
        Args:
            df (pd.DataFrame): Input dataframe
            
        Returns:
            pd.DataFrame: Resampled dataframe
        """
        resampled = df.resample(self.resample_rule).agg({
            'Open': 'first',
            'High': 'max',
            'Low': 'min',
            'Close': 'last',
            'Volume': 'sum'
        })
        
        return resampled
    
    def split_data(self, test_size=0.2, val_size=0.1):
        """
        Split data into training, validation, and test sets
        
        Args:
            test_size (float): Proportion of data for testing
            val_size (float): Proportion of data for validation
            
        Returns:
            tuple: (train_data, val_data, test_data)
        """
        if self.data is None:
            self.load_data()
            
        data = self.data.copy()
        
        # Calculate split indices
        n = len(data)
        test_idx = int(n * (1 - test_size))
        val_idx = int(test_idx * (1 - val_size))
        
        # Split data
        train_data = data.iloc[:val_idx]
        val_data = data.iloc[val_idx:test_idx]
        test_data = data.iloc[test_idx:]
        
        logger.info(f"Data split: train={train_data.shape}, val={val_data.shape}, test={test_data.shape}")
        
        return train_data, val_data, test_data
    
    def get_price_returns(self):
        """
        Calculate returns from price data
        
        Returns:
            pd.DataFrame: DataFrame with returns columns
        """
        if self.data is None:
            self.load_data()
            
        data = self.data.copy()
        
        # Calculate returns
        data['returns'] = data['Close'].pct_change()
        data['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        return data.dropna() 