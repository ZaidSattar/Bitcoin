import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from matplotlib.ticker import FuncFormatter
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BitcoinVisualizer:
    """
    Class for visualizing Bitcoin price data and model results
    """
    
    def __init__(self, figsize=(12, 8), style='seaborn-v0_8-darkgrid'):
        """
        Initialize BitcoinVisualizer
        
        Args:
            figsize (tuple): Default figure size
            style (str): Matplotlib style
        """
        self.figsize = figsize
        self.style = style
        plt.style.use(self.style)
        
    def plot_price_history(self, data, column='Close', 
                          start_date=None, end_date=None, title=None):
        """
        Plot Bitcoin price history
        
        Args:
            data (pd.DataFrame): DataFrame with price data
            column (str): Column to plot
            start_date (str): Start date for plot
            end_date (str): End date for plot
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Price history plot
        """
        # Filter data by date if specified
        plot_data = data.copy()
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
            
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(plot_data.index, plot_data[column], linewidth=1.5)
        
        # Format x-axis for dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Format y-axis for currency
        def currency_formatter(x, pos):
            if x >= 1000:
                return f'${x/1000:.1f}K'
            else:
                return f'${x:.2f}'
                
        ax.yaxis.set_major_formatter(FuncFormatter(currency_formatter))
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel(f'Bitcoin Price ({column})')
        ax.set_title(title or f'Bitcoin {column} Price History')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
        
    def plot_candlestick(self, data, start_date=None, end_date=None, title=None):
        """
        Create a candlestick chart of Bitcoin prices
        
        Args:
            data (pd.DataFrame): DataFrame with OHLC data
            start_date (str): Start date for plot
            end_date (str): End date for plot
            title (str): Plot title
            
        Returns:
            plotly.graph_objects.Figure: Candlestick chart
        """
        # Filter data by date if specified
        plot_data = data.copy()
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
            
        # Use limited number of points to avoid overloading the plot
        if len(plot_data) > 500:
            logger.info(f"Resampling data from {len(plot_data)} to 500 points")
            # Resample to daily data
            plot_data = plot_data.resample('D').agg({
                'Open': 'first',
                'High': 'max',
                'Low': 'min',
                'Close': 'last',
                'Volume': 'sum'
            }).dropna()
            
        # Create candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=plot_data.index,
            open=plot_data['Open'],
            high=plot_data['High'],
            low=plot_data['Low'],
            close=plot_data['Close'],
            name='OHLC'
        )])
        
        # Add volume as bar chart on secondary y-axis
        if 'Volume' in plot_data.columns:
            fig.add_trace(go.Bar(
                x=plot_data.index,
                y=plot_data['Volume'],
                name='Volume',
                yaxis='y2',
                marker_color='rgba(0, 0, 255, 0.3)'
            ))
            
            # Update layout for dual y-axis
            fig.update_layout(
                yaxis2=dict(
                    title='Volume',
                    overlaying='y',
                    side='right'
                )
            )
        
        # Update layout
        fig.update_layout(
            title=title or 'Bitcoin Price Candlestick Chart',
            xaxis_title='Date',
            yaxis_title='Price (USD)',
            xaxis_rangeslider_visible=False,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        
        return fig
    
    def plot_technical_indicators(self, data, indicators, start_date=None, end_date=None):
        """
        Plot Bitcoin price with technical indicators
        
        Args:
            data (pd.DataFrame): DataFrame with price and indicator data
            indicators (list): List of indicator columns to plot
            start_date (str): Start date for plot
            end_date (str): End date for plot
            
        Returns:
            matplotlib.figure.Figure: Technical indicators plot
        """
        # Filter data by date if specified
        plot_data = data.copy()
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
            
        # Create plot with price in the top panel and indicators below
        n_indicators = len(indicators)
        fig, axs = plt.subplots(n_indicators + 1, 1, figsize=(self.figsize[0], self.figsize[1] * (n_indicators + 1) // 2), 
                               sharex=True, gridspec_kw={'height_ratios': [3] + [1] * n_indicators})
        
        # Plot price in the top panel
        axs[0].plot(plot_data.index, plot_data['Close'], label='Close Price')
        
        # Add moving averages to price panel if they're in the indicators list
        sma_indicators = [ind for ind in indicators if 'SMA_' in ind]
        ema_indicators = [ind for ind in indicators if 'EMA_' in ind]
        
        for sma in sma_indicators:
            axs[0].plot(plot_data.index, plot_data[sma], label=sma, linestyle='--')
            indicators.remove(sma)
            
        for ema in ema_indicators:
            axs[0].plot(plot_data.index, plot_data[ema], label=ema, linestyle='-.')
            indicators.remove(ema)
            
        axs[0].set_ylabel('Price')
        axs[0].legend(loc='upper left')
        axs[0].set_title('Bitcoin Price with Technical Indicators')
        
        # Plot remaining indicators in separate panels
        for i, indicator in enumerate(indicators, 1):
            if i <= n_indicators:
                axs[i].plot(plot_data.index, plot_data[indicator], label=indicator)
                
                # Add reference lines for oscillators
                if 'RSI' in indicator:
                    axs[i].axhline(y=70, color='r', linestyle='--', alpha=0.3)
                    axs[i].axhline(y=30, color='g', linestyle='--', alpha=0.3)
                elif 'MACD' in indicator and 'hist' not in indicator:
                    axs[i].axhline(y=0, color='r', linestyle='--', alpha=0.3)
                    
                axs[i].set_ylabel(indicator)
                axs[i].legend(loc='upper left')
        
        # Format x-axis for dates
        axs[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        return fig
    
    def plot_return_distribution(self, returns, bins=100, title=None):
        """
        Plot the distribution of Bitcoin returns
        
        Args:
            returns (pd.Series): Series of returns
            bins (int): Number of bins for histogram
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Returns distribution plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot histogram with kernel density estimate
        sns.histplot(returns, bins=bins, kde=True, ax=ax)
        
        # Add vertical line at 0
        ax.axvline(x=0, color='r', linestyle='--')
        
        # Add mean and median lines
        mean_return = returns.mean()
        median_return = returns.median()
        
        ax.axvline(x=mean_return, color='g', linestyle='-', label=f'Mean: {mean_return:.4f}')
        ax.axvline(x=median_return, color='b', linestyle='-.', label=f'Median: {median_return:.4f}')
        
        # Add text with statistics
        stats_text = (
            f"Mean: {mean_return:.4f}\n"
            f"Median: {median_return:.4f}\n"
            f"Std Dev: {returns.std():.4f}\n"
            f"Skewness: {returns.skew():.4f}\n"
            f"Kurtosis: {returns.kurtosis():.4f}"
        )
        
        ax.text(0.03, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Set labels and title
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        ax.set_title(title or 'Bitcoin Returns Distribution')
        
        plt.legend()
        plt.tight_layout()
        
        return fig
    
    def plot_volatility(self, data, window=30, column='returns', start_date=None, end_date=None):
        """
        Plot Bitcoin price volatility over time
        
        Args:
            data (pd.DataFrame): DataFrame with returns data
            window (int): Rolling window size for volatility calculation
            column (str): Column to calculate volatility from
            start_date (str): Start date for plot
            end_date (str): End date for plot
            
        Returns:
            matplotlib.figure.Figure: Volatility plot
        """
        # Filter data by date if specified
        plot_data = data.copy()
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
            
        # Calculate rolling volatility
        volatility = plot_data[column].rolling(window=window).std() * np.sqrt(window)
        
        # Create plot
        fig, ax = plt.subplots(figsize=self.figsize)
        
        ax.plot(volatility.index, volatility, linewidth=1.5)
        
        # Format x-axis for dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Set labels and title
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.set_title(f'Bitcoin {window}-Day Rolling Volatility')
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    
    def plot_correlation_heatmap(self, data, columns=None, title=None):
        """
        Plot correlation heatmap for features
        
        Args:
            data (pd.DataFrame): DataFrame with features
            columns (list): List of columns to include
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Correlation heatmap
        """
        # Select columns if specified
        if columns:
            plot_data = data[columns].copy()
        else:
            plot_data = data.copy()
            
        # Calculate correlation matrix
        corr_matrix = plot_data.corr()
        
        # Create heatmap
        plt.figure(figsize=self.figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        heatmap = sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            linewidths=0.5,
            cbar_kws={'shrink': 0.8}
        )
        
        plt.title(title or 'Feature Correlation Heatmap')
        plt.tight_layout()
        
        return plt.gcf()
    
    def plot_clusters(self, data, labels, feature_x, feature_y, title=None):
        """
        Plot clusters in 2D
        
        Args:
            data (pd.DataFrame): DataFrame with features
            labels (array-like): Cluster labels
            feature_x (str): Feature for x-axis
            feature_y (str): Feature for y-axis
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Clusters plot
        """
        plt.figure(figsize=self.figsize)
        
        # Get unique labels
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        # Plot each cluster
        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            mask = labels == label
            plt.scatter(
                data[feature_x].values[mask], 
                data[feature_y].values[mask],
                c=[color], 
                label=f'Cluster {label}',
                alpha=0.7, 
                edgecolors='k', 
                s=80
            )
            
        plt.title(title or f'Clusters: {feature_x} vs {feature_y}')
        plt.xlabel(feature_x)
        plt.ylabel(feature_y)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def plot_anomalies(self, data, anomalies, feature='Close', time_index=None, title=None):
        """
        Plot anomalies in time series data
        
        Args:
            data (pd.DataFrame or array-like): Data with feature
            anomalies (array-like): Boolean array indicating anomalies
            feature (str): Feature to plot
            time_index (array-like): Time index for x-axis
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Anomalies plot
        """
        plt.figure(figsize=self.figsize)
        
        # Extract the feature to plot
        if isinstance(data, pd.DataFrame):
            feature_values = data[feature].values
            if time_index is None:
                time_index = data.index
        else:
            feature_values = data
            if time_index is None:
                time_index = np.arange(len(feature_values))
        
        # Find anomaly indices
        anomaly_indices = np.where(anomalies)[0]
        
        # Plot normal data
        plt.plot(time_index, feature_values, 'b-', label=feature)
        
        # Plot anomalies
        plt.scatter(
            time_index[anomaly_indices],
            feature_values[anomaly_indices],
            color='red',
            s=80,
            marker='o',
            alpha=0.7,
            label='Anomalies'
        )
        
        plt.title(title or f'Anomalies in {feature}')
        plt.xlabel('Time')
        plt.ylabel(feature)
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
    
    def plot_forecast(self, actual, predicted, prediction_dates=None, 
                     train_actual=None, train_dates=None, title=None):
        """
        Plot actual vs predicted values with forecast
        
        Args:
            actual (array-like): Actual test values
            predicted (array-like): Predicted values
            prediction_dates (array-like): Dates for predictions
            train_actual (array-like): Actual training values
            train_dates (array-like): Dates for training data
            title (str): Plot title
            
        Returns:
            matplotlib.figure.Figure: Forecast plot
        """
        plt.figure(figsize=self.figsize)
        
        # Plot training data if provided
        if train_actual is not None:
            if train_dates is None:
                train_dates = np.arange(len(train_actual))
            plt.plot(train_dates, train_actual, 'b-', label='Training Data')
        
        # Plot test data and predictions
        if prediction_dates is None:
            prediction_dates = np.arange(len(actual))
            if train_actual is not None:
                prediction_dates += len(train_actual)
        
        plt.plot(prediction_dates, actual, 'g-', label='Actual')
        plt.plot(prediction_dates, predicted, 'r--', label='Predicted')
        
        # Calculate error metrics
        mse = np.mean((actual - predicted) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(actual - predicted))
        
        # Add error metrics to plot
        plt.annotate(
            f"RMSE: {rmse:.4f}\nMAE: {mae:.4f}",
            xy=(0.02, 0.95),
            xycoords='axes fraction',
            fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
        )
        
        plt.title(title or 'Actual vs Predicted Values')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def create_dashboard(self, data, start_date=None, end_date=None):
        """
        Create interactive dashboard with Plotly
        
        Args:
            data (pd.DataFrame): DataFrame with price and indicator data
            start_date (str): Start date for dashboard
            end_date (str): End date for dashboard
            
        Returns:
            plotly.graph_objects.Figure: Dashboard figure
        """
        # Filter data by date if specified
        plot_data = data.copy()
        if start_date:
            plot_data = plot_data[plot_data.index >= start_date]
        if end_date:
            plot_data = plot_data[plot_data.index <= end_date]
            
        # Create subplots
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price', 'Volume', 'RSI', 'MACD'),
            row_heights=[0.5, 0.15, 0.15, 0.2]
        )
        
        # Add price candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=plot_data.index,
                open=plot_data['Open'],
                high=plot_data['High'],
                low=plot_data['Low'],
                close=plot_data['Close'],
                name='OHLC'
            ),
            row=1, col=1
        )
        
        # Add SMA lines if available
        if 'SMA_20' in plot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data['SMA_20'],
                    name='SMA 20',
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
            
        if 'SMA_50' in plot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index, 
                    y=plot_data['SMA_50'],
                    name='SMA 50',
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
            
        # Add volume chart
        fig.add_trace(
            go.Bar(
                x=plot_data.index,
                y=plot_data['Volume'],
                name='Volume',
                marker_color='rgba(0, 0, 255, 0.3)'
            ),
            row=2, col=1
        )
        
        # Add RSI chart if available
        if 'RSI_14' in plot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['RSI_14'],
                    name='RSI (14)',
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )
            
            # Add RSI reference lines
            fig.add_trace(
                go.Scatter(
                    x=[plot_data.index.min(), plot_data.index.max()],
                    y=[70, 70],
                    name='Overbought',
                    line=dict(color='red', width=1, dash='dash'),
                    showlegend=False
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=[plot_data.index.min(), plot_data.index.max()],
                    y=[30, 30],
                    name='Oversold',
                    line=dict(color='green', width=1, dash='dash'),
                    showlegend=False
                ),
                row=3, col=1
            )
        
        # Add MACD chart if available
        if all(col in plot_data.columns for col in ['MACD', 'MACD_signal']):
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['MACD'],
                    name='MACD',
                    line=dict(color='blue', width=1)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=plot_data.index,
                    y=plot_data['MACD_signal'],
                    name='Signal',
                    line=dict(color='red', width=1)
                ),
                row=4, col=1
            )
            
            # Add MACD histogram
            if 'MACD_hist' in plot_data.columns:
                fig.add_trace(
                    go.Bar(
                        x=plot_data.index,
                        y=plot_data['MACD_hist'],
                        name='Histogram',
                        marker_color='rgba(0, 150, 0, 0.5)'
                    ),
                    row=4, col=1
                )
                
            # Add zero line
            fig.add_trace(
                go.Scatter(
                    x=[plot_data.index.min(), plot_data.index.max()],
                    y=[0, 0],
                    name='Zero Line',
                    line=dict(color='black', width=1, dash='dash'),
                    showlegend=False
                ),
                row=4, col=1
            )
        
        # Update layout
        fig.update_layout(
            title='Bitcoin Price Analysis Dashboard',
            xaxis_rangeslider_visible=False,
            height=900,
            width=1200,
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        
        # Update y-axis titles
        fig.update_yaxes(title_text='Price (USD)', row=1, col=1)
        fig.update_yaxes(title_text='Volume', row=2, col=1)
        fig.update_yaxes(title_text='RSI', row=3, col=1)
        fig.update_yaxes(title_text='MACD', row=4, col=1)
        
        return fig