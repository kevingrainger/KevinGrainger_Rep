#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Climate-Stressed Portfolio Optimization: An Econophysics Approach
Integrates CAPM, Markowitz optimization with climate risk modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
from scipy.stats import norm, kurtosis
import requests
from datetime import datetime, timedelta
import networkx as nx

plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = [12, 8]

# ======================
# DATA LOADING MODULE
# ======================

class ClimateDataLoader:
    """Loads financial and climate datasets"""
    
    def __init__(self):
        self.tickers = {
            'clean_energy': ['ICLN', 'PBW', 'QCLN'],
            'traditional_energy': ['XOM', 'CVX', 'COP'],
            'climate_tech': ['TSLA', 'ENPH', 'SEDG', 'FSLR'],
            'utilities': ['NEE', 'D', 'SO'],
            'industrials': ['CAT', 'GE', 'HON']
        }
        self.climate_indicators = {}
        self.returns = None
        self.climate_data = None
        
    def load_financial_data(self, start_date='2018-01-01', end_date='2023-12-31'):
        """Download stock price data from Yahoo Finance"""
        all_tickers = [t for group in self.tickers.values() for t in group]
        data = yf.download(all_tickers, start=start_date, end=end_date)['Adj Close']
        self.returns = np.log(data/data.shift(1)).dropna()
        return self.returns
    
    def load_climate_data(self):
        """Load climate indicators from various sources"""
        # Simulated climate data - in practice would use API calls to NOAA etc.
        dates = pd.date_range(start='2018-01-01', end='2023-12-31', freq='M')
        
        # Temperature anomalies (simulated)
        temp_anomaly = np.cumsum(np.random.normal(0, 0.1, len(dates))) + np.sin(np.linspace(0, 6*np.pi, len(dates)))
        
        # Extreme weather events count (simulated)
        weather_events = np.random.poisson(3 + temp_anomaly*0.5)
        
        # Carbon prices (simulated EU ETS)
        carbon_price = 50 + 30*np.sin(np.linspace(0, 3*np.pi, len(dates))) + np.random.normal(0, 5, len(dates))
        
        self.climate_data = pd.DataFrame({
            'date': dates,
            'temp_anomaly': temp_anomaly,
            'weather_events': weather_events,
            'carbon_price': carbon_price
        }).set_index('date')
        
        return self.climate_data
    
    def get_combined_dataset(self):
        """Merge financial returns with climate data"""
        if self.returns is None:
            self.load_financial_data()
        if self.climate_data is None:
            self.load_climate_data()
            
        monthly_returns = self.returns.resample('M').mean()
        combined = monthly_returns.join(self.climate_data, how='left')
        return combined.dropna()

# ======================
# PORTFOLIO OPTIMIZATION 
# ======================

class PortfolioOptimizer:
    """Traditional and climate-stressed portfolio optimization"""
    
    def __init__(self, returns):
        self.returns = returns
        self.cov_matrix = returns.cov()
        self.mean_returns = returns.mean()
        self.num_assets = len(returns.columns)
        
    def traditional_frontier(self):
        """Calculate Markowitz efficient frontier"""
        num_portfolios = 10000
        results = np.zeros((3, num_portfolios))
        
        for i in range(num_portfolios):
            weights = np.random.random(self.num_assets)
            weights /= np.sum(weights)
            
            portfolio_return = np.sum(self.mean_returns * weights)
            portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
            
            results[0,i] = portfolio_return
            results[1,i] = portfolio_vol
            results[2,i] = portfolio_return / portfolio_vol
            
        return results
    
    def optimize_portfolio(self, target_return=None):
        """Optimize portfolio weights for given target return"""
        def portfolio_volatility(weights):
            return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0,1) for _ in range(self.num_assets))
        
        if target_return is None:
            # Maximize Sharpe ratio
            def negative_sharpe(weights):
                ret = np.sum(self.mean_returns * weights)
                vol = portfolio_volatility(weights)
                return -ret/vol
                
            initial_weights = np.array(self.num_assets * [1./self.num_assets])
            optimized = minimize(negative_sharpe, initial_weights, 
                               method='SLSQP', bounds=bounds, constraints=constraints)
        else:
            # Minimize volatility for target return
            constraints = (
                {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                {'type': 'eq', 'fun': lambda x: np.sum(self.mean_returns * x) - target_return}
            )
            initial_weights = np.array(self.num_assets * [1./self.num_assets])
            optimized = minimize(portfolio_volatility, initial_weights, 
                               method='SLSQP', bounds=bounds, constraints=constraints)
        
        return optimized.x
    
    def climate_stressed_var(self, weights, climate_factor=1.0, confidence=0.95):
        """Calculate climate-adjusted Value at Risk"""
        portfolio_returns = np.dot(self.returns, weights)
        
        # Stress the returns based on climate factor
        stressed_returns = portfolio_returns * (1 - 0.1*climate_factor)
        
        var = np.percentile(stressed_returns, 100*(1-confidence))
        return var

# ======================
# CLIMATE RISK MODELS
# ======================

class ClimateRiskModels:
    """Econophysics models for climate risk integration"""
    
    @staticmethod
    def catastrophe_model(X, Y, params):
        """
        Cusp catastrophe model for regime shifts
        X: Portfolio value/returns
        Y: Climate stress indicator
        params: (a0, a1, b0, b1, sigma)
        """
        a0, a1, b0, b1, sigma = params
        a = a0 + a1*Y
        b = b0 + b1*Y
        dX = 4*X**3 - 2*b*X + a + sigma*np.random.normal()
        return dX
    
    @staticmethod
    def sentiment_diffusion(initial_opinions, climate_shock, mu=0.35, steps=1000):
        """
        Deffuant-style opinion diffusion model
        Returns final opinion distribution after climate shock
        """
        N = len(initial_opinions)
        opinions = np.array(initial_opinions)
        
        for _ in range(steps):
            i, j = np.random.choice(N, 2, replace=False)
            if abs(opinions[i] - opinions[j]) < 0.5:  # Interaction threshold
                opinions[i] += mu * (opinions[j] - opinions[i]) * climate_shock
                opinions[j] += mu * (opinions[i] - opinions[j]) * climate_shock
                
        return opinions
    
    @staticmethod
    def market_contagion(corr_matrix, climate_stress, threshold=0.7):
        """
        Network contagion model based on Bornholdt Ising approach
        corr_matrix: Asset correlation matrix
        climate_stress: Vector of climate vulnerability scores
        """
        N = len(corr_matrix)
        S = np.random.choice([-1, 1], size=N)  # -1 = sell, +1 = buy
        J = np.copy(corr_matrix.values)
        alpha = 0.5  # Climate sensitivity
        
        for i in range(N):
            neighbor_influence = np.sum(J[i] * S)
            climate_effect = alpha * climate_stress[i] * np.mean(S)
            h = neighbor_influence - climate_effect
            p = 1 / (1 + np.exp(-2*h))
            S[i] = 1 if np.random.random() < p else -1
            
        return S

# ======================
# VISUALIZATION
# ======================

class Visualizer:
    """Visualization tools for results"""
    
    @staticmethod
    def plot_efficient_frontier(optimizer):
        """Plot traditional efficient frontier"""
        results = optimizer.traditional_frontier()
        
        plt.scatter(results[1,:], results[0,:], c=results[2,:], cmap='viridis', alpha=0.3)
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility')
        plt.ylabel('Return')
        plt.title('Efficient Frontier')
        plt.show()
    
    @staticmethod
    def plot_climate_sensitivity(returns, climate_data):
        """Plot asset returns vs climate indicators"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 12))
        
        for i, (asset, ret) in enumerate(returns.items()):
            if i >= 3: break  # Just show first 3 for example
            
            axes[0].plot(climate_data['temp_anomaly'], ret, 'o', label=asset, alpha=0.5)
            axes[1].plot(climate_data['weather_events'], ret, 'o', label=asset, alpha=0.5)
            axes[2].plot(climate_data['carbon_price'], ret, 'o', label=asset, alpha=0.5)
        
        axes[0].set_ylabel('Returns')
        axes[0].set_xlabel('Temperature Anomaly')
        axes[0].legend()
        
        axes[1].set_ylabel('Returns')
        axes[1].set_xlabel('Extreme Weather Events')
        
        axes[2].set_ylabel('Returns')
        axes[2].set_xlabel('Carbon Price')
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_sentiment_evolution(opinions_history):
        """Plot opinion convergence over time"""
        plt.figure(figsize=(12, 6))
        for i in range(opinions_history.shape[1]):
            plt.plot(opinions_history[:, i], alpha=0.5)
        plt.xlabel('Time Step')
        plt.ylabel('Opinion Value')
        plt.title('Investor Sentiment Evolution')
        plt.show()

# ======================
# MAIN ANALYSIS
# ======================

def main():
    print("Climate-Stressed Portfolio Optimization")
    print("Loading data...")
    
    # 1. Load and prepare data
    loader = ClimateDataLoader()
    returns = loader.load_financial_data()
    climate_data = loader.load_climate_data()
    combined_data = loader.get_combined_dataset()
    
    # 2. Traditional portfolio optimization
    print("\nRunning traditional portfolio optimization...")
    optimizer = PortfolioOptimizer(returns)
    Visualizer.plot_efficient_frontier(optimizer)
    
    # Optimal portfolio weights
    optimal_weights = optimizer.optimize_portfolio()
    print("\nOptimal Weights (Traditional):")
    for ticker, weight in zip(returns.columns, optimal_weights):
        print(f"{ticker}: {weight:.2%}")
    
    # 3. Climate stress analysis
    print("\nAnalyzing climate risk factors...")
    Visualizer.plot_climate_sensitivity(returns, climate_data)
    
    # 4. Catastrophe model simulation
    print("\nRunning catastrophe model simulation...")
    X = np.zeros(100)  # Portfolio value
    X[0] = 1
    climate_stress = np.linspace(0, 2, 100)  # Increasing climate stress
    
    for t in range(1, 100):
        X[t] = X[t-1] + ClimateRiskModels.catastrophe_model(
            X[t-1], climate_stress[t], 
            params=[0.5, -0.2, 1.0, 0.3, 0.1]
        )
    
    plt.plot(climate_stress, X)
    plt.xlabel('Climate Stress Level')
    plt.ylabel('Portfolio Value')
    plt.title('Catastrophe Model: Portfolio Value vs Climate Stress')
    plt.show()
    
    # 5. Investor sentiment modeling
    print("\nModeling investor sentiment shifts...")
    initial_opinions = np.random.uniform(0, 1, 50)  # 50 investors
    climate_shock = 1.5  # Major climate policy announcement
    
    opinions_history = np.zeros((100, len(initial_opinions)))
    opinions_history[0] = initial_opinions
    
    for t in range(1, 100):
        opinions_history[t] = ClimateRiskModels.sentiment_diffusion(
            opinions_history[t-1], climate_shock
        )
    
    Visualizer.plot_sentiment_evolution(opinions_history)
    
    # 6. Market contagion simulation
    print("\nSimulating market contagion under climate stress...")
    corr_matrix = returns.corr()
    climate_vulnerability = np.random.uniform(0, 1, len(returns.columns))  # Simulated vulnerability scores
    
    final_states = ClimateRiskModels.market_contagion(corr_matrix, climate_vulnerability)
    print("\nFinal Market States (1=buy, -1=sell):")
    for ticker, state in zip(returns.columns, final_states):
        print(f"{ticker}: {'Buy' if state > 0 else 'Sell'}")
    
    # 7. Climate-adjusted portfolio optimization
    print("\nCalculating climate-adjusted portfolio metrics...")
    var_traditional = optimizer.climate_stressed_var(optimal_weights, climate_factor=0)
    var_stressed = optimizer.climate_stressed_var(optimal_weights, climate_factor=2)
    
    print(f"\nValue at Risk (95% confidence):")
    print(f"Traditional: {var_traditional:.2%}")
    print(f"Climate-Stressed: {var_stressed:.2%}")

if __name__ == "__main__":
    main()