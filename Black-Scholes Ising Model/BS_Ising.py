# -*- coding: utf-8 -*-
#!/usr/bin/env python

"""
Sentiment-Driven Option Pricing using Ising Model and Black-Scholes Framework
Author: Based on Prof. Carlo R. da Cunha's work
Enhanced for sentiment-driven volatility modeling
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

class EnhancedBlackScholes:
    """Enhanced Black-Scholes with full Greeks calculation"""
    
    @staticmethod
    def black_scholes_full(S, K, T, r, sigma, option_type='call'):
        """
        Complete Black-Scholes pricing with Greeks
        
        Returns: (price, delta, gamma, theta, vega, rho)
        """
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type.lower() == 'call':
            price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            delta = norm.cdf(d1)
            rho = K*T*np.exp(-r*T)*norm.cdf(d2)
        else:  # put
            price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            delta = -norm.cdf(-d1)
            rho = -K*T*np.exp(-r*T)*norm.cdf(-d2)
        
        # Greeks (same for calls and puts)
        gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
        theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
                 r*K*np.exp(-r*T)*norm.cdf(d2 if option_type.lower()=='call' else -d2))
        vega = S*norm.pdf(d1)*np.sqrt(T)
        
        return price, delta, gamma, theta/365, vega/100, rho/100
    
    @staticmethod
    def implied_volatility(market_price, S, K, T, r, option_type='call'):
        """Extract implied volatility using Brent's method"""
        def objective(sigma):
            try:
                bs_price, _, _, _, _, _ = EnhancedBlackScholes.black_scholes_full(
                    S, K, T, r, sigma, option_type)
                return bs_price - market_price
            except:
                return float('inf')
        
        try:
            return brentq(objective, 0.001, 3.0)
        except:
            return np.nan

class SentimentIsing:
    """
    Ising model for market sentiment dynamics
    Adapted from Bornholdt model for financial applications
    """
    
    def __init__(self, N=32, J=1.0, alpha=4.0, beta=1.0/1.5):
        """
        N: Grid size (N x N traders)
        J: Interaction strength (herding parameter)
        alpha: Contrarian strength
        beta: Inverse temperature (1/uncertainty)
        """
        self.N = N
        self.J = J
        self.alpha = alpha
        self.beta = beta
        
        # Initialize sentiment grid: +1 = bullish, -1 = bearish
        self.S = np.random.choice([-1, 1], size=(N, N))
        # Initialize contrarian indicators
        self.C = np.random.choice([-1, 1], size=(N, N))
        
        self.sentiment_history = []
        self.external_field = 0.0  # News, fundamentals, etc.
        
    def get_neighbors(self, i, j):
        """Get neighboring traders for interaction"""
        neighbors = []
        for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
            ni, nj = (i + di) % self.N, (j + dj) % self.N
            neighbors.append((ni, nj))
        return neighbors
    
    def evolve_step(self, dt=1.0):
        """Single Monte Carlo evolution step"""
        M = np.mean(self.S)  # Market sentiment
        S_new = self.S.copy()
        C_new = self.C.copy()
        
        # Update each trader
        for i in range(self.N):
            for j in range(self.N):
                # Calculate local field from neighbors
                neighbor_sum = sum(self.S[ni, nj] for ni, nj in self.get_neighbors(i, j))
                
                # Effective field includes neighbors, contrarians, and external news
                h_eff = (self.J * neighbor_sum - 
                         self.alpha * self.C[i,j] * M + 
                         self.external_field)
                
                # Metropolis update probability
                p_flip = 1.0 / (1 + np.exp(-2 * self.beta * h_eff))
                
                if np.random.random() < p_flip:
                    S_new[i,j] = 1
                else:
                    S_new[i,j] = -1
                
                # Update contrarian behavior
                if self.alpha * self.S[i,j] * self.C[i,j] * neighbor_sum < 0:
                    C_new[i,j] = -self.C[i,j]
        
        self.S = S_new
        self.C = C_new
        
        current_sentiment = np.mean(self.S)
        self.sentiment_history.append(current_sentiment)
        return current_sentiment
    
    def add_external_field(self, field_strength):
        """Add external sentiment (news, earnings, etc.)"""
        self.external_field += field_strength
    
    def reset_external_field(self):
        """Reset external influences"""
        self.external_field = 0.0
    
    def get_sentiment_metrics(self):
        """Get current sentiment statistics"""
        sentiment = np.mean(self.S)
        volatility = np.std(self.S.flatten())
        clustering = np.mean([self.S[i,j] * sum(self.S[ni,nj] 
                            for ni,nj in self.get_neighbors(i,j)) 
                            for i in range(self.N) for j in range(self.N)])
        
        return {
            'sentiment': sentiment,
            'volatility': volatility, 
            'clustering': clustering,
            'bullish_fraction': np.mean(self.S == 1),
            'bearish_fraction': np.mean(self.S == -1)
        }

class SentimentVolatilityModel:
    """Couples sentiment to market volatility"""
    
    def __init__(self, base_vol=0.2, alpha=0.5, beta=0.3, gamma=0.1):
        """
        base_vol: Base volatility level
        alpha: Linear sentiment impact
        beta: Sentiment clustering effect (herding amplification)  
        gamma: Sentiment momentum effect
        """
        self.base_vol = base_vol
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.vol_history = []
    
    def sentiment_to_volatility(self, sentiment_metrics):
        """
        Transform sentiment metrics to volatility
        
        σ(t) = σ_base * (1 + α|S(t)| + β*clustering + γ*momentum)
        """
        s = sentiment_metrics['sentiment']
        clustering = abs(sentiment_metrics['clustering'])
        
        # Momentum from recent sentiment changes
        momentum = 0
        if len(self.vol_history) > 1:
            recent_changes = np.diff(self.vol_history[-10:])
            momentum = np.std(recent_changes) if len(recent_changes) > 0 else 0
        
        # Volatility scaling factors
        sentiment_factor = 1 + self.alpha * abs(s)
        clustering_factor = 1 + self.beta * clustering  
        momentum_factor = 1 + self.gamma * momentum
        
        vol = self.base_vol * sentiment_factor * clustering_factor * momentum_factor
        self.vol_history.append(vol)
        
        return vol

class SentimentDrivenOptionPricer:
    """Main class combining sentiment model with option pricing"""
    
    def __init__(self, sentiment_model, volatility_model, bs_model):
        self.sentiment_model = sentiment_model
        self.volatility_model = volatility_model
        self.bs_model = bs_model
        self.price_history = []
        
    def price_with_sentiment_evolution(self, S0, K, T, r, n_steps=100, 
                                     external_events=None):
        """
        Price options with evolving sentiment
        
        external_events: List of (time_step, event_strength) tuples
        """
        dt = T / n_steps
        prices = []
        sentiments = []
        volatilities = []
        
        for step in range(n_steps):
            # Add external events if specified
            if external_events:
                for event_time, strength in external_events:
                    if abs(step - event_time * n_steps) < 1:
                        self.sentiment_model.add_external_field(strength)
            
            # Evolve sentiment
            current_sentiment = self.sentiment_model.evolve_step(dt)
            sentiment_metrics = self.sentiment_model.get_sentiment_metrics()
            
            # Update volatility based on sentiment
            current_vol = self.volatility_model.sentiment_to_volatility(sentiment_metrics)
            
            # Price option with sentiment-adjusted volatility
            time_to_expiry = T - step * dt
            if time_to_expiry > 0:
                price, delta, gamma, theta, vega, rho = self.bs_model.black_scholes_full(
                    S0, K, time_to_expiry, r, current_vol, 'call')
                
                prices.append(price)
                sentiments.append(current_sentiment)
                volatilities.append(current_vol)
                
                # Feedback: high option activity can influence sentiment
                if len(prices) > 1:
                    price_change = abs(prices[-1] - prices[-2])
                    if price_change > 0.1:  # Significant price move
                        feedback_strength = 0.1 * np.sign(prices[-1] - prices[-2])
                        self.sentiment_model.add_external_field(feedback_strength)
            
            # Decay external influences
            self.sentiment_model.external_field *= 0.95
        
        return {
            'prices': prices,
            'sentiments': sentiments, 
            'volatilities': volatilities,
            'time_steps': np.linspace(0, T, len(prices))
        }
    
    def analyze_meme_stock_event(self, ticker='GME', start_date='2021-01-01', 
                               end_date='2021-02-28'):
        """
        Analyze historical meme stock events
        Compare model predictions with actual option prices
        """
        # This would fetch real data and compare with model predictions
        # Implementation would require options data API
        print(f"Analyzing {ticker} sentiment-driven volatility...")
        print(f"Period: {start_date} to {end_date}")
        
        # Simulate extreme sentiment event (like Reddit squeeze)
        extreme_events = [(0.1, 2.0), (0.3, -1.5), (0.7, 1.8)]  # (time, strength)
        
        results = self.price_with_sentiment_evolution(
            S0=100, K=110, T=0.25, r=0.01, n_steps=50,
            external_events=extreme_events
        )
        
        return results

def run_sentiment_pricing_demo():
    """Demonstration of the sentiment-driven option pricing model"""
    
    # Initialize models
    sentiment_model = SentimentIsing(N=20, J=1.2, alpha=3.0)
    volatility_model = SentimentVolatilityModel(base_vol=0.25, alpha=0.8, beta=0.4)
    bs_model = EnhancedBlackScholes()
    
    # Create main pricer
    pricer = SentimentDrivenOptionPricer(sentiment_model, volatility_model, bs_model)
    
    # Simulate meme stock scenario
    print("Running Sentiment-Driven Option Pricing Demo...")
    print("Simulating meme stock volatility with Reddit-style events...")
    
    results = pricer.analyze_meme_stock_event()
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    time_steps = results['time_steps']
    
    # Option prices over time
    ax1.plot(time_steps, results['prices'], 'b-', linewidth=2)
    ax1.set_title('Option Price Evolution')
    ax1.set_xlabel('Time to Expiry')
    ax1.set_ylabel('Option Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # Sentiment evolution
    ax2.plot(time_steps, results['sentiments'], 'r-', linewidth=2)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax2.set_title('Market Sentiment Evolution')
    ax2.set_xlabel('Time to Expiry')
    ax2.set_ylabel('Sentiment (-1=Bearish, +1=Bullish)')
    ax2.grid(True, alpha=0.3)
    
    # Volatility evolution
    ax3.plot(time_steps, results['volatilities'], 'g-', linewidth=2)
    ax3.set_title('Sentiment-Driven Volatility')
    ax3.set_xlabel('Time to Expiry')
    ax3.set_ylabel('Volatility')
    ax3.grid(True, alpha=0.3)
    
    # Sentiment vs Volatility correlation
    ax4.scatter(results['sentiments'], results['volatilities'], 
               c=time_steps, cmap='viridis', alpha=0.7)
    ax4.set_title('Sentiment-Volatility Relationship')
    ax4.set_xlabel('Market Sentiment')
    ax4.set_ylabel('Volatility')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return results

if __name__ == "__main__":
    # Run the demonstration
    results = run_sentiment_pricing_demo()
    
    print("\nModel Summary:")
    print(f"Final option price: ${results['prices'][-1]:.2f}")
    print(f"Final sentiment: {results['sentiments'][-1]:.3f}")
    print(f"Final volatility: {results['volatilities'][-1]:.1%}")
    print(f"Volatility range: {min(results['volatilities']):.1%} - {max(results['volatilities']):.1%}")