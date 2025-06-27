import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy.optimize import minimize
from hmmlearn import hmm
from arch import arch_model
from scipy.linalg import eigh
import warnings
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import LinearSegmentedColormap


warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
plt.rcParams.update({'figure.figsize': (16, 10), 'axes.grid': True})

#de-noiser with random matrices
class RMTCleaner:
    def clean(self, returns):
        T, N = returns.shape
        q = T / N
        corr_matrix = returns.corr().values
        eigenvals, eigenvecs = eigh(corr_matrix)
        lambda_plus = (1 + (1 / np.sqrt(q))) ** 2
        informative = eigenvals > lambda_plus
        cleaned_corr = (eigenvecs[:, informative] @ 
                        np.diag(eigenvals[informative]) @ 
                        eigenvecs[:, informative].T)
        std_devs = returns.std().values
        return np.outer(std_devs, std_devs) * cleaned_corr
#Imported HMM model - was easier to do it this way
class HMMRegimeDetector:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.scaler = StandardScaler()
        
    def fit(self, returns):
        clean_returns = returns.dropna().replace([np.inf, -np.inf], 0)
        features = pd.DataFrame({
            'returns': clean_returns,
            'abs_returns': np.abs(clean_returns),
            'squared_returns': clean_returns**2,
            'vol_5d': clean_returns.rolling(5).std().bfill(),
            'momentum_5d': clean_returns.rolling(5).mean().bfill()
        }).dropna()
        
        X = self.scaler.fit_transform(features.values)
        self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full",
                                    n_iter=200, random_state=42)
        self.model.fit(X)
        self.states = self.model.predict(X)
        return self

#Coupled With an imprted GARCH model
class GARCHVolatilityModel:
    def fit(self, returns):
        clean_returns = returns.dropna() * 100
        q01, q99 = clean_returns.quantile([0.01, 0.99])
        clean_returns = clean_returns.clip(q01, q99)
        self.model = arch_model(clean_returns, vol='GARCH', p=1, q=1, rescale=False)
        self.fitted_model = self.model.fit(disp='off')
        return self
        
    def get_volatility(self):
        return np.sqrt(self.fitted_model.conditional_volatility) / 100

#portfolio management WIP
class EnergyPortfolioManager:
    def __init__(self, capital=100000):
        self.capital = capital
        self.energy_sectors = {
            'Oil & Gas Majors': ['XOM', 'CVX', 'COP', 'EOG', 'PSX', 'VLO'],
            'Renewable Energy': ['ICLN', 'PBW', 'QCLN', 'ENPH', 'FSLR']
        }
        self.all_assets = [a for sector in self.energy_sectors.values() for a in sector]
        self.sector_map = {a: s for s, assets in self.energy_sectors.items() for a in assets}
        self.lookback_period = 5*365
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=self.lookback_period)
        self.geopolitical_events = {
            '2020-03-01': 'COVID-19 Pandemic',
            '2022-02-24': 'Russia Invades Ukraine',
            '2023-10-07': ''
        }

    def load_data(self):
        #oil data
        oil_data = yf.Ticker("CL=F").history(start=self.start_date, end=self.end_date)
        self.oil_returns = oil_data['Close'].pct_change().dropna()
        
        #asset data
        self.prices = pd.DataFrame({
            ticker: yf.Ticker(ticker).history(start=self.start_date, end=self.end_date)['Close']
            for ticker in self.all_assets
        }).ffill().dropna()
        self.returns = self.prices.pct_change().dropna()
        return self

    def detect_regimes(self):
        self.hmm = HMMRegimeDetector().fit(self.oil_returns)
        self.garch = GARCHVolatilityModel().fit(self.oil_returns)
        return self

    def optimize_portfolio(self, lookback=252):
        
        
        self.weights_history = pd.DataFrame(index=self.returns.index, columns=self.all_assets)
        self.weights_history = self.weights_history.fillna(0.0)  # Initialize with zeros
        
        for i in range(lookback, len(self.returns)):
            try:
                #rolling returns window
                window_returns = self.returns.iloc[i-lookback:i]
                
                cov_matrix = RMTCleaner().clean(window_returns) * 252
                
                current_vol = self.garch.get_volatility().iloc[i]
                if current_vol > 0.05:
                    target_vol = 0.18
                elif current_vol > 0.03:
                    target_vol = 0.22
                else:
                    target_vol = 0.25
                    
                
                def portfolio_risk(w): 
                    return np.sqrt(w.T @ cov_matrix @ w)
                    
                n_assets = len(window_returns.columns)
                result = minimize(
                    portfolio_risk,
                    x0=np.ones(n_assets)/n_assets,
                    constraints=[
                        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
                        {'type': 'ineq', 'fun': lambda x: target_vol - portfolio_risk(x)}
                    ],
                    bounds=[(0, 0.15) for _ in range(n_assets)],
                    method='SLSQP'
                )
                
                weights = result.x / result.x.sum()
                current_date = self.returns.index[i]
                
                
                for asset, weight in zip(window_returns.columns, weights):
                    self.weights_history.at[current_date, asset] = weight
                    
                if i == len(self.returns) - 1:
                    self.weights = dict(zip(window_returns.columns, weights))
                    
            except Exception as e:
                print(f"Optimization failed at {self.returns.index[i]}: {str(e)}")
                continue
        
        return self
        
        return self

    def visualize_results(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18))
    
        # oil plot
        oil_cumulative = (1 + self.oil_returns).cumprod()
        ax1.plot(oil_cumulative, label='Oil Returns', color='blue')
        ax1.set_ylabel('Cumulative Returns', color='blue')
    
        oil_vol = self.garch.get_volatility()
    
        #shading 
        vol_colors = LinearSegmentedColormap.from_list('vol_cmap', ['green', 'yellow', 'red'])
        for i in range(len(oil_vol)-1):
            ax1.axvspan(oil_vol.index[i], oil_vol.index[i+1], 
                   color=vol_colors(oil_vol.iloc[i]/oil_vol.max()), alpha=0.1)
    
        ax1.set_title('Oil Market: Returns with Volatility Background')
    
        #Renewables
        renewable = self.prices[self.energy_sectors['Renewable Energy']].mean(axis=1)
        renewable_returns = renewable.pct_change()
        renewable_cumulative = (1 + renewable_returns).cumprod()
        ax2.plot(renewable_cumulative, label='Renewable Returns', color='green')
        ax2.set_ylabel('Cumulative Returns', color='green')
    
        renew_vol = renewable_returns.rolling(20).std() * np.sqrt(252)
    
        #shading
        for i in range(len(renew_vol)-1):
            ax2.axvspan(renew_vol.index[i], renew_vol.index[i+1], 
                    color=vol_colors(renew_vol.iloc[i]/renew_vol.max()), alpha=0.1)
    
        ax2.set_title('Renewable Energy: Returns with Volatility Background')
    
        #allocation plot
        weights_df = pd.DataFrame(index=self.weights_history.index)
        for sector, assets in self.energy_sectors.items():
            weights_df[sector] = self.weights_history[assets].sum(axis=1)
        weights_df['Other'] = 1 - weights_df.sum(axis=1)
        
        colors = {
            'Oil & Gas Majors': 'darkblue',
            'Renewable Energy': 'green',
            'Other': 'gray'
        }
        
        for column in weights_df.columns:
            ax3.plot(weights_df[column], 
                    label=column,
                    color=colors.get(column, 'gray'))
        
        ax3.set_title('Dynamic Portfolio Allocation Over Time')
        ax3.set_ylabel('Percentage Allocation')
        ax3.legend(loc='upper left')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax3.set_ylim(0, 1)
        
        
        current_vol = self.garch.get_volatility().iloc[-1]
        oil_allocation = sum(self.weights.get(a, 0) for a in self.energy_sectors['Oil & Gas Majors'])
        renew_allocation = sum(self.weights.get(a, 0) for a in self.energy_sectors['Renewable Energy'])
        
        adjustment_text = (
            f"Current Portfolio Allocation (Volatility: {current_vol:.2%}):\n"
            f"Oil & Gas: {oil_allocation:.1%}\n"
            f"Renewable Energy: {renew_allocation:.1%}"
        )
        fig.text(0.75, 0.05, adjustment_text, bbox=dict(facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def run_analysis(self):
        self.load_data()
        self.detect_regimes()
        self.optimize_portfolio()
        fig = self.visualize_results()
        plt.show()
        return self

if __name__ == "__main__":
    EnergyPortfolioManager().run_analysis() 