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


#I am leanring this seaborn package thing slowly
plt.style.use('seaborn-v0_8') #needed for the nice shading
plt.rcParams.update({'figure.figsize': (16, 10), 'axes.grid': True}) #stacking plots


#---------Random Matrix Cleaner-------------------
#-----------------------------------------------------
class RMT_cleaner: #random matrix theory to cleasn data
    def clean(self, returns):
        T, N = returns.shape #get dimensions
        q = T / N #observations to varibles, small q = noisy
        corr_matrix = returns.corr().values #correlation matrix to array
        eigenvals, eigenvecs = eigh(corr_matrix)#imported, finding of eigenval/vecs
        lambda_plus = (1 + (1 / np.sqrt(q))) ** 2 #form pastur distrib, max eigenvalue fior 100% noise
        informative = eigenvals > lambda_plus #noise thres
        cleaned_corr = (eigenvecs[:, informative] @ 
                        np.diag(eigenvals[informative]) @ 
                        eigenvecs[:, informative].T) #reconstructed corr matrix w/o noise
        std_devs = returns.std().values
        return np.outer(std_devs, std_devs) * cleaned_corr #return stand dev matrix, outer product


#-----------Imported Markov regime detector-------------
#------------------------------------------------------
class hmm_regime_det:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.scaler = StandardScaler() #mean 0 std 1 , imported
        
    def fit(self, returns):
        clean_returns = returns.dropna().replace([np.inf, -np.inf], 0) #dropna to remove empties and infinites
        features = pd.DataFrame({
            'returns': clean_returns,
            'abs_returns': np.abs(clean_returns),
            'squared_returns': clean_returns**2,
            'vol_5d': clean_returns.rolling(5).std().bfill(),
            'momentum_5d': clean_returns.rolling(5).mean().bfill() #5d = 5 day rolling
        }).dropna()
        #model training
        X = self.scaler.fit_transform(features.values) #mean 0 std 1 for hmm 
        self.model = hmm.GaussianHMM(n_components=self.n_states, covariance_type="full", #imported guassian hmm
                                    n_iter=200, random_state=42) #reporducable, full covar matrix betw/ states
        self.model.fit(X) #raining using E-M alg
        self.states = self.model.predict(X) #assigning regime to each time period
        return self
#combined with GARCH model to imprive hmm results, just linearly combined not integrated to HMM

#-----------Standard Garch model imported again----------------
#-------------------------------------------------------
class GARCH_vol:
    def fit(self, returns):
        clean_returns = returns.dropna() * 100 #data cleaning, %
        q01, q99 = clean_returns.quantile([0.01, 0.99]) #clipping extrema
        clean_returns = clean_returns.clip(q01, q99)#caps end percentiles

        self.model = arch_model(clean_returns, vol='GARCH', p=1, q=1, rescale=False)#imported GARCH
        self.fitted_model = self.model.fit(disp='off') #maximum liklihood est
        return self
        
    def get_volatility(self): #easier to do it separate
        return np.sqrt(self.fitted_model.conditional_volatility) / 100


#----------------Portfolio Management, Modern Portfolio Theory-------------
#-------------------------------------------------------------
class Portfolio_manager:
    def __init__(self, capital=100000):
        self.capital = capital
        
        self.oil_gas_assets = ['XOM', 'CVX', 'COP', 'EOG', 'PSX', 'VLO'] #tickers, random kinda
        self.renewable_assets = ['ICLN', 'PBW', 'QCLN', 'ENPH', 'FSLR']
        self.all_assets = self.oil_gas_assets + self.renewable_assets
        
        self.energy_sectors = {
            'Oil & Gas': self.oil_gas_assets,
            'Renewable Energy': self.renewable_assets
        }
        
        self.lookback_period = 5*365 #5 years was needed to see HM meaningfully contribute
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=self.lookback_period)
        
#--------data loading------------------------
#-------------------------------------------
    def load_data(self):
        #oil data in gerneal, to detect over all volitlity of the oil market
        oil_data = yf.Ticker("CL=F").history(start=self.start_date, end=self.end_date) #crude oil futres CL=F
        self.oil_returns = oil_data['Close'].pct_change().dropna() #cleaned
        
        #asset data
        self.prices = pd.DataFrame({
            ticker: yf.Ticker(ticker).history(start=self.start_date, end=self.end_date)['Close']
            for ticker in self.all_assets
        }).ffill().dropna()
        self.returns = self.prices.pct_change().dropna()
        return self

#----------Calling Funcitons for a linear weighting of Garch and HMM methods-------------
#-----------------------------------------------------------------
    def detect_regimes(self):
        self.hmm = hmm_regime_det().fit(self.oil_returns) #some specific syntax
        self.garch = GARCH_vol().fit(self.oil_returns)
        return self

    def optimize_portfolio(self, lookback=252): #lessen to speed things up, this is why the graph is cut off
        self.weights_history = pd.DataFrame(index=self.returns.index, columns=self.all_assets).fillna(0.0) #sotring portfolio weights
        oil_vol = self.garch.get_volatility()
    
        for i in range(lookback, len(self.returns)):
            window_returns = self.returns.iloc[i-lookback:i] #rolling window
            cov_matrix = RMT_cleaner().clean(window_returns) * 252 #rnadom matrix cleaning of window
            
            current_vol = oil_vol.iloc[min(i, len(oil_vol)-1)] #combined volitiltys from each method
            current_regime = self.hmm.states[min(i, len(self.hmm.states)-1)] #gets market regime
            #to combine the inputs a multiplier is used to increase 'vol.' with a multiplier based on regime
            regime_multiplier = 1.2 if current_regime == 2 else 1.0 if current_regime == 1 else 0.8
            combined_vol = current_vol * regime_multiplier
            
            target_vol = 0.18 if combined_vol > 0.05 else 0.22 if combined_vol > 0.03 else 0.25 #needed to manage portfolio

            #------------------Risk Calculation and Allowcation------------
            #-------------------------------------------------------------------
            def portfolio_risk(w): 
                return np.sqrt(w.T @ cov_matrix @ w) #stand dev of profolio var
            
            result = minimize(
                portfolio_risk,
                x0=np.ones(len(window_returns.columns))/len(window_returns.columns), #x0 initial guess (equal weighting)
                constraints=[
                    {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}, #sum of wieghts
                    {'type': 'ineq', 'fun': lambda x: target_vol - portfolio_risk(x)} #risk target
                ],
                bounds=[(0, 0.15)] * len(window_returns.columns),
                method='SLSQP'
            )
            
            weights = result.x / result.x.sum() #sums to 1
            current_date = self.returns.index[i]
            
            for asset, weight in zip(window_returns.columns, weights):
                self.weights_history.at[current_date, asset] = weight
            
            if i == len(self.returns) - 1:
                self.weights = dict(zip(window_returns.columns, weights))
        
        return self
    

    #------------Visualisation--------------------------
    #--------------------------------------------------
    #now to define the graphes needed
    def visualize_results(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 18))
        #oil plot - with volatility background colours
        oil_cumulative = (1 + self.oil_returns).cumprod()
        ax1.plot(oil_cumulative, label='Oil Returns', color='blue')
        ax1.set_ylabel('Returns', color='blue')
        oil_vol = self.garch.get_volatility()
        #to add colored background volatility
        vol_colors = LinearSegmentedColormap.from_list('vol_cmap', ['green', 'yellow', 'red']) #imported
        for i in range(len(oil_vol)-1):
            ax1.axvspan(oil_vol.index[i], oil_vol.index[i+1], 
                   color=vol_colors(oil_vol.iloc[i]/oil_vol.max()), alpha=0.1)
    
        ax1.set_title('Oil Market Returns')
    
        #renewable energy plot - with volatility background colours
        renewable = self.prices[self.energy_sectors['Renewable Energy']].mean(axis=1)
        renewable_returns = renewable.pct_change()
        renewable_cumulative = (1 + renewable_returns).cumprod()
        ax2.plot(renewable_cumulative, label='Renewable Returns', color='green')
        ax2.set_ylabel('Returns', color='green')
    
        renew_vol = renewable_returns.rolling(20).std() * np.sqrt(252)
        #add colored background volatility
        for i in range(len(renew_vol)-1):
            ax2.axvspan(renew_vol.index[i], renew_vol.index[i+1], 
                    color=vol_colors(renew_vol.iloc[i]/renew_vol.max()), alpha=0.1)
        ax2.set_title('Renewable Energy Returns')
    
        #portfolio allocation plot
        weights_df = pd.DataFrame(index=self.weights_history.index)
        for sector, assets in self.energy_sectors.items():
            weights_df[sector] = self.weights_history[assets].sum(axis=1)
        weights_df['Other'] = 1 - weights_df.sum(axis=1)   
        #lines
        colors = {
            'Oil & Gas': 'darkblue',
            'Renewable Energy': 'green',
            'Other': 'gray'
        }
        
        for column in weights_df.columns:
            ax3.plot(weights_df[column], 
                    label=column,
                    color=colors.get(column, 'gray'))
        
        ax3.set_title('Portfolio Allocation vs Time')
        ax3.set_ylabel('% Allocation')
        ax3.legend(loc='upper left')
        ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax3.set_ylim(0, 1)
        
        current_vol = self.garch.get_volatility().iloc[-1]
        oil_allocation = sum(self.weights.get(a, 0) for a in self.energy_sectors['Oil & Gas'])
        renew_allocation = sum(self.weights.get(a, 0) for a in self.energy_sectors['Renewable Energy'])
        
        adjustment_text = (
            f"Current Portfolio Allocation (Volatility: {current_vol:.2%}):\n"
            f"Oil & Gas: {oil_allocation:.1%}\n"
            f"Renewable Energy: {renew_allocation:.1%}"
        )
        fig.text(0.75, 0.05, adjustment_text, bbox=dict(facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig

#--------Exicute-----------------
#------------------------------
    def run_analysis(self):
        self.load_data()
        self.detect_regimes()
        self.optimize_portfolio()
        fig = self.visualize_results()
        plt.show()
        return self

if __name__ == "__main__":
    Portfolio_manager().run_analysis()