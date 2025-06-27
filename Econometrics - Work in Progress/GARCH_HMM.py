import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.cluster import KMeans
from scipy.stats import norm
from scipy.optimize import minimize
import yfinance as yf
import ruptures as rpt
import warnings
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

class OilShockWarningModel:
    
    def __init__(self, n_states=4, garch_p=1, garch_q=1, max_iter=100, tol=1e-6, 
                 init_method='bcpd', shock_threshold=0.7):
        self.n_states = n_states
        self.garch_p = garch_p  
        self.garch_q = garch_q
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method.lower()
        self.shock_threshold = shock_threshold 
        self.fitted = False
        
       
        self.state_labels = {
            0: "Low Volatility",
            1: "Normal Volatility", 
            2: "Pre-Shock",
            3: "High Volatility/Shock"
        }
        
    def fit(self, returns):
        # model to oil price returns
        self.returns = returns.copy()
        self.T = len(returns)
        
        
        self._initialize_parameters()
        
        log_likelihood_prev = -np.inf
        self.log_likelihoods = []
        
        for iteration in range(self.max_iter):
            xi, gamma, log_likelihood = self._expectation_step()
            self._maximization_step(xi, gamma)
            self.log_likelihoods.append(log_likelihood)
            
            if abs(log_likelihood - log_likelihood_prev) < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break
                
            log_likelihood_prev = log_likelihood
            
        
        self.states = self._viterbi_decode()
        self.state_probabilities = gamma
        
        
        self._calculate_shock_metrics()
        
        
        n_params = self._count_parameters()
        self.aic = -2 * log_likelihood + 2 * n_params
        self.bic = -2 * log_likelihood + np.log(self.T) * n_params
        
        self.fitted = True
        print(f"Final log-likelihood: {log_likelihood:.4f}")
        print(f"AIC: {self.aic:.4f}, BIC: {self.bic:.4f}")
        
        return self
    
    def _calculate_shock_metrics(self):
        #calculate shock probability
        
        self.preshock_prob = self.state_probabilities[:, 2]
        self.shock_prob = self.state_probabilities[:, 3]
        
        #risk score
        self.risk_score = 0.6 * self.preshock_prob + 0.4 * self.shock_prob
        
        #current market status
        current_risk = self.risk_score[-1]
        current_state = self.states[-1]
        
        self.current_warning_level = self._get_warning_level(current_risk, current_state)
        
    def _get_warning_level(self, risk_score, current_state):
        #determine warning level based on risk score and HMM state
        if risk_score > 0.8 or current_state == 3:
            return "CRITICAL - Shock Imminent"
        elif risk_score > 0.6 or current_state == 2:
            return "HIGH - Pre-Shock Conditions"
        elif risk_score > 0.4:
            return "MODERATE - Elevated Risk"
        else:
            return "LOW - Normal Conditions"
    
    def get_current_warning(self):
        
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        current_risk = self.risk_score[-1]
        current_state = self.states[-1]
        current_preshock = self.preshock_prob[-1] 
        
        warning_dict = {
            'warning_level': self.current_warning_level,
            'risk_score': current_risk,
            'current_state': self.state_labels[current_state],
            'preshock_probability': current_preshock,
            'shock_probability': self.shock_prob[-1],
            'days_in_current_state': self._days_in_current_state()
        }
        
        return warning_dict
    
    def _days_in_current_state(self):
        
        current_state = self.states[-1]
        days = 1
        
        for i in range(len(self.states) - 2, -1, -1):
            if self.states[i] == current_state:
                days += 1
            else:
                break
                
        return days

    
    def _initialize_parameters(self):
        
        if self.init_method == 'bcpd':
            model = rpt.Pelt(model="rbf").fit(self.returns.values)
            breaks = model.predict(pen=10)
            
            labels = np.zeros(self.T, dtype=int)
            prev = 0
            for i, brk in enumerate(breaks):
                labels[prev:brk] = i
                prev = brk
            
            labels = labels % self.n_states
            self._bcpd_breaks = breaks
        else:
            returns_2d = self.returns.values.reshape(-1, 1)
            kmeans = KMeans(n_clusters=self.n_states, random_state=42, n_init=10)
            labels = kmeans.fit_predict(returns_2d)
            self._bcpd_breaks = []
        
        # transition matrix with transition bias
        self.A = np.zeros((self.n_states, self.n_states))
        for t in range(1, self.T):
            self.A[labels[t-1], labels[t]] += 1
        
        self.A += 0.01
        self.A[1, 2] += 0.05  # Normal to Pre-shock
        self.A[2, 3] += 0.1   # Pre-shock to Shock
        
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        
        self.pi = np.bincount(labels, minlength=self.n_states) / self.T
        
        #initialize GARCH params
        self.garch_params = {}
        self.conditional_variances = np.zeros((self.T, self.n_states))
        
        #ordering by volitility
        state_volatilities = {}
        for state in range(self.n_states):
            state_mask = labels == state
            state_returns = self.returns[state_mask]
            
            if len(state_returns) > 5:
                state_volatilities[state] = state_returns.std()
            else:
                state_volatilities[state] = self.returns.std()
        
        
        sorted_states = sorted(state_volatilities.items(), key=lambda x: x[1])
        
        for idx, (state, vol) in enumerate(sorted_states):
            state_mask = labels == state
            state_returns = self.returns[state_mask]
            
            if len(state_returns) < 10:
                
                base_var = self.returns.var()
                multipliers = [0.3, 0.7, 1.2, 2.0]  # Low, Normal, Pre-shock, Shock
                omega = base_var * multipliers[idx]
                alpha = 0.1 + idx * 0.05
                beta = 0.8 - idx * 0.1
            else:
                try:
                    garch_model = arch_model(state_returns, vol='GARCH', 
                                           p=self.garch_p, q=self.garch_q)
                    garch_fit = garch_model.fit(disp='off')
                    
                    omega = garch_fit.params['omega']
                    alpha = garch_fit.params.get('alpha[1]', 0.1)
                    beta = garch_fit.params.get('beta[1]', 0.8)
                except:
                    base_var = state_returns.var()
                    multipliers = [0.3, 0.7, 1.2, 2.0]
                    omega = base_var * multipliers[min(idx, 3)]
                    alpha = 0.1 + idx * 0.05
                    beta = 0.8 - idx * 0.1
            
            omega = max(omega, 1e-6)
            alpha = np.clip(alpha, 0.001, 0.999)
            beta = np.clip(beta, 0.001, 0.999)
            
            if alpha + beta >= 1:
                alpha = 0.1 + idx * 0.05
                beta = 0.8 - idx * 0.1
                
            self.garch_params[state] = {
                'omega': omega,
                'alpha': alpha, 
                'beta': beta
            }
            
            self.conditional_variances[:, state] = self._compute_conditional_variance(
                self.returns, **self.garch_params[state]
            )
    
    def _compute_conditional_variance(self, returns, omega, alpha, beta):
        
        T = len(returns)
        sigma2 = np.zeros(T)
        sigma2[0] = omega / (1 - alpha - beta)
        
        for t in range(1, T):
            sigma2[t] = omega + alpha * returns.iloc[t-1]**2 + beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], 1e-8)
            
        return sigma2
    
    def _expectation_step(self):
        
        for state in range(self.n_states):
            self.conditional_variances[:, state] = self._compute_conditional_variance(
                self.returns, **self.garch_params[state]
            )
        
        alpha = np.zeros((self.T, self.n_states))
        c = np.zeros(self.T)
        
        for state in range(self.n_states):
            alpha[0, state] = self.pi[state] * self._emission_probability(0, state)
        
        c[0] = alpha[0].sum()
        alpha[0] /= c[0]
        
        for t in range(1, self.T):
            for j in range(self.n_states):
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self._emission_probability(t, j)
            
            c[t] = alpha[t].sum()
            if c[t] > 0:
                alpha[t] /= c[t]
            else:
                alpha[t] = 1.0 / self.n_states
                c[t] = 1.0
        
        beta = np.zeros((self.T, self.n_states))
        beta[-1] = 1.0
        
        for t in range(self.T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.A[i] * beta[t+1] * 
                    np.array([self._emission_probability(t+1, j) for j in range(self.n_states)]))
            
            if beta[t].sum() > 0:
                beta[t] /= c[t+1]
        
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)
        
        xi = np.zeros((self.T-1, self.n_states, self.n_states))
        for t in range(self.T-1):
            for i in range(self.n_states):
                for j in range(self.n_states):
                    xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                  self._emission_probability(t+1, j) * beta[t+1, j])
            
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum
        
        log_likelihood = np.sum(np.log(c + 1e-300))
        
        return xi, gamma, log_likelihood
    
    def _emission_probability(self, t, state):
        
        return_t = self.returns.iloc[t]
        variance_t = self.conditional_variances[t, state]
        
        prob = (1.0 / np.sqrt(2 * np.pi * variance_t)) * \
               np.exp(-0.5 * return_t**2 / variance_t)
        
        return max(prob, 1e-300)
    
    def _maximization_step(self, xi, gamma):
        
        self.pi = gamma[0]
        
        xi_sum = xi.sum(axis=0)
        gamma_sum = gamma[:-1].sum(axis=0)
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                if gamma_sum[i] > 0:
                    self.A[i, j] = xi_sum[i, j] / gamma_sum[i]
                else:
                    self.A[i, j] = 1.0 / self.n_states
        
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        for state in range(self.n_states):
            self._update_garch_parameters(state, gamma[:, state])
    
    def _update_garch_parameters(self, state, state_probs):
        
        def neg_log_likelihood(params):
            omega, alpha, beta = params
            
            if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1:
                return 1e6
            
            sigma2 = np.zeros(self.T)
            sigma2[0] = omega / (1 - alpha - beta)
            
            for t in range(1, self.T):
                sigma2[t] = omega + alpha * self.returns.iloc[t-1]**2 + beta * sigma2[t-1]
                if sigma2[t] <= 0:
                    return 1e6
            
            log_lik = 0
            for t in range(self.T):
                log_lik += state_probs[t] * (
                    -0.5 * np.log(2 * np.pi * sigma2[t]) - 
                    0.5 * self.returns.iloc[t]**2 / sigma2[t]
                )
            
            return -log_lik
        
        current_params = self.garch_params[state]
        x0 = [current_params['omega'], current_params['alpha'], current_params['beta']]
        
        bounds = [(1e-6, None), (1e-6, 0.999), (1e-6, 0.999)]
        constraints = {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]}
        
        try:
            result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', 
                            bounds=bounds, constraints=constraints)
            
            if result.success:
                omega, alpha, beta = result.x
                self.garch_params[state] = {
                    'omega': omega,
                    'alpha': alpha,
                    'beta': beta
                }
            
        except:
            pass
    
    def _viterbi_decode(self):
        #most likely next state using Viterbi algorithm
        delta = np.zeros((self.T, self.n_states))
        psi = np.zeros((self.T, self.n_states), dtype=int)
        
        for state in range(self.n_states):
            delta[0, state] = np.log(self.pi[state]) + np.log(self._emission_probability(0, state))
        
        for t in range(1, self.T):
            for j in range(self.n_states):
                trans_probs = delta[t-1] + np.log(self.A[:, j] + 1e-300)
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = np.max(trans_probs) + np.log(self._emission_probability(t, j))
        
        states = np.zeros(self.T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        
        for t in range(self.T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def _count_parameters(self):
        
        return self.n_states * (self.n_states - 1) + (self.n_states - 1) + 3 * self.n_states
    
    def predict_volatility(self, horizon=1):
       
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        current_state = self.states[-1]
        current_variance = self.conditional_variances[-1, current_state]
        last_return = self.returns.iloc[-1]
        
        params = self.garch_params[current_state]
        
        forecasts = []
        for h in range(1, horizon + 1):
            if h == 1:
                forecast = (params['omega'] + 
                          params['alpha'] * last_return**2 + 
                          params['beta'] * current_variance)
            else:
                unconditional_var = params['omega'] / (1 - params['alpha'] - params['beta'])
                persistence = params['alpha'] + params['beta']
                
                forecast = (unconditional_var + 
                          (forecasts[0] - unconditional_var) * persistence**(h-1))
            
            forecasts.append(forecast)
        
        return np.sqrt(forecasts)
    
    def detect_regime_changes(self, threshold=0.8):

        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        max_probs = np.max(self.state_probabilities, axis=1)
        high_confidence = max_probs > threshold
        
        state_changes = np.diff(self.states) != 0
        change_points = np.where(state_changes)[0] + 1
        
        
        critical_changes = []
        for cp in change_points:
            if cp < len(high_confidence) and high_confidence[cp]:
                if self.states[cp] >= 2:  
                    critical_changes.append(cp)
        
        return critical_changes


class OilDataLoader:
    
    
    @staticmethod
    def load_oil_prices(commodity='CL', period='2y'):
        """
        - 'CL' WTI Crude
        - 'BZ' Brent Crude
        - 'NG' Natural Gas
        """
        try:
            ticker = yf.Ticker(f"{commodity}=F") 
            data = ticker.history(period=period)
            
            if data.empty:
                raise ValueError(f"No data found for {commodity}")
            
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            data = data.dropna()
            
            print(f"Loaded {len(data)} observations for {commodity}")
            print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
            
            return data
            
        except Exception as e:
            print(f"Error loading oil data: {e}")
            return None

def plot_shock_warning_results(model, oil_data, title="Oil Market Shock Warning System"):
   
    if not model.fitted:
        raise ValueError("error")
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 14))
    
    dates = oil_data.index
    prices = oil_data['Close']
    returns = oil_data['Log_Returns']
    colors = ['green', 'blue', 'orange', 'red']  # Low, Normal, Pre-shock, Shock
    
   
    for state in range(model.n_states):
        mask = model.states == state
        axes[0].scatter(dates[mask], prices[mask], 
                       c=colors[state], alpha=0.7, s=15, 
                       label=f'{model.state_labels[state]}')
    
    axes[0].plot(dates, prices, 'k-', alpha=0.3, linewidth=0.5)
    axes[0].set_title(f'{title} - Price by Volatility Regime')
    axes[0].set_ylabel('Price ($)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    
    for state in range(model.n_states):
        mask = model.states == state
        axes[1].scatter(dates[mask], returns[mask], 
                       c=colors[state], alpha=0.6, s=10)
    
    axes[1].plot(dates, returns, 'k-', alpha=0.3, linewidth=0.5)
    axes[1].set_title('Daily Log Returns by Regime')
    axes[1].set_ylabel('Returns')
    axes[1].grid(True, alpha=0.3)
    
    
    axes[2].plot(dates, model.risk_score, 'red', linewidth=2, label='Risk Score')
    axes[2].axhline(y=0.6, color='orange', linestyle='--', alpha=0.7, label='High Risk ')
    axes[2].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Critical Risk ')
    axes[2].fill_between(dates, model.risk_score, alpha=0.3, color='red')
    axes[2].set_title('Market Shock Risk Score')
    axes[2].set_ylabel('Risk Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    
    axes[3].plot(dates, model.preshock_prob, 'orange', linewidth=2, alpha=0.8, label='Pre-Shock Probability')
    axes[3].plot(dates, model.shock_prob, 'red', linewidth=2, alpha=0.8, label='Shock Probability')
    axes[3].fill_between(dates, model.preshock_prob, alpha=0.3, color='orange')
    axes[3].fill_between(dates, model.shock_prob, alpha=0.3, color='red')
    axes[3].set_title('Shock Warning Indicators')
    axes[3].set_ylabel('Probability')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    warning_info = model.get_current_warning()
    
    print("\n" + "="*70)
    print("="*70)
    print(f"Current Warning Level: {warning_info['warning_level']}")
    print(f"Risk Score: {warning_info['risk_score']:.3f}")
    print(f"Current Market State: {warning_info['current_state']}")
    print(f"Pre-Shock Probability: {warning_info['preshock_probability']:.1%}")
    print(f"Shock Probability: {warning_info['shock_probability']:.1%}")
    print(f"Days in Current State: {warning_info['days_in_current_state']}")
    print("="*70)

def analyze_oil_shock_warning():
    #main func
    print("="*70)
    print("🚨 OIL MARKET SHOCK EARLY WARNING SYSTEM 🚨")
    print("="*70)
    print("Loading current oil market data for shock analysis...")
    
    
    oil_data = OilDataLoader.load_oil_prices(commodity='CL', period='2y')
    if oil_data is None:
        print("Failed to load oil data")
        return
    
    returns = oil_data['Log_Returns']
    
    print("\nLoading shock warning model..")
    model = OilShockWarningModel(n_states=4, init_method='bcpd', shock_threshold=0.7)
    model.fit(returns)
    
    
    plot_shock_warning_results(model, oil_data, "WTI Crude Oil Shock Warning System")
    
   
    current_warning = model.get_current_warning()
    
    
    vol_forecast = model.predict_volatility(horizon=5)
    print(f"\nVolatility forecast (next 5 days): {vol_forecast}")
    
    
    critical_changes = model.detect_regime_changes()
    if critical_changes:
        change_dates = [oil_data.index[i].strftime('%Y-%m-%d') for i in critical_changes]
        print(f"\nCritical volatility regime changes detected: {change_dates}")
    
    print(f"Market Assessment:")
    print(f"The oil market is currently showing {current_warning['warning_level'].lower()} conditions")
    
    if current_warning['risk_score'] > 0.6:
        print("Consider risk management strategies for oil-exposed positions")

if __name__ == "__main__":
    analyze_oil_shock_warning()