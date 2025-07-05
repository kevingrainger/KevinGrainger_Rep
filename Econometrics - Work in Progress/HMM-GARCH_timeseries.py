import pandas as pd
import numpy as np  # Added missing import
import matplotlib.pyplot as plt
from arch import arch_model
from sklearn.cluster import KMeans
from scipy.optimize import minimize
import yfinance as yf
import ruptures as rpt
import warnings
warnings.filterwarnings('ignore')

class GarchHmmOilModel:
   
    def __init__(self, n_states=4, max_iterations=100, tol=1e-6):
        self.n_states = n_states
        self.max_iterations = max_iterations
        self.tol = tol
        self.fitted = False
        
        self.state_labels = {
            0: "Normal Volatility", 1: "High Volatility", 
            2: "Pre-Shock", 3: "Shock"
        }
    

    #using expectation and maximisation functions
    def fit(self, returns):
        #training function for the HMM
        self.returns = returns.copy() #so not to overwrite
        self.T = len(returns) #timeseries save
        #initialize params, HMM transition matrix
        self._initialize_parameters()
        log_likelihood_prev = -1e20 #lower than tol
        #using ruptures lib
        #expecation-maximisation loop
        for iteration in range(self.max_iterations):
            #Expect-step
            xi, gamma, log_likelihood = self._expectation_step()
            
            #Maximisation-step - Update HMM and GARCH parameters
            self._maximization_step(xi, gamma)
            
            if abs(log_likelihood - log_likelihood_prev) < self.tol:
                print(f"Converged after {iteration + 1} iterations")
                break

            log_likelihood_prev = log_likelihood
        

    
        self.states = self._viterbi_decode() #defined later
        self.state_probabilities = gamma #emission prob
        self._calculate_shock_metrics()
        self.fitted = True
        return self #fitted or not fitted
    

    def _initialize_parameters(self):
        
        #using imported tools
        model = rpt.Pelt(model="rbf").fit(self.returns.values) #from the ruptures library, rpt reduces computations, then radial basis function used
        breaks = model.predict(pen=5) #detect change points, penalty reduced to 5 for more sensitivity
        #break = statisitcal change
        #create state labels from change points
        labels = np.zeros(self.T, dtype=int)
        prev = 0 #labelling segments
        for i, seg in enumerate(breaks):
            labels[prev:seg] = i % self.n_states
            prev = seg #labelling segments 
        
        #initialize transition matrix - probabilties of moving between states
        self.A = np.zeros((self.n_states, self.n_states)) #n*n matrix
        for t in range(1, self.T):
            self.A[labels[t-1], labels[t]] += 1 #incrementthrough matrix dpending on lables encountered
        self.A += 0.01  #aoviding zeros
        self.A = self.A / self.A.sum(axis=1, keepdims=True) #normalize to yield probabilites
        
        # probability of starting in each state by counting occurances (crude enough)
        self.pi = np.bincount(labels, minlength=self.n_states) / self.T
        #Initialize GARCH parameters for each state
        self.garch_params = {}
        self.conditional_variances = np.zeros((self.T, self.n_states))
        #Sort states by volatility and assign GARCH parameters
        state_vols = {}
        for state in range(self.n_states):
            state_returns = self.returns[labels == state] #lables the time series as booleans (for each state)
            state_vols[state] = state_returns.std() if len(state_returns) > 5 else self.returns.std() #to stop unreliable and NAN results
        
        # set states volatility level - adjusted for more sensitivity
        vol_multipliers = [0.3, 0.7, 1.5, 2.0]  # Normal, High, Pre-shock, Shock
        base_var = self.returns.var()
        #we need to empericaly calculate the volitilty of each state also,  and give the appropriate labels 
        for idx, state in enumerate(sorted(state_vols.keys(), key=lambda x: state_vols[x])): #.keys gives us the index sorted by vol
            #try fitting individual GARCH, fallback to scaled parameters
            state_returns = self.returns[labels == state]
            
            if len(state_returns) >= 10:  #minimum data points needed for fitting
                try:
                    garch_model = arch_model(state_returns, vol='GARCH', p=1, q=1)
                    garch_fit = garch_model.fit(disp='off')
                    params = garch_fit.params
                    omega = params['omega']
                    alpha = params['alpha[1]']
                    beta = params['beta[1]']
                except Exception as e:
                    #fallback to wider parameters if fitting isnt possible
                    omega = base_var * vol_multipliers[idx] * 0.1
                    alpha = 0.1
                    beta = 0.8
            else:
                #use wider parameters for states with insufficient data
                omega = base_var * vol_multipliers[idx] * 0.1
                alpha = 0.1
                beta = 0.8
                        
            # make sure parameter constraints are met for meaningful results
            omega = max(omega, 1e-6)
            alpha = np.clip(alpha, 0.001, 0.3)
            beta = np.clip(beta, 0.001, 0.95)
            if alpha + beta >= 1:
                alpha, beta = 0.1, 0.85
            
            self.garch_params[state] = {'omega': omega, 'alpha': alpha, 'beta': beta}
            self.conditional_variances[:, state] = self._compute_garch_variance(state)
    

    def _compute_garch_variance(self, state):
        #computes GARCH(1,1) variance for each state
        params = self.garch_params[state]
        omega = params['omega'] #using labels from above
        alpha = params['alpha']
        beta =  params['beta']
        
        sigma2 = np.zeros(self.T)
        sigma2[0] = omega / (1 - alpha - beta)  #long run average varience
        
        for t in range(1, self.T):
            #GARCH formula
            sigma2[t] = omega + alpha * self.returns.iloc[t-1]**2 + beta * sigma2[t-1]
            sigma2[t] = max(sigma2[t], 1e-8)  #avoid zeros
        
        return sigma2
    
    def _expectation_step(self):
        #Expectation step - forward-backward algorithm with GARCH emissions
        #update conditional variances for all states
        for state in range(self.n_states):
            self.conditional_variances[:, state] = self._compute_garch_variance(state) #compute each at every time point
        #forward algorithm,  to predict the current state given past data
        alpha = np.zeros((self.T, self.n_states))
        c = np.zeros(self.T)
        
        #initialize
        for state in range(self.n_states):
            alpha[0, state] = self.pi[state] * self._garch_emission_prob(0, state) #Bayes
        c[0] = alpha[0].sum() #sum of all probs
        alpha[0] /= c[0] #proprtion
        
        #forward pass (back data)
        for t in range(1, self.T):
            for j in range(self.n_states): #prob of state j in time t
                alpha[t, j] = np.sum(alpha[t-1] * self.A[:, j]) * self._garch_emission_prob(t, j)#reversed bayesian
            c[t] = alpha[t].sum()
            alpha[t] /= max(c[t], 1e-300)
        
        log_likelihood = np.sum(np.log(c + 1e-300)) #log yeilds more stability
        #backward algorithm intialization
        beta = np.zeros((self.T, self.n_states))
        beta[-1] = 1.0
        
        for t in range(self.T-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.A[i] * beta[t+1] * 
                    [self._garch_emission_prob(t+1, j) for j in range(self.n_states)]
                )
            beta[t] /= max(c[t+1], 1e-300)#max to prevent zeros
        
        #compute gamma (state probabilities) and xi (transition probabilities)
        gamma = alpha * beta
        gamma = gamma / gamma.sum(axis=1, keepdims=True)#normalise
        #computing our transitions
        xi = np.zeros((self.T-1, self.n_states, self.n_states))
        for t in range(self.T-1):
            for i in range(self.n_states):
                for j in range(self.n_states): #trnaitsion from state i to j at time t
                    xi[t, i, j] = (alpha[t, i] * self.A[i, j] * 
                                  self._garch_emission_prob(t+1, j) * beta[t+1, j])#matrix of 2x1 matrices
            xi_sum = xi[t].sum()
            if xi_sum > 0:
                xi[t] /= xi_sum
        
        
        return xi, gamma, log_likelihood
    
    def _garch_emission_prob(self, t, state):
        #garch - emission probability - how likely each state is given vol.
        return_t = self.returns.iloc[t] #todays price change
        variance_t = self.conditional_variances[t, state] #expected vol. for state 
        
        #normal distrib with GARCH variance
        prob = (1.0 / np.sqrt(2 * np.pi * variance_t)) * \
               np.exp(-0.5 * return_t**2 / variance_t)
        
        return max(prob, 1e-300) # avoid zeros
    
    def _maximization_step(self, xi, gamma):
        #maximisation step - update HMM and GARCH params using E-step
        #update HMM parameters
        self.pi = gamma[0] #probs of each state at t=0
        
        xi_sum = xi.sum(axis=0) #sum transitions
        gamma_sum = gamma[:-1].sum(axis=0) #sum of occupancy/ prob of being in each state
        
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.A[i, j] = xi_sum[i, j] / max(gamma_sum[i], 1e-300) #normalise porbs in matrix
        
        self.A = self.A / self.A.sum(axis=1, keepdims=True)
        
        #update GARCH parameters for each state
        for state in range(self.n_states):
            self._update_garch_parameters(state, gamma[:, state]) # prob of being in each state for each step
    
    def _update_garch_parameters(self, state, state_probs):
        #update GARCH parameters using weighted maximum likelihood
        def neg_log_likelihood(params): # to minimize the negative func
            omega, alpha, beta = params #overall vol, past vol effect, persistance ofpast vol :) 
            
            if omega <= 0 or alpha <= 0 or beta <= 0 or alpha + beta >= 1: #constraints on const.
                return 1e6 #makes sure we dont end up on an invalid value 
            
            sigma2 = np.zeros(self.T) #stroing our variences
            sigma2[0] = omega / (1 - alpha - beta)
            
            for t in range(1, self.T):
                sigma2[t] = omega + alpha * self.returns.iloc[t-1]**2 + beta * sigma2[t-1] #GARCH formula
                if sigma2[t] <= 0:
                    return 1e6 
            
            #weighted log-likelihood
            log_lik = 0  #different name 
            for t in range(self.T):
                log_lik += state_probs[t] * (
                    -0.5 * np.log(2 * np.pi * sigma2[t]) - 
                    0.5 * self.returns.iloc[t]**2 / sigma2[t]
                )
            
            return -log_lik
        
        current_params = self.garch_params[state]
        x0 = [current_params['omega'], current_params['alpha'], current_params['beta']]
        
        bounds = [(1e-6, None), (1e-6, 0.5), (1e-6, 0.95)] #sensitvity settings
        constraints = {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]} #probility constraint (noramlized)
        #for optimizer
        
        result = minimize(neg_log_likelihood, x0, method='L-BFGS-B', 
                            bounds=bounds, constraints=constraints) #minimization function all imported (should change)
            
        if result.success:
            omega, alpha, beta = result.x #result.'x' syntax , dont change order, 
            self.garch_params[state] = {
                    'omega': omega, 'alpha': alpha, 'beta': beta
                }
        
    
    def _viterbi_decode(self):
        #viterbi algorithm for most likely sequence of states
        delta = np.zeros((self.T, self.n_states)) #max prob of state x at time t
        psi = np.zeros((self.T, self.n_states), dtype=int) #previous state that would lead to current state.
        
        #initialize
        for state in range(self.n_states):
            delta[0, state] = np.log(self.pi[state]) + np.log(self._garch_emission_prob(0, state))#prob of startig in state then prob of obvsering state
        #forward pass - path finding
        for t in range(1, self.T):
            for j in range(self.n_states):
                trans_probs = delta[t-1] + np.log(self.A[:, j] + 1e-300)
                psi[t, j] = np.argmax(trans_probs) #path the has the best 'score'
                delta[t, j] = np.max(trans_probs) + np.log(self._garch_emission_prob(t, j))#final fig
        
        #backward pass - reconstruct best path
        states = np.zeros(self.T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(self.T-2, -1, -1):
            states[t] = psi[t+1, states[t+1]]
        
        return states
    
    def _calculate_shock_metrics(self):
        
        self.preshock_prob = self.state_probabilities[:, 2]  #pre-shock state
        self.shock_prob = self.state_probabilities[:, 3]     #shock state
        
        #risk score combining pre-shock and shock probabilities
        self.risk_score = 0.6 * self.preshock_prob + 0.4 * self.shock_prob
        
        
    
    def get_current_status(self):
        
        current_state = self.states[-1]
        days_in_state = 1
        for i in range(len(self.states) - 2, -1, -1):
            if self.states[i] == current_state:
                days_in_state += 1
            else:
                break
        #final metrics :)
        return {
            'risk_score': self.risk_score[-1],
            'current_state': self.state_labels[current_state],
            'preshock_probability': self.preshock_prob[-1],
            'shock_probability': self.shock_prob[-1],
            'days_in_current_state': days_in_state,
            'garch_params': self.garch_params[current_state]
        }
    
    def predict_volatility(self, horizon=5):
        #predicting volatility using current state GARCH params
        if not self.fitted:
            raise ValueError("Model must be fitted first")
        
        current_state = self.states[-1]
        params = self.garch_params[current_state]
        current_variance = self.conditional_variances[-1, current_state]
        last_return = self.returns.iloc[-1]
        
        forecasts = []
        for h in range(1, horizon + 1):
            if h == 1:
                forecast_var = (params['omega'] + 
                              params['alpha'] * last_return**2 + 
                              params['beta'] * current_variance)
            else:
                #multi-step forecast
                unconditional_var = params['omega'] / (1 - params['alpha'] - params['beta'])
                persistence = params['alpha'] + params['beta']
                forecast_var = (unconditional_var + 
                              (forecasts[0]**2 - unconditional_var) * persistence**(h-1))
            
            forecasts.append(np.sqrt(max(forecast_var, 1e-8)))
        
        return np.array(forecasts)

#standard yahoo finance data , or any time series data in the correct form
def load_oil_data(commodity='CL', period='2y'):
    ticker = yf.Ticker(f"{commodity}=F")
    data = ticker.history(period=period)
    data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
    return data.dropna()

def plot_results(model, oil_data):
    fig, axes = plt.subplots(2, 1, figsize=(15, 10)) 
    dates = oil_data.index
    colors = ['green', 'blue', 'orange', 'red']
    
    #plot 1 - prices with regimes labelled - using state probabilities to color points
    close_prices = oil_data['Close'].values
    
    for i in range(len(dates)):
        #get the state probabilities for this time point
        state_probs = model.state_probabilities[i]
        #find the most likely state but also consider if pre-shock or shock probs are high
        if state_probs[3] > 0.3 or state_probs[2] > 0.5:  #shock or pre-shock
            if state_probs[3] > state_probs[2]:
                color = colors[3]  #red for shock
            else:
                color = colors[2]  #orange for pre-shock
        elif state_probs[2] > 0.3:  #moderate pre-shock
            color = colors[2]  #orange
        elif state_probs[1] > state_probs[0]:  #normal vs low vol
            color = colors[1]  #blue for normal
        else:
            color = colors[0]  #green for low vol
            
        axes[0].scatter(dates[i], close_prices[i], c=color, alpha=0.7, s=15)
    
    #add legend
    for state in range(model.n_states):
        axes[0].scatter([], [], c=colors[state], label=f'{model.state_labels[state]}', s=50)
    
    axes[0].set_title('Oil Price states via GARCH-HMM')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    #plot 2: state probabilities (moved from plot 3)
    for state in [2, 3]:  #pre-shock and shock states
        axes[1].plot(dates, model.state_probabilities[:, state], 
                    color=colors[state], linewidth=2, alpha=0.8,
                    label=f'{model.state_labels[state]}')
    
    axes[1].set_title('Shock State Probabilities')
    axes[1].set_ylabel('Probability')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    #print current status
    status = model.get_current_status()
    print(f"Current State: {status['current_state']}")
    print(f"Days in State: {status['days_in_current_state']}")

def main():
    
    print("Loading :)   ...")
    oil_data = load_oil_data('CL', '2y')
    model = GarchHmmOilModel(n_states=4)
    model.fit(oil_data['Log_Returns'])
    plot_results(model, oil_data)
    vol_forecast = model.predict_volatility(5)


if __name__ == "__main__":
    main()