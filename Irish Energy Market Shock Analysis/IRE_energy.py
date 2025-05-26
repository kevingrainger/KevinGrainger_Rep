import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from datetime import datetime
from numpy.linalg import eigh

# ---------------------
# Load Irish gas price data
# ---------------------
# Replace with your actual data paths or APIs
# Data should include a column 'Price' and 'Date'
data = pd.read_csv('irish_gas_prices.csv', parse_dates=['Date'])
data = data.set_index('Date')
prices = data['Price']

# ---------------------
# Detect regime change (Feb 24, 2022)
# ---------------------
invasion_date = pd.Timestamp('2022-02-24')
pre_war = prices[prices.index < invasion_date]
post_war = prices[prices.index >= invasion_date]

# ---------------------
# Normalize and analyze volatility
# ---------------------
def normalized_returns(series):
    log_ret = np.diff(np.log(series))
    return (log_ret - np.mean(log_ret)) / np.std(log_ret)

r_pre = normalized_returns(pre_war)
r_post = normalized_returns(post_war)

# ---------------------
# Plot volatility clustering
# ---------------------
plt.figure(figsize=(12,4))
plt.plot(pre_war.index[1:], r_pre, color='gray', label='Pre-Invasion')
plt.plot(post_war.index[1:], r_post, color='darkred', label='Post-Invasion')
plt.axvline(invasion_date, linestyle='--', color='black', label='Invasion Day')
plt.title("Normalized Log Returns of Irish Gas Prices")
plt.ylabel("Normalized Returns")
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.show()

# ---------------------
# Bornholdt-style sentiment dynamics
# ---------------------
N = 32
J = 1.0
beta = 1.0 / 1.5
alpha = 3.0
S = np.random.choice([-1, 1], size=(N, N))
C = np.random.choice([-1, 1], size=(N, N))

def neighbors(i, j, N):
    z = []
    if i > 0: z.append((i-1,j))
    if i < N-1: z.append((i+1,j))
    if j > 0: z.append((i,j-1))
    if j < N-1: z.append((i,j+1))
    return z

steps = 100
M_values = []

for _ in range(steps):
    M = np.mean(S)
    Sn = np.zeros_like(S)
    Cn = np.zeros_like(C)

    for i in range(N):
        for j in range(N):
            h = sum(S[x][y] for x,y in neighbors(i,j,N))
            h = J * h - alpha * C[i,j] * M
            p = 1.0 / (1 + np.exp(-2 * beta * h))
            Sn[i,j] = 1 if np.random.rand() < p else -1
            Cn[i,j] = -C[i,j] if alpha * S[i,j] * C[i,j] * h < 0 else C[i,j]

    S = Sn
    C = Cn
    M_values.append(M)

plt.figure()
plt.plot(M_values, color='darkgreen')
plt.title("Modeled Market Sentiment Over Time")
plt.xlabel("Time")
plt.ylabel("Average Spin (Sentiment)")
plt.grid(True)
plt.tight_layout()
plt.show()

# ---------------------
# Marcenko-Pastur-based correlation analysis
# ---------------------
# Simulated example with Irish, UK, EU gas prices
# Replace with real correlation matrices from data
np.random.seed(0)
n_assets = 10
n_days = 250

# Pre-war simulated returns
R_pre = np.random.normal(0, 1, size=(n_days, n_assets))
C_pre = np.corrcoef(R_pre, rowvar=False)

# Post-war simulated returns
R_post = np.random.normal(0, 1.5, size=(n_days, n_assets))  # Increased volatility
C_post = np.corrcoef(R_post, rowvar=False)

# Eigenvalue distribution
eigs_pre = eigh(C_pre)[0]
eigs_post = eigh(C_post)[0]

plt.figure()
plt.hist(eigs_pre, bins=20, alpha=0.6, label='Pre-Invasion', color='skyblue')
plt.hist(eigs_post, bins=20, alpha=0.6, label='Post-Invasion', color='salmon')
plt.title("Eigenvalue Spectrum of Energy Price Correlations")
plt.xlabel("Eigenvalue")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.show()
