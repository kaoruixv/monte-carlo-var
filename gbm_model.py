import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# 1. Fetch Data
print("Fetching SPY data...")
data = yf.download('SPY', start='2021-01-01', end='2026-07-10')['Close']
returns = data.pct_change().dropna()

# 2. Extract Math Parameters (Forcing pure float scalars to bypass Pandas)
mu = float(np.squeeze(returns.to_numpy().mean()))
sigma = float(np.squeeze(returns.to_numpy().std()))
drift = mu - (0.5 * sigma**2)

# 3. Setup Simulation
days = 30
simulations = 10000
last_price = float(np.squeeze(data.to_numpy()[-1]))

# 4. Generate Random Shocks (GBM)
print("Running 10,000 simulations...")
Z = np.random.normal(0, 1, (days, simulations))
daily_returns = np.exp(drift + sigma * Z)

# 5. Compound Prices
price_paths = np.zeros_like(daily_returns)
price_paths[0] = last_price
for t in range(1, days):
    price_paths[t] = price_paths[t-1] * daily_returns[t]

# 6. Calculate Value at Risk (VaR)
final_prices = price_paths[-1]
worst_case_price = np.percentile(final_prices, 1)
var_99 = last_price - worst_case_price

print("-" * 30)
print(f"Current SPY Price: ${last_price:.2f}")
print(f"99% Worst Case Price in 30 Days: ${worst_case_price:.2f}")
print(f"Maximum Expected Loss (VaR): ${var_99:.2f} per share")
print("-" * 30)

# 7. Visualize and Save
print("Generating probability chart...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [3, 1]})

ax1.plot(price_paths, color='royalblue', alpha=0.02)
ax1.axhline(last_price, color='black', linewidth=1.5, label=f'Current Price (${last_price:.2f})')
ax1.axhline(worst_case_price, color='crimson', linewidth=2, linestyle='--', label=f'99% VaR (${worst_case_price:.2f})')
ax1.set_title('Monte Carlo Simulation: 30-Day SPY Futures', fontweight='bold')
ax1.set_xlabel('Trading Days')
ax1.set_ylabel('Simulated Price ($)')
ax1.set_xlim(0, days - 1)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax2.hist(final_prices, bins=50, color='royalblue', alpha=0.7, orientation='horizontal')
ax2.axhline(last_price, color='black', linewidth=1.5)
ax2.axhline(worst_case_price, color='crimson', linewidth=2, linestyle='--')
ax2.set_title('Final Day Distribution', fontweight='bold')
ax2.set_xlabel('Frequency')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('monte_carlo_var.png', dpi=300, bbox_inches='tight')
print("Saved 'monte_carlo_var.png' to your folder.")