import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

stocks = ["THYAO.IS", "ASELS.IS", "TUPRS.IS", "BIMAS.IS", "GARAN.IS"]

data = yf.download(
    stocks,
    start="2020-01-01",
    end="2024-12-31",
    auto_adjust=True
)["Close"]

returns = data.pct_change().dropna()

print("\n--- ANALİZ KONTROL ---")
print("Fiyat veri boyutu:", data.shape)
print("Getiri veri boyutu:", returns.shape)
print("Analiz başlangıç:", returns.index.min())
print("Analiz bitiş:", returns.index.max())
print("----------------------\n")

print("Fiyat verileri:")
print(data.head())

print("\nGünlük getiriler:")
print(returns.head())

corr_matrix = returns.corr()

plt.imshow(corr_matrix)
plt.colorbar()

plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=45)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)

plt.title("Korelasyon Heatmap")

plt.show()

num_portfolios = 10000
results = np.zeros((3, num_portfolios))
weights_record = []

for i in range(num_portfolios):
    weights = np.random.random(len(stocks))
    weights /= np.sum(weights)
    weights_record.append(weights)

    portfolio_return = np.sum(returns.mean() * weights) * 252
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))
    sharpe_ratio = portfolio_return / portfolio_risk

    results[0,i] = portfolio_return
    results[1,i] = portfolio_risk
    results[2,i] = sharpe_ratio

plt.scatter(results[1,:], results[0,:], c=results[2,:])
plt.xlabel("Risk")
plt.ylabel("Return")
plt.title("Efficient Frontier")
plt.colorbar(label="Sharpe Ratio")
plt.show()

max_sharpe_idx = np.argmax(results[2])
optimal_weights = weights_record[max_sharpe_idx]

optimal_portfolio = pd.DataFrame({
    "Hisse": stocks,
    "Ağırlık": optimal_weights
})

optimal_portfolio["Ağırlık (%)"] = optimal_portfolio["Ağırlık"] * 100

print(optimal_portfolio)

optimal_portfolio.set_index("Hisse")["Ağırlık (%)"].plot(kind="bar")
plt.title("Optimal Portföy Dağılımı")
plt.ylabel("Ağırlık (%)")
plt.show()
