from skfolio.optimization import MeanRisk
from skfolio.moments import EmpiricalCovariance, LedoitWolf
from skfolio import RiskMeasure
from skfolio.prior import EmpiricalPrior
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


fx_raw = pd.read_csv("FX_rates.csv")

currency_row = fx_raw.iloc[0, 1:].values
fx = fx_raw.iloc[1:].copy()
fx["Time Period"] = pd.PeriodIndex(fx["Time Period"], freq="M")
fx = fx.set_index("Time Period")
fx = fx.apply(pd.to_numeric, errors="coerce")


fx_adj = fx.copy()
for col, curr in zip(fx.columns, currency_row):
    if curr != "USD":
        fx_adj[col] = 1.0 / fx_adj[col]

delta_s = np.log(fx_adj).diff().dropna(how="all").round(6)

rf_raw = pd.read_csv("Risk Free Rates.csv")
currency_to_country = {
    "USD": "USD",
    "AUD": "AUSTRALIA", "CAD": "CANADA", "EUR": "EURO AREA",
    "BRL": "BRAZIL", "CHF": "SWITZERLAND", "DKK": "DENMARK",
    "GBP": "UNITED KINGDOM", "JPY": "JAPAN", "KRW": "KOREA",
    "ZAR": "SOUTH AFRICA", "TWD": "TAIWAN", "SGD": "SINGAPORE",
    "NZD": "NEW ZEALAND", "MXN": "MEXICO", "NOK": "NORWAY",
    "INR": "INDIA", "CNY": "CHINA", "SEK": "SWEDEN","HUF":"HUNGARY",
    "CZK":"CZECHIA","TRY":"TURKEY"
}
rf = rf_raw.rename(columns=currency_to_country)

fx_countries = delta_s.columns
for country in fx_countries:
    if country not in rf.columns:
        rf[country] = np.nan

rf["Time Period"] = pd.to_datetime(rf["Time Period"], format="%Y%m%d", errors='coerce')
rf = rf.set_index("Time Period")
rf = rf.apply(pd.to_numeric, errors="coerce")

rf_monthly = rf.resample('ME').mean().round(6)
rf_monthly.index = rf_monthly.index.to_period("M")


common_index = delta_s.index.intersection(rf_monthly.index)
delta_s = delta_s.loc[common_index]
rf_monthly = rf_monthly.loc[common_index]

usd_rf = rf_monthly["USD"]

excess_returns = pd.DataFrame(index=common_index, columns=delta_s.columns, dtype=float)

for col in delta_s.columns:
    if col not in rf_monthly.columns:
        continue

    mask = delta_s[col].notna() & rf_monthly[col].notna() & usd_rf.notna()
    excess_returns.loc[mask, col] = rf_monthly.loc[mask, col] - usd_rf.loc[mask] - delta_s.loc[mask, col]

excess_returns.round(6).drop(["INDIA", "CHINA", "KOREA"], axis=1).to_csv("FX_excess_returns.csv", na_rep="NaN")



df = pd.read_csv("FX_excess_returns.csv")

df["Time Period"] = pd.to_datetime(df["Time Period"])
df = df.set_index("Time Period")

returns = df.astype(float)


WINDOW = 60  # rolling window length (5 years)
n_assets = returns.shape[1]
assets = returns.columns

rf = pd.Series(0.0, index=returns.index)

oos_returns = {
    "Markowitz": [],
    "EqualWeight": [],
    "Shrinkage": [],
    "Constrained": [],
    "Resampled": []
}

oos_dates = []

population_all = None


mu_hat = returns.mean().values
Sigma_hat = EmpiricalCovariance().fit(returns.values).covariance_

inv_Sigma = np.linalg.inv(Sigma_hat)
ones = np.ones(n_assets)

w_is = inv_Sigma @ mu_hat / (ones @ inv_Sigma @ mu_hat)

benchmark_weights = pd.Series(w_is, index=assets)
# print(benchmark_weights)



threshold = 1e-4
round_off = 8

oos_weights = {
    "Markowitz": [],
    "EqualWeight": [],
    "Shrinkage": [],
    "Constrained": [],
    "Resampled": []
}

for t in range(WINDOW, len(returns) - 1):

    train = returns.iloc[t-WINDOW:t]
    test = returns.iloc[t+1]
    oos_dates.append(returns.index[t+1])

    # ---------------- Markowitz ----------------
    mv = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
    mv.fit(train.values)
    w_mv = mv.weights_
    w_mv[np.abs(w_mv) < threshold] = 0       
    w_mv = np.round(w_mv, round_off)                  
    oos_returns["Markowitz"].append(np.dot(w_mv, test.values))
    oos_weights["Markowitz"].append(w_mv)

    # ---------------- Equal Weight ----------------
    w_eq = np.ones(n_assets) / n_assets
    w_eq[np.abs(w_eq) < threshold] = 0
    w_eq = np.round(w_eq, round_off)
    oos_returns["EqualWeight"].append(np.dot(w_eq, test.values))
    oos_weights["EqualWeight"].append(w_eq)

    # ---------------- Shrinkage ----------------
    mv_shrink = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        prior_estimator=EmpiricalPrior(covariance_estimator=LedoitWolf())
    )
    mv_shrink.fit(train.values)
    w_shrink = mv_shrink.weights_
    w_shrink[np.abs(w_shrink) < threshold] = 0
    w_shrink = np.round(w_shrink, round_off)
    oos_returns["Shrinkage"].append(np.dot(w_shrink, test.values))
    oos_weights["Shrinkage"].append(w_shrink)

    # ---------------- Constrained ----------------
    mv_const = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        min_weights=0.0,
        max_weights=0.20
    )
    mv_const.fit(train.values)
    w_const = mv_const.weights_
    w_const[np.abs(w_const) < threshold] = 0
    w_const = np.round(w_const, round_off)
    oos_returns["Constrained"].append(np.dot(w_const, test.values))
    oos_weights["Constrained"].append(w_const)

    # ---------------- Resampled ----------------
    B = 60
    weights_boot = np.zeros((B, n_assets))
    for b in range(B):
        idx = np.random.choice(WINDOW, size=WINDOW, replace=True)
        boot_sample = train.iloc[idx]
        mv_boot = MeanRisk(risk_measure=RiskMeasure.VARIANCE)
        mv_boot.fit(boot_sample.values)
        weights_boot[b, :] = mv_boot.weights_
    w_resampled = weights_boot.mean(axis=0)
    w_resampled[np.abs(w_resampled) < threshold] = 0
    w_resampled = np.round(w_resampled, round_off)
    oos_returns["Resampled"].append(np.dot(w_resampled, test.values))
    oos_weights["Resampled"].append(w_resampled)

weights_dfs = {strategy: pd.DataFrame(oos_weights[strategy], index=oos_dates, columns=assets)
               for strategy in oos_weights.keys()}


for strategy, df_w in weights_dfs.items():
    df_w.to_csv(f"{strategy}_weights.csv")


oos_df = pd.DataFrame(oos_returns, index=oos_dates)
oos_df.to_csv("oos_returns.csv", index=True)

sharpe_results = {}

for col in oos_df.columns:
    excess = oos_df[col] - rf.loc[oos_df.index]
    sharpe_results[col] = excess.mean() / excess.std()

sharpe_table = (
    oos_df.mean() / oos_df.std()
).to_frame("Out-of-Sample Sharpe Ratio") \
 .sort_values(by="Out-of-Sample Sharpe Ratio", ascending=False)

sharpe_table

cum_returns = (1 + oos_df).cumprod()

plt.figure(figsize=(10, 6))
for col in cum_returns.columns:
    plt.plot(cum_returns.index, cum_returns[col], label=col)

plt.legend()
plt.title("Cumulative Out-of-Sample Portfolio Performance")
plt.xlabel("Time")
plt.ylabel("Cumulative Wealth")
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
y_max = int(cum_returns.max().max()) + 1  # get max across all columns
plt.yticks(range(0, y_max + 1, 5))
plt.tight_layout()
plt.grid(True,linestyle='--',linewidth=0.75)
plt.show()


for strategy, weights_df in weights_dfs.items():
    df = weights_df.copy()
    
    df['Year'] = df.index.year
    annual_weights = df.groupby('Year').mean()
    annual_weights = annual_weights.drop(columns=['Year'], errors='ignore')

    annual_weights_pct = annual_weights * 100
    # annual_weights_sorted = pd.DataFrame(
    #     annual_weights_pct.apply(lambda row: row.sort_values().values, axis=1).tolist(),
    #     index=annual_weights_pct.index,
    #     columns=annual_weights_pct.columns[np.argsort(annual_weights_pct.mean(axis=0))]
    # )

    ax = annual_weights_pct.plot(
        kind='bar',
        stacked=True,
        figsize=(12, 6),
        colormap='tab20c',
        zorder=2
    )
    ax.set_yticks(np.arange(0, 101, 10))
    ax.grid(axis='y', linestyle='--', linewidth=0.75, zorder=1)
    
    plt.ylabel("Portfolio Weights (%)")
    plt.xlabel("Year")
    plt.title(f"Annual Portfolio Weights - {strategy}")
    plt.legend(title="Country", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    oos_df = pd.DataFrame(oos_returns, index=pd.to_datetime(oos_dates).to_period('M'))

start_period = oos_df.index.min()
rf_usd_aligned = rf_monthly["USD"].loc[start_period:]

if not isinstance(rf_usd_aligned.index, pd.PeriodIndex):
    rf_usd_aligned.index = rf_usd_aligned.index.to_period('M')

rf_usd_aligned = rf_usd_aligned.reindex(oos_df.index).ffill()
excess_returns = oos_df.subtract(rf_usd_aligned, axis=0)

rolling_window = 36
rolling_sharpe = excess_returns.rolling(window=rolling_window).mean() / \
                 excess_returns.rolling(window=rolling_window).std()
rolling_sharpe = rolling_sharpe.dropna(how='all')

plt.figure(figsize=(12,6))
for col in rolling_sharpe.columns:
    # Convert PeriodIndex to datetime for matplotlib
    plt.plot(rolling_sharpe.index.to_timestamp(), rolling_sharpe[col], label=col)

plt.axhline(0, color='black', linestyle='--', linewidth=0.75)
plt.title(f"{rolling_window}-Month Rolling Sharpe Ratios")
plt.ylabel("Rolling Sharpe Ratio")
plt.xlabel("Date")
plt.grid(True, linestyle='--', linewidth=0.75)
plt.legend(title="Strategy")
plt.tight_layout()
plt.show()
