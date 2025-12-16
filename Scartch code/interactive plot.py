import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotly.io import show

from skfolio import PerfMeasure, RatioMeasure, RiskMeasure
from skfolio.optimization import MeanRisk
from skfolio.prior import EmpiricalPrior
from skfolio.moments import LedoitWolf

# -------------------------
# Load CSV and prepare data
# -------------------------
df = pd.read_csv("FX_excess_returns.csv")
df["Time Period"] = pd.to_datetime(df["Time Period"])
df = df.set_index("Time Period")
returns = df.astype(float)

WINDOW = 60
n_assets = returns.shape[1]
assets = returns.columns
rf = pd.Series(0.0, index=returns.index)

# -------------------------
# Out-of-sample storage
# -------------------------
oos_returns = {k: [] for k in ["Markowitz", "EqualWeight", "Shrinkage", "Constrained", "Resampled"]}
oos_dates = []

# -------------------------
# Population for frontier plotting
# -------------------------
population_all = None

# -------------------------
# Rolling-window loop
# -------------------------
for t in range(WINDOW, len(returns) - 1):
    train = returns.iloc[t-WINDOW:t]
    test = returns.iloc[t+1]
    oos_dates.append(returns.index[t+1])

    # ---- MARKOWITZ ----
    mv = MeanRisk(risk_measure=RiskMeasure.VARIANCE, efficient_frontier_size=30)
    mv.fit(train.values)
    mv_pop = mv.predict(train.values)
    w_mv = mv_pop[0].weights  # first portfolio along frontier
    oos_returns["Markowitz"].append(float(np.dot(w_mv, test.values)))

    # Add to Population
    mv_pop.set_portfolio_params(tag=f"Markowitz Window {t}")
    population_all = mv_pop if population_all is None else population_all + mv_pop

    # ---- EQUAL WEIGHT ----
    w_eq = np.ones(n_assets) / n_assets
    oos_returns["EqualWeight"].append(float(np.dot(w_eq, test.values)))

    # ---- SHRINKAGE (LEDOIT-WOLF) ----
    mv_shrink = MeanRisk(
        risk_measure=RiskMeasure.VARIANCE,
        efficient_frontier_size=30,
        prior_estimator=EmpiricalPrior(covariance_estimator=LedoitWolf())
    )
    mv_shrink.fit(train.values)
    shrink_pop = mv_shrink.predict(train.values)
    w_shrink = shrink_pop[0].weights
    oos_returns["Shrinkage"].append(float(np.dot(w_shrink, test.values)))

    # Add to Population
    shrink_pop.set_portfolio_params(tag=f"Shrinkage Window {t}")
    population_all += shrink_pop

    # ---- CONSTRAINED (0 <= w <= 0.2) ----
    mv_const = MeanRisk(risk_measure=RiskMeasure.VARIANCE, min_weights=0.0, max_weights=0.2, efficient_frontier_size=30)
    mv_const.fit(train.values)
    const_pop = mv_const.predict(train.values)
    w_const = const_pop[0].weights
    oos_returns["Constrained"].append(float(np.dot(w_const, test.values)))

    # ---- RESAMPLED / BOOTSTRAPPED ----
    B = 60
    weights_boot = np.zeros((B, n_assets))
    for b in range(B):
        idx = np.random.choice(WINDOW, size=WINDOW, replace=True)
        boot_sample = train.iloc[idx]
        mv_boot = MeanRisk(risk_measure=RiskMeasure.VARIANCE, efficient_frontier_size=2)
        mv_boot.fit(boot_sample.values)
        w_boot = mv_boot.predict(boot_sample.values)[0].weights
        weights_boot[b, :] = w_boot
    w_resampled = weights_boot.mean(axis=0)
    oos_returns["Resampled"].append(float(np.dot(w_resampled, test.values)))

# -------------------------
# Convert OOS returns to DataFrame
# -------------------------
oos_df = pd.DataFrame(oos_returns, index=oos_dates)
print(oos_df.head())

# -------------------------
# Compute Out-of-Sample Sharpe Ratios
# -------------------------
sharpe_table = (oos_df.mean() / oos_df.std()).to_frame("Out-of-Sample Sharpe Ratio") \
    .sort_values(by="Out-of-Sample Sharpe Ratio", ascending=False)
print(sharpe_table)

# -------------------------
# Plot Cumulative Returns
# -------------------------
cum_returns = (1 + oos_df).cumprod()
plt.figure(figsize=(10, 6))
for col in cum_returns.columns:
    plt.plot(cum_returns.index, cum_returns[col], label=col)
plt.legend()
plt.title("Cumulative Out-of-Sample Portfolio Performance")
plt.xlabel("Time")
plt.ylabel("Cumulative Wealth")
plt.tight_layout()
plt.show()

# -------------------------
# Plot Weight Dispersion
# -------------------------
weight_dispersion = pd.DataFrame({
    "Markowitz": np.std(w_mv),
    "Resampled": np.std(w_resampled)
}, index=["Weight Dispersion"]).T

weight_dispersion.plot(kind="bar", legend=False, title="Portfolio Weight Dispersion")
plt.ylabel("Standard Deviation of Weights")
plt.tight_layout()
plt.show()

# -------------------------
# Interactive In-Sample Frontier
# -------------------------
fig = population_all.plot_measures(
    x=RiskMeasure.ANNUALIZED_STANDARD_DEVIATION,
    y=PerfMeasure.ANNUALIZED_MEAN,
    color_scale=RatioMeasure.ANNUALIZED_SHARPE_RATIO,
    hover_measures=[RiskMeasure.MAX_DRAWDOWN, RatioMeasure.ANNUALIZED_SORTINO_RATIO],
)
show(fig)