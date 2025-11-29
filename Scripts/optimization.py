import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load per-customer probabilistic CLV from BG/NBD
clv_df = pd.read_csv("Data/processed/customer_clv_bgnbd.csv")

# For now, drop customers with missing CLV
clv_df = clv_df.dropna(subset=["pred_clv"]).reset_index(drop=True)

# Base CLV vector
clv = clv_df["pred_clv"].values
n = len(clv)

# Total marketing budget (you can tune this or read from config)
BUDGET = 100000.0  # dollars

# Response model parameters: controls max lift and speed of saturation
A = 0.3   # max 30% uplift in CLV at very high spend
B = 0.0001  # how fast returns saturate


def total_clv_with_spend(x):
    """
    Given a vector of per-customer spend x (length n),
    return total expected CLV using a concave lift function.

    clv_i(x_i) = clv_i * (1 + A * (1 - exp(-B * x_i)))
    """
    lift = 1.0 + A * (1.0 - np.exp(-B * x))
    return np.sum(clv * lift)


def objective(x):
    # We want to MAXIMIZE total_clv_with_spend, but scipy.minimize minimizes,
    # so return the negative.
    return -total_clv_with_spend(x)


def budget_constraint(x):
    # Total spend must be <= BUDGET → BUDGET - sum(x) >= 0
    return BUDGET - np.sum(x)


# Bounds: each customer's spend >= 0
bounds = [(0.0, None)] * n

# Constraint for SLSQP: inequality type, budget_constraint(x) >= 0
constraints = [{"type": "ineq", "fun": budget_constraint}]

# Start from uniform allocation
x0 = np.full(n, BUDGET / n)

# Run optimization
result = minimize(
    objective,
    x0,
    method="SLSQP",
    bounds=bounds,
    constraints=constraints,
    options={"maxiter": 500}
)

if not result.success:
    print("Optimization did not fully converge:", result.message)

optimal_spend = result.x

# Compute baseline (uniform) CLV and optimized CLV
lift_uniform = 1.0 + A * (1.0 - np.exp(-B * x0))
clv_uniform_total = np.sum(clv * lift_uniform)

lift_opt = 1.0 + A * (1.0 - np.exp(-B * optimal_spend))
clv_opt_total = np.sum(clv * lift_opt)

delta = clv_opt_total - clv_uniform_total
percent_improvement = delta / clv_uniform_total * 100.0

print(f"Budget B = ${BUDGET:,.0f}")
print(f"Total CLV (uniform spend): ${clv_uniform_total:,.2f}")
print(f"Total CLV (optimized spend): ${clv_opt_total:,.2f}")
print(f"Incremental CLV (optimized - uniform): ${delta:,.2f}")
print(f"Relative improvement over uniform: {percent_improvement:.2f}%")

# Attach optimal spend back to customers and save
clv_df["optimal_spend"] = optimal_spend
clv_df.to_csv("Data/processed/customer_clv_optimized.csv", index=False)
print("Saved optimized allocations to Data/processed/customer_clv_optimized.csv")

# Nice one-line conclusion string you can reuse in README / notebook
conclusion = (
    f"By implementing a probabilistic BG/NBD-based CLV model and a "
    f"budget-constrained optimization strategy, we can re-allocate the existing "
    f"${BUDGET:,.0f} budget to increase total customer value by an estimated "
    f"${delta:,.0f}, a {percent_improvement:.1f}% improvement over a uniform "
    f"spending strategy."
)
print("\n", conclusion)
