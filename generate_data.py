"""
generate_data.py
Generates a realistic synthetic e-commerce order-level dataset with
intentional, findable patterns (category margins, customer purchase
frequency tiers, seasonality) so downstream EDA / RFM analysis has
real signal instead of pure noise.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------------------------------------------------------
# 1. Reference data
# ---------------------------------------------------------------

CATEGORIES = {
    # category: (cost_low, cost_high, margin_low, margin_high, weight)
    "Beauty":         (150,  1200, 0.38, 0.50, 0.12),
    "Apparel":        (200,  2000, 0.32, 0.46, 0.20),
    "Sports":         (300,  3500, 0.20, 0.30, 0.10),
    "Home & Kitchen": (250,  4000, 0.22, 0.32, 0.16),
    "Electronics":    (800, 25000, 0.08, 0.18, 0.22),
    "Groceries":      (50,   800,  0.06, 0.14, 0.20),
}
CAT_NAMES = list(CATEGORIES.keys())
CAT_WEIGHTS = [CATEGORIES[c][4] for c in CAT_NAMES]

CITIES = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Pune",
          "Chennai", "Kolkata", "Ahmedabad", "Jaipur", "Lucknow"]
CITY_WEIGHTS = [0.16, 0.15, 0.14, 0.09, 0.09, 0.09, 0.08, 0.07, 0.07, 0.06]

START_DATE = datetime(2024, 1, 1)
END_DATE = datetime(2025, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days

N_CUSTOMERS = 2000

# ---------------------------------------------------------------
# 2. Build customer purchase-frequency tiers (creates real RFM signal)
# ---------------------------------------------------------------
# one-time buyers, occasional/regular buyers, power/repeat buyers
tier_choice = np.random.choice(
    ["one_time", "occasional", "power"],
    size=N_CUSTOMERS,
    p=[0.35, 0.45, 0.20]
)

customer_ids = [f"CUST{str(i).zfill(5)}" for i in range(1, N_CUSTOMERS + 1)]
customer_city = np.random.choice(CITIES, size=N_CUSTOMERS, p=CITY_WEIGHTS)

# number of orders per customer, and recency behaviour per tier
n_orders_per_cust = []
recency_bias = []  # 'active' customers order more recently, 'churned' skew earlier
for t in tier_choice:
    if t == "one_time":
        n_orders_per_cust.append(1)
        recency_bias.append(np.random.choice(["recent", "old"], p=[0.45, 0.55]))
    elif t == "occasional":
        n_orders_per_cust.append(np.random.randint(3, 12))
        recency_bias.append(np.random.choice(["recent", "old"], p=[0.55, 0.45]))
    else:  # power
        n_orders_per_cust.append(np.random.randint(12, 30))
        recency_bias.append("recent")  # power users stay active

customers_df = pd.DataFrame({
    "customer_id": customer_ids,
    "customer_city": customer_city,
    "tier": tier_choice,
    "n_orders": n_orders_per_cust,
    "recency_bias": recency_bias,
})

# ---------------------------------------------------------------
# 3. Generate order-level rows
# ---------------------------------------------------------------
rows = []
order_counter = 1

# mild monthly seasonality multiplier (e.g., Oct-Nov festive/holiday bump)
def month_seasonality(month):
    bump_months = {10: 1.35, 11: 1.45, 12: 1.20, 1: 0.85, 2: 0.90}
    return bump_months.get(month, 1.0)

for _, cust in customers_df.iterrows():
    n = cust["n_orders"]
    for _ in range(n):
        # recency bias: 'recent' customers order mostly in last 6 months,
        # 'old'/churned customers order mostly in the first 12 months
        if cust["recency_bias"] == "recent":
            day_offset = int(np.clip(np.random.normal(TOTAL_DAYS * 0.78, TOTAL_DAYS * 0.18), 0, TOTAL_DAYS))
        else:
            day_offset = int(np.clip(np.random.normal(TOTAL_DAYS * 0.30, TOTAL_DAYS * 0.20), 0, TOTAL_DAYS))

        order_date = START_DATE + timedelta(days=day_offset)
        season_mult = month_seasonality(order_date.month)

        # skip probability inverse to seasonality to bake seasonality into volume too
        if np.random.rand() > min(season_mult / 1.5, 1.0) * 0.98 + 0.02:
            pass  # keep order anyway; seasonality mainly affects density via day_offset draws

        category = np.random.choice(CAT_NAMES, p=CAT_WEIGHTS)
        cost_low, cost_high, margin_low, margin_high, _ = CATEGORIES[category]

        cost_price = round(np.random.uniform(cost_low, cost_high), 2)
        margin_pct = np.random.uniform(margin_low, margin_high)
        selling_price = round(cost_price / (1 - margin_pct), 2)

        quantity = np.random.choice([1, 1, 1, 2, 2, 3, 4], p=[0.35, 0.2, 0.15, 0.15, 0.08, 0.04, 0.03])

        rows.append({
            "order_id": f"ORD{str(order_counter).zfill(6)}",
            "customer_id": cust["customer_id"],
            "order_date": order_date.strftime("%Y-%m-%d"),
            "product_category": category,
            "cost_price": cost_price,
            "selling_price": selling_price,
            "quantity": int(quantity),
            "customer_city": cust["customer_city"],
        })
        order_counter += 1

df = pd.DataFrame(rows)

# ---------------------------------------------------------------
# 4. Derived columns
# ---------------------------------------------------------------
df["total_cost"] = (df["cost_price"] * df["quantity"]).round(2)
df["total_revenue"] = (df["selling_price"] * df["quantity"]).round(2)
df["profit"] = (df["total_revenue"] - df["total_cost"]).round(2)
df["profit_margin_pct"] = (df["profit"] / df["total_revenue"] * 100).round(2)

df = df.sort_values("order_date").reset_index(drop=True)

print(f"Generated {len(df):,} order rows across {df['customer_id'].nunique():,} unique customers")
print(df.groupby("product_category")["profit_margin_pct"].mean().sort_values(ascending=False))

df.to_csv("/home/claude/ecommerce_project/data/ecommerce_orders.csv", index=False)
print("\nSaved to data/ecommerce_orders.csv")
