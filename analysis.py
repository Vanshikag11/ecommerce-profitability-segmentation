"""
analysis.py
E-commerce Profitability & Customer Segmentation Analysis

Covers:
  1. Category-wise profit margin analysis
  2. RFM (Recency, Frequency, Monetary) customer segmentation
  3. Revenue/order concentration by segment
  4. Category-wise AOV vs margin relationship
  5. Monthly revenue & profit trend

Outputs 5 PNG charts to charts/ and prints all key numbers used
in the README / resume bullet.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

CHART_DIR = "charts"

# -----------------------------------------------------------------
# Load data
# -----------------------------------------------------------------
df = pd.read_csv("data/ecommerce_orders.csv", parse_dates=["order_date"])

print("=" * 70)
print(f"Dataset: {len(df):,} orders | {df['customer_id'].nunique():,} unique customers")
print(f"Date range: {df['order_date'].min().date()} to {df['order_date'].max().date()}")
print("=" * 70)

# -----------------------------------------------------------------
# 1. CATEGORY-WISE PROFIT MARGIN
# -----------------------------------------------------------------
cat_summary = (
    df.groupby("product_category")
    .agg(
        total_revenue=("total_revenue", "sum"),
        total_profit=("profit", "sum"),
        orders=("order_id", "count"),
        avg_order_value=("total_revenue", "mean"),
    )
    .reset_index()
)
cat_summary["profit_margin_pct"] = (cat_summary["total_profit"] / cat_summary["total_revenue"] * 100).round(2)
cat_summary = cat_summary.sort_values("profit_margin_pct", ascending=False)

print("\n--- Category-wise profit margin ---")
print(cat_summary[["product_category", "profit_margin_pct", "avg_order_value", "total_revenue"]]
      .to_string(index=False))

highest_margin_cat = cat_summary.iloc[0]
lowest_margin_cat = cat_summary.iloc[-1]
# "high-margin categories" = top categories above overall average margin
overall_margin = df["profit"].sum() / df["total_revenue"].sum() * 100
high_margin_cats = cat_summary[cat_summary["profit_margin_pct"] > overall_margin]
pct_high_margin_cats = round(len(high_margin_cats) / len(cat_summary) * 100, 1)

print(f"\nOverall blended profit margin: {overall_margin:.2f}%")
print(f"Highest-margin category: {highest_margin_cat['product_category']} "
      f"({highest_margin_cat['profit_margin_pct']:.2f}%)")
print(f"Lowest-margin category: {lowest_margin_cat['product_category']} "
      f"({lowest_margin_cat['profit_margin_pct']:.2f}%)")
print(f"{len(high_margin_cats)} of {len(cat_summary)} categories "
      f"({pct_high_margin_cats}%) sit above the blended average margin.")

# Chart 1: Category profit margin bar chart
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#2ecc71" if v > overall_margin else "#e74c3c" for v in cat_summary["profit_margin_pct"]]
bars = ax.bar(cat_summary["product_category"], cat_summary["profit_margin_pct"], color=colors)
ax.axhline(overall_margin, color="grey", linestyle="--", linewidth=1,
           label=f"Blended avg ({overall_margin:.1f}%)")
for bar, val in zip(bars, cat_summary["profit_margin_pct"]):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.7, f"{val:.1f}%",
            ha="center", fontsize=9, fontweight="bold")
ax.set_title("Profit Margin by Product Category", fontsize=14, fontweight="bold")
ax.set_ylabel("Profit Margin (%)")
ax.set_xlabel("")
ax.legend()
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/01_category_profit_margin.png")
plt.close()

# -----------------------------------------------------------------
# 2. RFM CUSTOMER SEGMENTATION
# -----------------------------------------------------------------
snapshot_date = df["order_date"].max() + pd.Timedelta(days=1)

rfm = df.groupby("customer_id").agg(
    recency=("order_date", lambda x: (snapshot_date - x.max()).days),
    frequency=("order_id", "count"),
    monetary=("total_revenue", "sum"),
).reset_index()

# Score 1-4 (4 = best) using quartiles
rfm["R_score"] = pd.qcut(rfm["recency"].rank(method="first"), 4, labels=[4, 3, 2, 1]).astype(int)
rfm["F_score"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["M_score"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
rfm["RFM_score"] = rfm["R_score"] + rfm["F_score"] + rfm["M_score"]


def segment_customer(row):
    if row["R_score"] >= 3 and row["F_score"] >= 3 and row["M_score"] >= 3:
        return "High-Value"
    elif row["F_score"] == 1 and row["R_score"] <= 2:
        # only ever ordered once (or nearly) and it was a while ago
        return "Churned"
    elif row["R_score"] <= 2 and row["F_score"] >= 2:
        return "At-Risk"
    elif row["frequency"] == 1 and row["R_score"] >= 3:
        return "New"
    else:
        return "Regular"


rfm["segment"] = rfm.apply(segment_customer, axis=1)

seg_summary = (
    rfm.groupby("segment")
    .agg(customers=("customer_id", "count"), revenue=("monetary", "sum"), orders=("frequency", "sum"))
    .reset_index()
)
seg_summary["pct_customers"] = (seg_summary["customers"] / seg_summary["customers"].sum() * 100).round(1)
seg_summary["pct_revenue"] = (seg_summary["revenue"] / seg_summary["revenue"].sum() * 100).round(1)
seg_summary["pct_orders"] = (seg_summary["orders"] / seg_summary["orders"].sum() * 100).round(1)
seg_summary = seg_summary.sort_values("revenue", ascending=False)

print("\n--- RFM customer segments ---")
print(seg_summary.to_string(index=False))

high_value_row = seg_summary[seg_summary["segment"] == "High-Value"].iloc[0]
print(f"\nHigh-Value segment: {high_value_row['pct_customers']}% of customers "
      f"drive {high_value_row['pct_revenue']}% of revenue "
      f"({high_value_row['pct_orders']}% of order volume).")

# Chart 2: Segment distribution (customer count) + revenue share
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
seg_order = seg_summary.sort_values("customers", ascending=False)
sns.barplot(data=seg_order, x="segment", y="pct_customers", hue="segment", ax=axes[0], palette="viridis", legend=False)
axes[0].set_title("Customer Segment Distribution (%)", fontweight="bold")
axes[0].set_ylabel("% of Customers")
axes[0].set_xlabel("")
for p in axes[0].patches:
    axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                      ha="center", fontsize=9, fontweight="bold")

seg_rev_order = seg_summary.sort_values("revenue", ascending=False)
sns.barplot(data=seg_rev_order, x="segment", y="pct_revenue", hue="segment", ax=axes[1], palette="magma", legend=False)
axes[1].set_title("Revenue Share by Segment (%)", fontweight="bold")
axes[1].set_ylabel("% of Total Revenue")
axes[1].set_xlabel("")
for p in axes[1].patches:
    axes[1].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                      ha="center", fontsize=9, fontweight="bold")

for ax in axes:
    ax.tick_params(axis="x", rotation=15)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/02_customer_segments.png")
plt.close()

# Chart 3: Revenue concentration - customers vs revenue % (Pareto-style)
rfm_sorted = rfm.sort_values("monetary", ascending=False).reset_index(drop=True)
rfm_sorted["cum_customers_pct"] = (np.arange(1, len(rfm_sorted) + 1) / len(rfm_sorted)) * 100
rfm_sorted["cum_revenue_pct"] = (rfm_sorted["monetary"].cumsum() / rfm_sorted["monetary"].sum()) * 100

top20_idx = (rfm_sorted["cum_customers_pct"] <= 20).sum()
revenue_from_top20 = rfm_sorted.loc[:top20_idx - 1, "cum_revenue_pct"].iloc[-1] if top20_idx > 0 else 0

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.plot(rfm_sorted["cum_customers_pct"], rfm_sorted["cum_revenue_pct"], color="#8e44ad", linewidth=2.5)
ax.fill_between(rfm_sorted["cum_customers_pct"], rfm_sorted["cum_revenue_pct"], alpha=0.15, color="#8e44ad")
ax.axvline(20, color="grey", linestyle="--", linewidth=1)
ax.axhline(revenue_from_top20, color="grey", linestyle="--", linewidth=1)
ax.scatter([20], [revenue_from_top20], color="#e74c3c", zorder=5, s=60)
ax.annotate(f"Top 20% of customers\n= {revenue_from_top20:.1f}% of revenue",
            xy=(20, revenue_from_top20), xytext=(30, revenue_from_top20 - 20),
            fontsize=10, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="black"))
ax.set_title("Revenue Concentration Across Customers", fontsize=14, fontweight="bold")
ax.set_xlabel("Cumulative % of Customers (ranked by spend)")
ax.set_ylabel("Cumulative % of Revenue")
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/03_revenue_concentration.png")
plt.close()

print(f"\nRevenue concentration: top 20% of customers by spend generate "
      f"{revenue_from_top20:.1f}% of total revenue.")

# -----------------------------------------------------------------
# 3. CATEGORY-WISE AOV vs MARGIN
# -----------------------------------------------------------------
print("\n--- AOV vs margin relationship ---")
corr = cat_summary["avg_order_value"].corr(cat_summary["profit_margin_pct"])
print(f"Correlation between AOV and profit margin across categories: {corr:.2f}")

fig, ax = plt.subplots(figsize=(8.5, 5.5))
sns.scatterplot(data=cat_summary, x="avg_order_value", y="profit_margin_pct",
                 s=180, hue="product_category", palette="tab10", ax=ax, legend=False)
for _, row in cat_summary.iterrows():
    ax.annotate(row["product_category"], (row["avg_order_value"], row["profit_margin_pct"]),
                textcoords="offset points", xytext=(8, 5), fontsize=9, fontweight="bold")
ax.set_title("Average Order Value vs. Profit Margin by Category", fontsize=14, fontweight="bold")
ax.set_xlabel("Average Order Value (₹)")
ax.set_ylabel("Profit Margin (%)")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/04_aov_vs_margin.png")
plt.close()

# -----------------------------------------------------------------
# 4. MONTHLY REVENUE / PROFIT TREND
# -----------------------------------------------------------------
df["order_month"] = df["order_date"].dt.to_period("M").dt.to_timestamp()
monthly = df.groupby("order_month").agg(revenue=("total_revenue", "sum"), profit=("profit", "sum")).reset_index()

print("\n--- Monthly revenue/profit trend (last 6 months) ---")
print(monthly.tail(6).to_string(index=False))

best_month = monthly.loc[monthly["revenue"].idxmax()]
print(f"\nPeak month: {best_month['order_month'].strftime('%b %Y')} "
      f"with ₹{best_month['revenue']:,.0f} revenue.")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(monthly["order_month"], monthly["revenue"], marker="o", linewidth=2, label="Revenue", color="#2980b9")
ax.plot(monthly["order_month"], monthly["profit"], marker="o", linewidth=2, label="Profit", color="#27ae60")
ax.set_title("Monthly Revenue & Profit Trend", fontsize=14, fontweight="bold")
ax.set_xlabel("Month")
ax.set_ylabel("Amount (₹)")
ax.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/05_monthly_trend.png")
plt.close()

# -----------------------------------------------------------------
# Save summary tables for README generation
# -----------------------------------------------------------------
cat_summary.to_csv("data/category_summary.csv", index=False)
seg_summary.to_csv("data/segment_summary.csv", index=False)
rfm.to_csv("data/rfm_table.csv", index=False)
monthly.to_csv("data/monthly_trend.csv", index=False)

print("\n" + "=" * 70)
print("KEY NUMBERS FOR README / RESUME BULLET")
print("=" * 70)
print(f"Highest-margin category: {highest_margin_cat['product_category']} at {highest_margin_cat['profit_margin_pct']:.1f}% margin")
print(f"Lowest-margin category: {lowest_margin_cat['product_category']} at {lowest_margin_cat['profit_margin_pct']:.1f}% margin")
print(f"High-margin categories above blended avg: {pct_high_margin_cats}% of categories ({len(high_margin_cats)}/{len(cat_summary)})")
print(f"High-Value segment: {high_value_row['pct_customers']}% of customers -> {high_value_row['pct_revenue']}% of revenue")
print(f"Top 20% of customers generate {revenue_from_top20:.1f}% of revenue")
print("Charts saved to charts/. Summary tables saved to data/.")
