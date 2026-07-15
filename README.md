# E-commerce Profitability & Customer Segmentation Analysis

End-to-end Python/EDA project analyzing profitability and customer behavior for a
synthetic e-commerce business (~15,000 orders, 2,000 customers, Jan 2024–Dec 2025).
The goal: identify which product categories are actually profitable and which
customers drive the business, to inform pricing and retention strategy.

## Project Description

Using order-level transaction data, this project answers five core business
questions an analytics/MIS role would be expected to tackle:

1. Which product categories have the highest and lowest profit margins?
2. How do customers segment using RFM (Recency, Frequency, Monetary) analysis?
3. Which customer segment drives the most revenue vs. order count — is revenue
   concentrated in a few high-value customers?
4. How does category-wise average order value (AOV) relate to profit margin?
5. What does the monthly revenue/profit trend look like?

All analysis is done in Pandas; all visuals are built with Matplotlib/Seaborn.

## Key Findings

**1. Profit margin varies sharply by category — and revenue share doesn't match margin share.**
Beauty (44.2%) and Apparel (39.2%) are the highest-margin categories, while
Groceries (10.1%) and Electronics (13.0%) are the lowest. Electronics alone
generates ₹6.8 Cr in revenue (the largest of any category) but converts only
13% of it to profit — a classic high-volume, thin-margin category. 4 of 6
categories (67%) sit above the blended average margin of 18.2%.

| Category | Profit Margin | Avg Order Value | Total Revenue |
|---|---|---|---|
| Beauty | 44.2% | ₹1,680 | ₹30.4L |
| Apparel | 39.2% | ₹2,534 | ₹76.3L |
| Home & Kitchen | 27.2% | ₹4,159 | ₹1.03Cr |
| Sports | 25.1% | ₹3,457 | ₹52.7L |
| Electronics | 13.0% | ₹20,742 | ₹6.8Cr |
| Groceries | 10.1% | ₹663 | ₹20.0L |

**2. Revenue is heavily concentrated in a small share of customers.**
RFM segmentation split the 2,000 customers into five behavioral segments:

| Segment | % of Customers | % of Revenue | % of Orders |
|---|---|---|---|
| High-Value | 33.9% | 72.2% | 70.1% |
| At-Risk | 29.2% | 18.9% | 18.8% |
| Regular | 10.2% | 5.1% | 7.5% |
| Churned | 20.8% | 2.9% | 2.8% |
| New | 5.9% | 0.8% | 0.8% |

The **High-Value segment (34% of customers) drives 72% of total revenue**,
confirming revenue is concentrated in repeat, high-spend customers rather than
distributed evenly. A Pareto (concentration) analysis backs this up further:
the top 20% of customers by spend alone generate **58.5% of total revenue**.

**3. AOV and margin are inversely related across categories.**
Correlation between average order value and profit margin across categories
is **-0.45** — big-ticket categories (Electronics) carry high order values but
thin margins, while smaller-basket categories (Beauty) carry the healthiest
margins. This has direct pricing implications: growing basket size alone
(e.g., pushing more Electronics) won't necessarily grow profit at the same rate.

**4. Revenue is seasonal, peaking in the Oct–Dec festive window.**
Monthly revenue shows a clear seasonal spike, with December being the single
highest-revenue month in the dataset (₹1.30 Cr) — consistent with festive/
holiday-season demand patterns typical of Indian e-commerce.

## Tech Stack

- **Python 3.12**
- **Pandas** — data manipulation, groupby aggregation, RFM scoring
- **NumPy** — synthetic data generation, quartile/cumulative calculations
- **Matplotlib** — chart rendering
- **Seaborn** — statistical visualization styling

## Repo Structure

```
ecommerce_project/
├── data/
│   ├── ecommerce_orders.csv       # Raw order-level dataset (~15K rows)
│   ├── category_summary.csv       # Aggregated category-level margin table
│   ├── segment_summary.csv        # RFM segment summary table
│   ├── rfm_table.csv              # Per-customer RFM scores & segments
│   └── monthly_trend.csv          # Monthly revenue/profit rollup
├── charts/
│   ├── 01_category_profit_margin.png
│   ├── 02_customer_segments.png
│   ├── 03_revenue_concentration.png
│   ├── 04_aov_vs_margin.png
│   └── 05_monthly_trend.png
├── notebooks/
│   └── ecommerce_analysis.ipynb   # Full analysis notebook (with outputs)
├── generate_data.py               # Synthetic dataset generator
├── analysis.py                    # Standalone analysis script
└── README.md
```

## How to Run

```bash
# 1. Clone the repo and install dependencies
pip install pandas numpy matplotlib seaborn

# 2. (Optional) Regenerate the synthetic dataset
python generate_data.py

# 3. Run the full analysis — prints findings, saves charts + summary CSVs
python analysis.py

# Or open the notebook for the full walkthrough with inline commentary
jupyter notebook notebooks/ecommerce_analysis.ipynb
```

## Dataset Notes

The dataset is synthetically generated (see `generate_data.py`) but built with
intentional, realistic patterns rather than random noise:
- Category-level cost/margin bands mirror real-world category economics
  (Electronics = low margin/high AOV, Beauty = high margin/low AOV).
- Customers are assigned to purchase-frequency tiers (one-time, occasional,
  power buyers) with recency bias, so RFM segmentation surfaces genuine signal.
- Monthly order density is skewed to create a festive-season (Oct–Dec) revenue bump.


