"""
Generate a synthetic (fake) customer dataset for practice.

I couldn't use real customer data (privacy!), so I built a small dataset that
looks like what a telecom / subscription company might have. I also added some
"messy" problems on purpose (duplicates, missing values, outliers, inconsistent
text) so I can practice the cleaning steps from my Data Visualization class.

Run:  python data/generate_data.py
"""
import numpy as np
import pandas as pd

np.random.seed(42)

# 4 hidden groups of customers (K-means should try to find them later)
groups = {
    # name: (count, tenure_mean, spend_mean, calls_mean, data_mean)
    "loyal_high_spend": (120, 48, 420, 1.0, 35),
    "new_struggling":   (100, 4,  150, 6.0, 12),
    "heavy_data":       (130, 20, 260, 2.0, 80),
    "low_engagement":   (110, 36, 90,  0.5, 5),
}

rows = []
for name, (n, tenure, spend, calls, data) in groups.items():
    for _ in range(n):
        rows.append({
            "tenure_months": max(1, int(np.random.normal(tenure, tenure * 0.25 + 1))),
            "monthly_spend_sar": round(max(30, np.random.normal(spend, spend * 0.15)), 2),
            "support_calls_6m": int(np.random.poisson(calls)),
            "data_usage_gb": round(max(0.5, np.random.normal(data, data * 0.2 + 1)), 1),
            "plan_type": np.random.choice(["Prepaid", "Postpaid"], p=[0.4, 0.6]),
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
df.insert(0, "customer_id", [f"C{1000 + i}" for i in range(len(df))])

# ---- make the data messy on purpose ----
# 1) inconsistent text formatting
messy_idx = df.sample(40, random_state=1).index
df.loc[messy_idx, "plan_type"] = df.loc[messy_idx, "plan_type"].apply(
    lambda s: np.random.choice([s.upper(), s.lower(), " " + s + " "])
)

# 2) missing values
df.loc[df.sample(15, random_state=2).index, "monthly_spend_sar"] = np.nan
df.loc[df.sample(12, random_state=3).index, "data_usage_gb"] = np.nan

# 3) a few extreme outliers (e.g. data entry errors)
out_idx = df.sample(6, random_state=4).index
df.loc[out_idx, "monthly_spend_sar"] = [2500, 3100, 1990, 2750, 4000, 2200]

# 4) duplicate rows
df = pd.concat([df, df.sample(10, random_state=5)], ignore_index=True)

df.to_csv("data/customers_raw.csv", index=False)
print("Saved data/customers_raw.csv with shape", df.shape)
