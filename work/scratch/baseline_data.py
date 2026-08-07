import pandas as pd
import numpy as np

print("Running baseline check on local CSV...")
df = pd.read_csv('data/raw/content_refresh_anonymized.csv')

# Calculate label
df['label_down'] = (df['trend_direction'] == 'down').astype(int)

print(f"Number of rows: {len(df)}")
print(f"Base rate of 'down': {df['label_down'].mean():.4f}")

# Let's try the rule: High competition (>0.71) + Young page (<180 days)
young = (df["content_age_days"] < 180).astype(int)
high_comp = (df["competition"] > 0.71).astype(int)
visible = (df["impressions_prev_30d"] >= 100).astype(int)

print("Correlations with label_down:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corrs = df[numeric_cols].corr()['label_down'].sort_values()
print(corrs.head(10))
print(corrs.tail(10))

# Try rule: declining traffic in last 30d compared to prev 30d?
# wait, impressions_prev_30d vs impressions_90d?
# Let's try high comp * young * impressions_prev_30d
df['score'] = (df['competition'] > 0.71).astype(int) * (df['content_age_days'] < 180).astype(int) * df['impressions_prev_30d']

def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(labels)[order[:k]].mean()

for k in [20, 50, 100, 500]:
    print(f"Precision@{k} (high_comp * young * impr): {precision_at_k(df['score'], df['label_down'], k):.4f}")

df = df.sort_values(by='score', ascending=False)
print("\nTop 20 pages:")
print(df[['content_id', 'competition', 'content_age_days', 'impressions_prev_30d', 'score', 'label_down']].head(20))
