"""Test script: runs all code cells from w05_model.ipynb sequentially."""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ── Cell 1: method choice ──
print('Method chosen: Logistic Regression')

# ── Cell 2: split design ──
csv_path = 'data/raw/content_refresh_anonymized.csv'
if not os.path.exists(csv_path):
    csv_path = '../data/raw/content_refresh_anonymized.csv'
if not os.path.exists(csv_path):
    url = 'https://raw.githubusercontent.com/mohamed-6513/flyrank_ml/main/data/raw/content_refresh_anonymized.csv'
    csv_path = url

df = pd.read_csv(csv_path)

# Label: is declining
df['is_declining_label'] = (df['trend_direction'] == 'down').astype(int)

# Group Shuffle Split by client_id
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['client_id']))

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

print(f"Train set: {len(df_train)} pages, {df_train['client_id'].nunique()} clients")
print(f"Test set: {len(df_test)} pages, {df_test['client_id'].nunique()} clients")

# ── Cell 3: train + compare ──
features = ['content_age_days', 'competition', 'impressions_prev_30d', 'search_volume', 'sessions_prev_30d']
target = 'is_declining_label'

model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, class_weight='balanced'))
])
model.fit(df_train[features], df_train[target])

# Predict on test
df_test['model_prob'] = model.predict_proba(df_test[features])[:, 1]

# Recreate baseline score on test set
df_test['is_young'] = (df_test['content_age_days'] < 180).astype(int)
df_test['is_high_comp'] = (df_test['competition'] > 0.71).astype(int)
df_test['baseline_score'] = df_test[['is_young', 'is_high_comp']].max(axis=1) * df_test['impressions_prev_30d']

# Precision@K Evaluation
def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(labels)[order[:k]].mean()

base_rate = df_test['is_declining_label'].mean()
print(f"Base rate (Test set): {base_rate:.4f}\n")

print(f"{'K':<5} | {'Baseline P@K':<15} | {'Model P@K':<15}")
print("-" * 40)
for k in [20, 50, 100, 500]:
    p_baseline = precision_at_k(df_test['baseline_score'], df_test['is_declining_label'], k)
    p_model = precision_at_k(df_test['model_prob'], df_test['is_declining_label'], k)
    print(f"{k:<5} | {p_baseline:.4f}          | {p_model:.4f}")

# ── Cell 4: errors and interpretation ──
coefs = model.named_steps['clf'].coef_[0]
importance = pd.DataFrame({'Feature': features, 'Coefficient': coefs})
importance = importance.sort_values(by='Coefficient', ascending=False)
print("\nFeature Coefficients (Positive = Drives 'Declining' prediction):")
print(importance.to_string(index=False))

# Error Analysis: Top False Positives
df_test_sorted = df_test.sort_values(by='model_prob', ascending=False)
false_positives = df_test_sorted[df_test_sorted['is_declining_label'] == 0].head(3)

print("\nTop 3 False Positives (Model predicted high risk of decline, but page was stable/up):")
print(false_positives[['content_id', 'client_id', 'model_prob', 'trend_direction'] + features].to_string())

print("\n=== ALL CELLS PASSED ===")
