# Code cell not needed for this section, but left empty to satisfy the skeleton
print('Method chosen: Logistic Regression')
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# Load data (local paths first, then GitHub for Colab)
csv_path = 'data/raw/content_refresh_anonymized.csv'
if not os.path.exists(csv_path):
    csv_path = '../data/raw/content_refresh_anonymized.csv'
if not os.path.exists(csv_path):
    csv_path = 'https://raw.githubusercontent.com/mohamed-6513/flyrank_ml/main/data/raw/content_refresh_anonymized.csv'

df = pd.read_csv(csv_path)

# Filter to pages with meaningful traffic (same as baseline)
df = df[df['impressions_prev_30d'] > 100].copy()
print(f'Loaded {len(df)} pages.')

# Label: is declining
df['is_declining_label'] = (df['trend_direction'] == 'down').astype(int)

# Group Shuffle Split by client_id
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
train_idx, test_idx = next(gss.split(df, groups=df['client_id']))

df_train = df.iloc[train_idx].copy()
df_test = df.iloc[test_idx].copy()

print(f"Train set: {len(df_train)} pages, {df_train['client_id'].nunique()} clients")
print(f"Test set: {len(df_test)} pages, {df_test['client_id'].nunique()} clients")
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Features
features = ['content_age_days', 'competition', 'impressions_prev_30d', 'search_volume', 'sessions_prev_30d']
target = 'is_declining_label'

# Train Logistic Regression
model = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value=0, add_indicator=True)),
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
# 1. Feature Importance
# Get dynamic feature names from the imputer (which now adds _missing_indicator columns)
final_features = model.named_steps['imputer'].get_feature_names_out(features)
coefs = model.named_steps['clf'].coef_[0]
importance = pd.DataFrame({'Feature': final_features, 'Coefficient': coefs})
importance = importance.sort_values(by='Coefficient', ascending=False)
print("Feature Coefficients (Positive = Drives 'Declining' prediction):")
display(importance)

# 2. Error Analysis: Top False Positives
df_test_sorted = df_test.sort_values(by='model_prob', ascending=False)
false_positives = df_test_sorted[df_test_sorted['is_declining_label'] == 0].head(3)

print("\nTop 3 False Positives (Model predicted high risk of decline, but page was stable/up):")
display(false_positives[['content_id', 'client_id', 'model_prob', 'trend_direction'] + features])