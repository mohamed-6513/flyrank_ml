import os, pandas as pd, numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

csv_path = 'data/raw/content_refresh_anonymized.csv'
df = pd.read_csv(csv_path)
df = df[df['impressions_prev_30d'] > 100].copy()
df['is_declining_label'] = (df['trend_direction'] == 'down').astype(int)

features = ['content_age_days', 'competition', 'impressions_prev_30d', 'search_volume', 'sessions_prev_30d']
target = 'is_declining_label'

def precision_at_k(scores, labels, k):
    order = np.argsort(-np.asarray(scores))
    return np.asarray(labels)[order[:k]].mean()

def evaluate_model(train_idx, test_idx, split_name):
    df_train = df.iloc[train_idx].copy()
    df_test = df.iloc[test_idx].copy()
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=42, class_weight='balanced'))
    ])
    model.fit(df_train[features], df_train[target])
    df_test['model_prob'] = model.predict_proba(df_test[features])[:, 1]
    print(f'--- {split_name} ---')
    print(f'Base rate: {df_test[target].mean():.4f}')
    for k in [20, 50, 100]:
        p = precision_at_k(df_test['model_prob'], df_test[target], k)
        print(f'  P@{k}: {p:.4f}')
    print()

# Random split
tr, te = train_test_split(np.arange(len(df)), test_size=0.3, random_state=42)
evaluate_model(tr, te, 'Random Split')

# Grouped split
gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
tr_g, te_g = next(gss.split(df, groups=df['client_id']))
evaluate_model(tr_g, te_g, 'Grouped Split')

# Sabotage test
leaky_features = features + ['trend_pct']
df_tr = df.iloc[tr_g].copy()
df_te = df.iloc[te_g].copy()
df_tr['trend_pct'] = df_tr['trend_pct'].fillna(0)
df_te['trend_pct'] = df_te['trend_pct'].fillna(0)
m = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, class_weight='balanced'))
])
m.fit(df_tr[leaky_features], df_tr[target])
df_te['model_prob'] = m.predict_proba(df_te[leaky_features])[:, 1]
order = np.argsort(-np.asarray(df_te['model_prob']))
p50 = np.asarray(df_te[target])[order[:50]].mean()
print(f'Sabotage P@50: {p50:.4f} (Expected: 1.0000)')
print('ALL CELLS OK - No errors')
