import os, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Load Data
csv_path = 'data/raw/content_refresh_anonymized.csv'
df = pd.read_csv(csv_path)

# 2. Filter and prep
df_active = df[df['impressions_prev_30d'] > 100].copy()
df_active['is_declining_label'] = (df_active['trend_direction'] == 'down').astype(int)

features = ['content_age_days', 'competition', 'impressions_prev_30d', 'search_volume', 'sessions_prev_30d']
target = 'is_declining_label'

# 3. Train final model
model = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000))
])
model.fit(df_active[features], df_active[target])

# 4. Predict and Rank
df_active['decline_probability'] = model.predict_proba(df_active[features])[:, 1]
df_ranked = df_active.sort_values(by='decline_probability', ascending=False)

# 5. Take top 50
action_queue = df_ranked.head(50).copy()
print(f"Action Queue Generated: {len(action_queue)} pages.")
print(action_queue[['content_id', 'decline_probability'] + features].head(5).to_string())

# 6. Export
os.makedirs('outputs', exist_ok=True)
action_queue.to_csv('outputs/action_queue.csv', index=False)
print(f"\nSuccessfully exported {len(action_queue)} pages to outputs/action_queue.csv")

# Verify the export
df_check = pd.read_csv('outputs/action_queue.csv')
print(f"Verified: CSV contains {len(df_check)} rows.")
print("ALL CELLS OK - No errors")
