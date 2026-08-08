import json

file_path = 'w05_model.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 4 to have BOTH the Colab fallback and the traffic filter
cell_4_code = """import os
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
print(f"Test set: {len(df_test)} pages, {df_test['client_id'].nunique()} clients")"""

# Convert to list of lines
nb['cells'][4]['source'] = [line + '\n' for line in cell_4_code.split('\n')][:-1] + [cell_4_code.split('\n')[-1]]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Fixed Cell 4!")
