import pandas as pd
import numpy as np

# We use the local anonymized dataset which contains content_age_days
df = pd.read_csv('data/raw/content_refresh_anonymized.csv')

# Filter out pages that don't meet our minimum traffic threshold
df_filtered = df[df['impressions_prev_30d'] > 100].copy()

# Apply the rule components
df_filtered['is_young'] = (df_filtered['content_age_days'] < 180).astype(int)
df_filtered['is_high_comp'] = (df_filtered['competition'] > 0.71).astype(int)

# Calculate the score: (young OR high_comp) * impressions_prior
df_filtered['score'] = df_filtered[['is_young', 'is_high_comp']].max(axis=1) * df_filtered['impressions_prev_30d']

# Assign reason codes
def get_reason_code(row):
    if row['is_young'] and row['is_high_comp']:
        return 'YOUNG_AND_HIGH_COMP'
    elif row['is_young']:
        return 'YOUNG_VOLATILE'
    elif row['is_high_comp']:
        return 'HIGH_COMPETITION'
    else:
        return 'NONE'

df_filtered['reason_code'] = df_filtered.apply(get_reason_code, axis=1)

# Filter out pages with score 0 and rank the rest
df_queue = df_filtered[df_filtered['score'] > 0].copy()
df_queue = df_queue.sort_values(by='score', ascending=False)
df_queue['rank'] = range(1, len(df_queue) + 1)

print("\nTop 20 pages:")
print(df_queue[['rank', 'content_id', 'competition', 'content_age_days', 'impressions_prev_30d', 'score', 'reason_code', 'trend_direction']].head(20).to_string())
