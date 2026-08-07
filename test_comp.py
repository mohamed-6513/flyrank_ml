import pandas as pd
df = pd.read_csv('data/raw/content_refresh_anonymized.csv')
df_comp = df[df['competition'] > 0].copy()
df_comp['comp_tier'] = pd.qcut(df_comp['competition'], q=5, duplicates='drop')
sig2 = df_comp.groupby('comp_tier', observed=True).agg(
    n=('content_id', 'count'),
    pct_declining=('trend_direction', lambda x: round((x == 'down').mean() * 100, 1))
)
print(sig2)
