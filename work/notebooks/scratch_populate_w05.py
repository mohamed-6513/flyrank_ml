import json

file_path = r'c:\Users\acs\Documents\GitHub\flyrank_ml\work\notebooks\w05_model.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Fix Cell index 4 (the split-design CODE cell) to add Colab URL fallback
nb['cells'][4]['source'] = [
    "import os\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.model_selection import GroupShuffleSplit\n",
    "\n",
    "# Load data (local paths first, then GitHub for Colab)\n",
    "csv_path = 'data/raw/content_refresh_anonymized.csv'\n",
    "if not os.path.exists(csv_path):\n",
    "    csv_path = '../data/raw/content_refresh_anonymized.csv'\n",
    "if not os.path.exists(csv_path):\n",
    "    csv_path = 'https://raw.githubusercontent.com/mohamed-6513/flyrank_ml/main/data/raw/content_refresh_anonymized.csv'\n",
    "\n",
    "df = pd.read_csv(csv_path)\n",
    "print(f'Loaded {len(df)} pages.')\n",
    "\n",
    "# Label: is declining\n",
    "df['is_declining_label'] = (df['trend_direction'] == 'down').astype(int)\n",
    "\n",
    "# Group Shuffle Split by client_id\n",
    "gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)\n",
    "train_idx, test_idx = next(gss.split(df, groups=df['client_id']))\n",
    "\n",
    "df_train = df.iloc[train_idx].copy()\n",
    "df_test = df.iloc[test_idx].copy()\n",
    "\n",
    "print(f\"Train set: {len(df_train)} pages, {df_train['client_id'].nunique()} clients\")\n",
    "print(f\"Test set: {len(df_test)} pages, {df_test['client_id'].nunique()} clients\")"
]

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Done — cell 4 updated with Colab fallback.")
