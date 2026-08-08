import json

file_path = 'w05_model.ipynb'

with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Update cell 8 to use dynamic feature names
c8 = """# 1. Feature Importance
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

print("\\nTop 3 False Positives (Model predicted high risk of decline, but page was stable/up):")
display(false_positives[['content_id', 'client_id', 'model_prob', 'trend_direction'] + features])"""

def to_lines(text):
    return [line + '\n' for line in text.split('\n')][:-1] + [text.split('\n')[-1]]

nb['cells'][8]['source'] = to_lines(c8)

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('Updated Cell 8 to handle dynamic feature names')
