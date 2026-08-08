"""Extract all code cells from w05_model.ipynb and run them, replacing display() with print()."""
import json

with open('w05_model.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

code_parts = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = ''.join(cell['source'])
        src = src.replace('display(', 'print(')
        code_parts.append(f'# === CELL {i} ===')
        code_parts.append(src)

full_code = '\n'.join(code_parts)
with open('scratch_run_all.py', 'w', encoding='utf-8') as f:
    f.write(full_code)

print(f'Extracted {len(code_parts)//2} code cells')
