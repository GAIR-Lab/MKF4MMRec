import pandas as pd
import re

def parse_model_name(full_name):
    """
    Parse model name, extract model, dataset, and embedding processing method.
    """
    parts = full_name.split('-')
    
    # Find the start position of the date part (months like Jul, Jan, Feb, etc.)
    date_start_idx = -1
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    for i, part in enumerate(parts):
        if part in months:
            date_start_idx = i
            break
    
    if date_start_idx == -1 or date_start_idx == 2:
        # No embedding processing method, format like BM3-Games-Jul-02-2025-23-41-37
        model = parts[0]
        dataset = parts[1]
        embedding = 'default'
    else:
        # With embedding processing method, format like BM3-Games-mean-Jul-02-2025-22-28-45
        model = parts[0]
        dataset = parts[1]
        embedding = '-'.join(parts[2:date_start_idx])
    
    return model, dataset, embedding

# Read CSV file
csv_path = "log_summary.csv"
df = pd.read_csv(csv_path)

# Parse model name
df[['model_name', 'dataset', 'embedding']] = df['model'].apply(
    lambda x: pd.Series(parse_model_name(x))
)

# Ignore specified models
models_to_ignore = [
    'BPR', 'LightGCN', 'LayerGCN', 'SELFCFED_LGN', 'MVGAE', 'ItemKNNCBF'
]
df = df[~df['model_name'].isin(models_to_ignore)].copy()

# Filter valid metrics, keeping only @5 and @10
valid_columns = [col for col in df.columns if (
    (col.startswith('valid_') or col.startswith('test_')) and (col.endswith('@5') or col.endswith('@10'))
)]

# Sort by metric name and number
valid_columns.sort(key=lambda x: (x.split('@')[0], int(x.split('@')[1])))

# Extract valid data
df_filtered = df[['model_name', 'dataset', 'embedding'] + valid_columns].copy()
for col in valid_columns:
    df_filtered[col] = pd.to_numeric(df_filtered[col], errors='coerce')

# Calculate the percentage improvement relative to default
result_rows = []
for (model, dataset), group in df_filtered.groupby(['model_name', 'dataset']):
    default_row = group[group['embedding'] == 'baseline']
    if default_row.empty:
        continue
    default_values = default_row.iloc[0]

    # Default row
    base_result = [f"{model}-{dataset}"]
    for col in valid_columns:
        base_result.append(f"{default_values[col]:.4f}")
    result_rows.append(base_result)

    # Other embedding rows
    other_embeddings = group[group['embedding'] != 'baseline'].sort_values('embedding')
    for _, row in other_embeddings.iterrows():
        comp_result = [f"{model}-{dataset}-{row['embedding']}"]
        for col in valid_columns:
            default_val = default_values[col]
            curr_val = row[col]
            if pd.isna(default_val) or pd.isna(curr_val):
                diff_str = f"{curr_val:.4f}   N/A"
            elif default_val == 0:
                if curr_val == 0:
                    diff_str = f"{curr_val:.4f}   +0%"
                else:
                    diff_str = f"{curr_val:.4f}   +∞%"
            else:
                delta = (curr_val - default_val) / default_val * 100
                sign = '+' if delta >= 0 else ''
                diff_str = f"{curr_val:.4f}   {sign}{delta:.1f}%"
            comp_result.append(diff_str)
        result_rows.append(comp_result)

# Save results
final_columns = ['model'] + valid_columns
result_df = pd.DataFrame(result_rows, columns=final_columns)

output_path = "embedding_comparison_result.csv"
result_df.to_csv(output_path, index=False)
print(f"Results saved to: {output_path}")

# Display the first few rows of the results
print("\nPreview of the first 20 rows of results:")
print("=" * 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None) 
print(result_df.head(20).to_string(index=False))