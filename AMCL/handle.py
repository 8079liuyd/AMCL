import pandas as pd
from sklearn.model_selection import KFold
import os

RANDOM_STATE = 42
N_SPLITS = 5
INPUT_FILENAME = 'data.csv'

try:
    print(f"Loading data from {INPUT_FILENAME}...")
    df = pd.read_csv(INPUT_FILENAME)
    print(f"Loaded {len(df)} rows.")

    unique_genes = df['gene'].unique()
    gene_map = {gene: i for i, gene in enumerate(unique_genes)}
    df_gene_map = pd.DataFrame(list(gene_map.items()), columns=['gene_name', 'gene_id'])
    gene_map_filename = 'gene_mapping.csv'
    df_gene_map.to_csv(gene_map_filename, index=False)
    print(f"Created {gene_map_filename} with {len(unique_genes)} unique genes.")

    unique_drugs = df['drug'].unique()
    drug_map = {drug: i for i, drug in enumerate(unique_drugs)}
    df_drug_map = pd.DataFrame(list(drug_map.items()), columns=['drug_name', 'drug_id'])
    drug_map_filename = 'drug_mapping.csv'
    df_drug_map.to_csv(drug_map_filename, index=False)
    print(f"Created {drug_map_filename} with {len(unique_drugs)} unique drugs.")

    df_processed = pd.DataFrame({
        'gene': df['gene'].map(gene_map),
        'drug': df['drug'].map(drug_map),
        'interaction': df['interaction']
    })
    print(f"\nProcessed DataFrame head (with IDs):\n{df_processed.head()}")

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    print(f"\nCreating {N_SPLITS}-fold cross-validation files (80% train, 20% test)...")
    generated_files = [gene_map_filename, drug_map_filename]

    for i, (train_index, test_index) in enumerate(kf.split(df_processed)):
        df_train = df_processed.iloc[train_index]
        df_test = df_processed.iloc[test_index]

        train_filename = f'train{i}.csv'
        test_filename = f'test{i}.csv'

        df_train.to_csv(train_filename, index=False)
        df_test.to_csv(test_filename, index=False)

        generated_files.extend([train_filename, test_filename])

        print(f"  - Created {train_filename} ({len(df_train)} rows)")
        print(f"  - Created {test_filename} ({len(df_test)} rows)")

    print("\nFile creation complete.")
    print("Generated files:")
    print(generated_files)

except FileNotFoundError:
    print(f"Error: Input file '{INPUT_FILENAME}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")