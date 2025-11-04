import pandas as pd
import os 

file = os.path.join(os.path.dirname(__file__))

values = 0

final_table = None

for filename in os.listdir(file):
    if filename.endswith('_filtered.csv') :
        if filename.startswith('jobs_export'):
            continue

        df = pd.read_csv(os.path.join(file, filename))

        if final_table is None:
            final_table = df
        else:
            final_table = pd.concat([final_table, df], ignore_index=True)
        
        values += len(df)
print(f'Total number of job listings in all CSV files: {values}')

final_table = final_table.drop_duplicates().reset_index(drop=True)
out_path = os.path.join(os.path.dirname(__file__), "filtered_jobs.csv")
final_table.to_csv(out_path, index=False, encoding='utf-8')