import os
import glob
import pandas as pd

data_dir = "data/retrieval"
xlsx_files = glob.glob(os.path.join(data_dir, "*.xlsx"))

print(f"Found {len(xlsx_files)} .xlsx files.")

for f in xlsx_files:
    try:
        df = pd.read_excel(f)
        csv_filename = f.replace(".xlsx", ".csv")
        df.to_csv(csv_filename, index=False)
        print(f"Converted: {f} -> {csv_filename}")
    except Exception as e:
        print(f"Failed to convert {f}: {e}")
