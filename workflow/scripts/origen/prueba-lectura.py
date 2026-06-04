from pathlib import Path
import pyarrow.parquet as pq

paths = sorted(Path(r"C:/wf-ie-data/results/features/tasa_erosion").glob("*.parquet"))

for p in paths:
    print("Leyendo:", p)
    t = pq.read_table(p, use_threads=False)
    df = t.to_pandas()
    print(df.shape, df.columns.tolist())