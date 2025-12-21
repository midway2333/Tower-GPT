import pandas as pd
import pyarrow.parquet as pq

PARQUET_FILE = "data_with_doc_id.parquet"

# 只读取元数据，不加载数据
pf = pq.ParquetFile(PARQUET_FILE)
print("文件列名:", pf.schema.names)
print("\n文件结构:")
print(pf.schema)

# 读取第一个row group的前2行（如果文件分块）
first_row_group = pf.read_row_group(0)
first_two_rows = first_row_group.slice(0, 2)
print("\n第一个row group的前2行:")
print(first_two_rows.to_pandas())