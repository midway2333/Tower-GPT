import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
from tqdm import tqdm
import os

# ========================
# 配置区（请根据实际情况修改）
# ========================
PARQUET_FILE = "data_with_doc_id.parquet"   # 原始输入文件
DEDUPED_IDS_FILE = "deduped_ids.txt"   # Step 3 输出的保留 doc_id 列表
OUTPUT_PARQUET = "deduped_data.parquet" # 最终输出
BATCH_SIZE = 500000                     # 每批处理行数（内存友好）

# ========================
# Step 4: 流式读取原始数据 + 过滤保留 doc_id → 流式写入新 Parquet
# ========================
print("🗃️  正在流式生成去重后的完整数据文件...")

# 1. 加载去重后保留的 doc_id 集合（从 deduped_ids.txt）
print(f"📂 读取保留 doc_id 列表: {DEDUPED_IDS_FILE}")
with open(DEDUPED_IDS_FILE, "r", encoding="utf-8") as f:
    kept_ids = set(int(line.strip()) for line in f if line.strip())

print(f"📌 共加载 {len(kept_ids)} 个保留 doc_id")

# 2. 打开原始 Parquet 文件
parquet_file = pq.ParquetFile(PARQUET_FILE)
total_rows = parquet_file.metadata.num_rows
print(f"📊 原始数据总行数: {total_rows}")

# 3. 获取 schema（保持列结构一致）
schema = parquet_file.schema_arrow

# 4. 初始化 ParquetWriter
writer = None
total_written = 0

try:
    for batch in tqdm(parquet_file.iter_batches(batch_size=BATCH_SIZE),
                      total=(total_rows + BATCH_SIZE - 1) // BATCH_SIZE,
                      desc="流式写入去重数据"):
        # 转为 Pandas DataFrame
        df_batch = batch.to_pandas()

        # 过滤：只保留 kept_ids 中的行
        df_filtered = df_batch[df_batch['doc_id'].isin(kept_ids)]

        if len(df_filtered) == 0:
            continue

        # 转回 PyArrow Table（指定 schema 避免类型错乱）
        table_batch = pa.Table.from_pandas(df_filtered, schema=schema, preserve_index=False)

        # 首次写入时初始化 writer
        if writer is None:
            writer = pq.ParquetWriter(
                OUTPUT_PARQUET,
                schema,
                compression='snappy',  # 可选：'gzip', 'zstd', None
                use_dictionary=True,
                write_statistics=True
            )

        writer.write_table(table_batch)
        total_written += len(df_filtered)

    print(f"✅ 成功写入 {total_written} 行到 {OUTPUT_PARQUET}")
except Exception as e:
    print(f"❌ 写入过程中出错: {e}")
finally:
    if writer:
        writer.close()
        print("🔒 Parquet 文件已安全关闭")

print("🎉 Step 4 完成！")