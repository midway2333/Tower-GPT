import pandas as pd
import pyarrow.parquet as pq
import pickle
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import pyarrow as pa

# ========================
# 配置区
# ========================
PARQUET_FILE = "data_with_doc_id.parquet"
SIGNATURE_FILE = "minhash_signatures.pkl"
DEDUP_OUTPUT = "deduped_ids.txt"
BATCH_SIZE = 500000
NUM_PERM = 128
LSH_THRESHOLD = 0.85
N_JOBS = max(1, cpu_count() - 1)

# ========================
# MinHash 计算函数（供多进程使用）— 必须在顶层定义
# ========================
def compute_minhash_for_row(args):
    doc_id, text = args
    m = MinHash(num_perm=NUM_PERM)
    if pd.isna(text):
        return doc_id, m
    for word in str(text).split():
        word = word.strip()
        if word:
            m.update(word.encode('utf-8'))
    return doc_id, m

# ========================
# 主函数封装 —— 避免进程冲突
# ========================
def main():
    print("🚀 Step 1: 流式读取 Parquet 并多进程计算 MinHash...")

    if os.path.exists(SIGNATURE_FILE):
        os.remove(SIGNATURE_FILE)

    parquet_file = pq.ParquetFile(PARQUET_FILE)
    total_rows = parquet_file.metadata.num_rows
    print(f"总行数: {total_rows}")
    print(f"使用 {N_JOBS} 个进程并行计算...")

    with open(SIGNATURE_FILE, "wb") as f_sig:
        for batch in tqdm(parquet_file.iter_batches(batch_size=BATCH_SIZE, columns=['doc_id', 'text']),
                          total=(total_rows + BATCH_SIZE - 1) // BATCH_SIZE,
                          desc="处理批次"):
            df_batch = batch.to_pandas()
            tasks = df_batch[['doc_id', 'text']].values.tolist()

            # ⚠️ 关键：在每个 batch 内部创建 Pool，避免全局池冲突
            with Pool(processes=N_JOBS) as pool:
                results = list(tqdm(
                    pool.imap_unordered(compute_minhash_for_row, tasks, chunksize=1000),
                    total=len(tasks),
                    desc="并行计算MinHash",
                    leave=False
                ))

            for doc_id, m in results:
                pickle.dump((doc_id, m), f_sig)

    print("✅ MinHash 签名已全部写入磁盘")

    # ========================
    # Step 2: LSH 去重
    # ========================
    print("🔍 Step 2: 构建 LSH 并执行去重...")

    lsh = MinHashLSH(threshold=LSH_THRESHOLD, num_perm=NUM_PERM)
    duplicates = set()
    kept = set()

    with open(SIGNATURE_FILE, "rb") as f_sig:
        pbar = tqdm(desc="去重中")
        while True:
            try:
                doc_id, m = pickle.load(f_sig)
                pbar.update(1)

                candidates = lsh.query(m)
                if candidates:
                    duplicates.add(doc_id)
                else:
                    kept.add(doc_id)
                    lsh.insert(doc_id, m)

            except EOFError:
                break
        pbar.close()

    print(f"📌 总文档数: {len(kept) + len(duplicates)}")
    print(f"✅ 保留文档数: {len(kept)}")
    print(f"🗑️  去重文档数: {len(duplicates)}")

    # ========================
    # Step 3: 保存结果
    # ========================
    print(f"💾 保存去重后的 doc_id 到 {DEDUP_OUTPUT}...")

    with open(DEDUP_OUTPUT, "w", encoding="utf-8") as f_out:
        for doc_id in sorted(kept):
            f_out.write(f"{doc_id}\n")

    print("🎉 全部完成！")

# ========================
# ⚠️ 关键：防止多进程冲突
# ========================
if __name__ == '__main__':
    main()