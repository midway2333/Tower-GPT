import pyarrow.parquet as pq
import json
import os
from typing import List, Tuple, Optional

def split_parquet_to_jsonl(
    input_parquet_path: str,
    output_dir: str = "./output",
    remove_field: str = "doc_id",
    ratios: Tuple[float, float, float] = (0.60, 0.25, 0.15),
    extra_rows: int = 10000,
    output_names: Tuple[str, str, str, str] = ("train.jsonl", "val.jsonl", "test.jsonl", "extra.jsonl"),
    batch_size: int = 10000,
    seed: Optional[int] = None
):
    """
    流式读取 Parquet 文件，移除指定字段，按比例 + 额外行数分割为 4 个 JSONL 文件。

    Args:
        input_parquet_path (str): 输入 Parquet 文件路径
        output_dir (str): 输出目录
        remove_field (str): 要移除的字段名
        ratios (tuple): 前三部分的比例 (train, val, test)，总和应为 1.0
        extra_rows (int): 额外抽取的行数（从剩余数据中不重复抽取）
        output_names (tuple): 输出文件名 (train, val, test, extra)
        batch_size (int): 每次读取的行数（流式控制）
        seed (int, optional): 随机种子（用于确定性分割，暂未实现随机，顺序分割）
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开 Parquet 文件（流式）
    parquet_file = pq.ParquetFile(input_parquet_path)
    total_rows = parquet_file.metadata.num_rows
    print(f"总行数: {total_rows}")

    # 计算各部分行数
    train_rows = int(total_rows * ratios[0]) - extra_rows
    val_rows = int(total_rows * ratios[1])
    test_rows = int(total_rows * ratios[2])
    allocated = train_rows + val_rows + test_rows

    # 确保前三部分不超过总数
    if allocated > total_rows:
        raise ValueError("比例之和超过 1.0 或计算后行数溢出")

    # 额外部分不能超过剩余
    remaining = total_rows - allocated
    actual_extra = min(extra_rows, remaining)
    print(f"分配: 1={train_rows}, 2={val_rows}, 3={test_rows}, extra={actual_extra}")

    # 打开输出文件
    train_path = os.path.join(output_dir, output_names[0])
    val_path = os.path.join(output_dir, output_names[1])
    test_path = os.path.join(output_dir, output_names[2])
    extra_path = os.path.join(output_dir, output_names[3])

    train_f = open(train_path, 'w', encoding='utf-8')
    val_f = open(val_path, 'w', encoding='utf-8')
    test_f = open(test_path, 'w', encoding='utf-8')
    extra_f = open(extra_path, 'w', encoding='utf-8')

    # 初始化计数器
    row_count = 0
    train_written = 0
    val_written = 0
    test_written = 0
    extra_written = 0

    # 流式读取批次
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        records = batch.to_pylist()
        for record in records:
            # 移除指定字段
            if remove_field in record:
                del record[remove_field]

            # 写入对应文件
            if train_written < train_rows:
                train_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                train_written += 1
            elif val_written < val_rows:
                val_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                val_written += 1
            elif test_written < test_rows:
                test_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                test_written += 1
            elif extra_written < actual_extra:
                extra_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                extra_written += 1
            else:
                # 超出部分丢弃（或可选写入其他文件）
                pass

            row_count += 1
            if row_count % 100000 == 0:
                print(f"已处理 {row_count} 行...")

    # 关闭文件
    train_f.close()
    val_f.close()
    test_f.close()
    extra_f.close()

    print("✅ 处理完成！")
    print(f"最终写入: train={train_written}, val={val_written}, test={test_written}, extra={extra_written}")


# 示例调用
if __name__ == "__main__":
    split_parquet_to_jsonl(
        input_parquet_path="deduped_data.parquet",
        output_dir="./split_output",
        remove_field="doc_id",
        ratios=(0.60, 0.25, 0.15),
        extra_rows=10000,
        output_names=("train-256.jsonl", "train-512.jsonl", "train-1024.jsonl", "val.jsonl"),
        batch_size=500000
    )