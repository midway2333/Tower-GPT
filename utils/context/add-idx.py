import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import os

def add_doc_id_streaming_by_row(
    input_path,
    output_path,
    doc_id_col_name='doc_id',
    chunk_size=1000000,
    start_id=0  # 起始 ID，可自定义
):
    """
    流式为 Parquet 文件每一行添加递增 doc_id（行号），不改变原始列。
    适用于：每行是一个文档，你想给它一个唯一 ID。
    
    参数:
        input_path: 输入 Parquet 文件路径
        output_path: 输出 Parquet 文件路径
        doc_id_col_name: doc_id 列名，默认 'doc_id'
        chunk_size: 每次读取行数
        start_id: doc_id 起始值
    """
    print(f"📂 开始流式处理: {input_path}")

    # 获取元数据
    meta = pq.read_metadata(input_path)
    total_rows = meta.num_rows
    schema_orig = meta.schema.to_arrow_schema()
    print(f"📊 总行数: {total_rows}, 分块大小: {chunk_size}")

    # 构建新 schema：在最前面插入 doc_id 列
    new_fields = [pa.field(doc_id_col_name, pa.int64())] + list(schema_orig)
    new_schema = pa.schema(new_fields)

    writer = None
    current_id = start_id
    processed_rows = 0

    try:
        parquet_file = pq.ParquetFile(input_path)
        
        for batch in parquet_file.iter_batches(batch_size=chunk_size):
            df_chunk = batch.to_pandas()

            # 👇 核心：添加递增 doc_id（基于行号）
            chunk_size_actual = len(df_chunk)
            df_chunk.insert(0, doc_id_col_name, range(current_id, current_id + chunk_size_actual))

            # 更新当前 ID
            current_id += chunk_size_actual

            # 转回 Arrow Table（使用新 schema）
            table = pa.Table.from_pandas(df_chunk, schema=new_schema)

            # 初始化 writer
            if writer is None:
                writer = pq.ParquetWriter(output_path, new_schema)

            writer.write_table(table)
            processed_rows += chunk_size_actual
            print(f"✅ 已处理 {processed_rows} / {total_rows} 行 (当前 doc_id 到 {current_id - 1})")

    finally:
        if writer:
            writer.close()
            print(f"💾 最终文件已保存: {output_path}")
        else:
            print("⚠️ 未写入任何数据。")

    print("🎉 流式添加 doc_id 完成！")

add_doc_id_streaming_by_row(
    input_path="data.parquet",    # 原始文件，哪怕只有一列 'text'
    output_path="data_with_doc_id.parquet",
    doc_id_col_name="doc_id",
    chunk_size=500000,
    start_id=0  # 可选：从 0 开始编号
)