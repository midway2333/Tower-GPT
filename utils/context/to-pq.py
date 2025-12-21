import json
import pandas as pd
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_large_jsonl_to_parquet(input_dir, output_file, batch_size=10000):
    """
    分批处理大JSONL文件，避免OOM
    """
    input_path = Path(input_dir)
    output_path = Path(output_file)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JSONL文件
    jsonl_files = list(input_path.glob("*.jsonl"))
    if not jsonl_files:
        logger.error("未找到JSONL文件")
        return False
    
    logger.info(f"找到 {len(jsonl_files)} 个JSONL文件")
    
    # 第一次遍历：获取所有字段名
    all_fields = set()
    for file_path in jsonl_files:
        logger.info(f"扫描字段: {file_path.name}")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        all_fields.update(data.keys())
                    except:
                        continue
                if len(all_fields) > 1000:  # 防止字段太多
                    break
    
    logger.info(f"发现字段: {list(all_fields)}")
    
    # 初始化Parquet writer
    schema = None
    writer = None
    total_records = 0
    batch_count = 0
    
    # 分批处理数据
    for file_path in jsonl_files:
        logger.info(f"处理文件: {file_path.name}")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            batch_data = []
            
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        batch_data.append(data)
                        total_records += 1
                        
                        # 达到批次大小，写入文件
                        if len(batch_data) >= batch_size:
                            df_batch = pd.DataFrame(batch_data)
                            
                            if writer is None:
                                # 第一次写入，创建writer
                                table = pa.Table.from_pandas(df_batch)
                                writer = pq.ParquetWriter(output_path, table.schema)
                                writer.write_table(table)
                            else:
                                table = pa.Table.from_pandas(df_batch)
                                writer.write_table(table)
                            
                            batch_count += 1
                            logger.info(f"已处理批次 {batch_count}, 总记录数: {total_records}")
                            batch_data = []
                            
                    except json.JSONDecodeError:
                        continue
            
            # 处理最后一批数据
            if batch_data:
                df_batch = pd.DataFrame(batch_data)
                if writer is None:
                    table = pa.Table.from_pandas(df_batch)
                    writer = pq.ParquetWriter(output_path, table.schema)
                    writer.write_table(table)
                else:
                    table = pa.Table.from_pandas(df_batch)
                    writer.write_table(table)
                
                batch_count += 1
                logger.info(f"最后批次 {batch_count}, 总记录数: {total_records}")
    
    # 关闭writer
    if writer:
        writer.close()
    
    logger.info(f"转换完成！总记录数: {total_records}")
    return True

if __name__ == "__main__":
    process_large_jsonl_to_parquet(
        input_dir="output",
        output_file="data.parquet",
        batch_size=50000  # 根据内存调整批次大小
    )