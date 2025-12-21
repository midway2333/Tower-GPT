# 清理长文本文件的文件夹

这个文件夹包含用于清理和管理长文本文件的脚本和工具 <br>
乱的很, 看清楚用 <br>
除了这个文件其他都是 AI 写的 ~~AI 真好用~~

### 文件使用

- `clean.py`: 常规去重与清洗脚本
- `dedup.py`: 用于 LSH 去重的脚本, 生成一个对应目录 txt
- `finally-dedup.py`: 使用 `dedup.py` 生成的txt文件, 进行最终去重
- `to-pq.py`: 用于将 jsonl 文件转换为 parquet 文件
- `add-idx.py`: 为 parquet 文件添加索引列
- `pq-see.py`: 用于查看 parquet 文件的脚本
- `to-jsonl.py`: 用于将 parquet 文件做最后处理并转换为 jsonl 文件
- `README.md`: 说明文档
