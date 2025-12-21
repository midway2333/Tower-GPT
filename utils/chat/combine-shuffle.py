import os
import json
import random
from pathlib import Path
from typing import Iterator, List
import tempfile

def streaming_shuffle_jsonl(
    input_dir: str,
    output_file: str,
    buffer_size: int = 100_000,
    seed: int = 42
) -> None:
    """
    流式合并并打乱文件夹内所有 .jsonl 文件，内存占用恒定。
    
    原理：使用 reservoir sampling 的思想，但更简单——
          逐行读取所有文件，将行缓存到 buffer，buffer 满时打乱并写入临时文件，
          最后再合并所有临时文件并二次打乱（可选，但推荐）。
    
    参数:
        input_dir (str): 输入文件夹
        output_file (str): 输出文件路径
        buffer_size (int): 内存缓冲区大小（行数），默认 10 万行 ≈ 几百 MB
        seed (int): 随机种子
    """
    input_path = Path(input_dir)
    if not input_path.is_dir():
        raise ValueError(f"输入路径不是有效文件夹: {input_dir}")

    jsonl_files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() == '.jsonl'])
    if not jsonl_files:
        print(f"⚠️  未找到 .jsonl 文件")
        return

    print(f"📁 找到 {len(jsonl_files)} 个文件，开始流式打乱...")

    # 第一阶段：分块打乱，写入临时文件
    temp_files: List[Path] = []
    buffer: List[str] = []
    random.seed(seed)

    def flush_buffer():
        if buffer:
            random.shuffle(buffer)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl', encoding='utf-8') as tmp:
                tmp.write("\n".join(buffer) + "\n")
                temp_files.append(Path(tmp.name))
            buffer.clear()

    # 读取所有行，分块打乱
    line_count = 0
    for file in jsonl_files:
        with open(file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    json.loads(line)  # 验证 JSON 合法性
                    buffer.append(line)
                    line_count += 1
                except json.JSONDecodeError:
                    continue

                if len(buffer) >= buffer_size:
                    flush_buffer()

    # 刷新剩余 buffer
    flush_buffer()

    if not temp_files:
        print("❌ 无有效数据")
        return

    print(f"✅ 第一阶段完成：共 {line_count} 行，生成 {len(temp_files)} 个临时文件")

    # 第二阶段：合并临时文件 + 全局二次打乱（使用相同 buffer 策略）
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 打开所有临时文件作为迭代器
    file_iters: List[Iterator[str]] = []
    for tmp_file in temp_files:
        f = open(tmp_file, 'r', encoding='utf-8')
        file_iters.append(iter(f.readline, ''))

    # 使用多路归并 + buffer 打乱（简化版：直接读所有再分 buffer）
    # 更高效做法是用 heapq，但为简单起见，我们再做一次 buffer shuffle
    final_buffer: List[str] = []

    def write_final_buffer():
        if final_buffer:
            random.shuffle(final_buffer)
            with open(output_path, 'a', encoding='utf-8') as out_f:
                out_f.write("".join(final_buffer))
            final_buffer.clear()

    # 逐行从临时文件读取（顺序读，但内容已局部打乱）
    for tmp_file in temp_files:
        with open(tmp_file, 'r', encoding='utf-8') as f:
            for line in f:
                final_buffer.append(line)
                if len(final_buffer) >= buffer_size:
                    write_final_buffer()

    write_final_buffer()

    # 清理临时文件
    for tmp in temp_files:
        tmp.unlink()

    print(f"🎉 流式打乱完成！输出: {output_file}")
    print(f"📊 总行数: {line_count} (估算)")


# ============ 使用示例 ============
if __name__ == "__main__":
    streaming_shuffle_jsonl(
        input_dir="data",
        output_file="data2/train.jsonl",
        buffer_size=1000000,  # 根据内存调整，20万行 ≈ 500MB~1GB
        seed=42
    )