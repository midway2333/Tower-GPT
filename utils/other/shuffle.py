#!/usr/bin/env python3
import os
import random
import tempfile

# ====== 配置区（只需改这里！）======
INPUT_FILE = "corpus.txt"      # ← 改成你的原始语料路径
OUTPUT_FILE = "shuffled_corpus.txt" # ← 改成你想要的输出路径
BUFFER_SIZE = 1000000                # 每次读取的行数（10万行 ≈ 50~200MB 内存）
RANDOM_SEED = 42                    # 随机种子（保证结果可复现）
# =================================

def split_and_shuffle(input_path, output_path, buffer_size=100000):
    temp_files = []
    
    # 第一阶段：分块打乱
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = []
        for line in f:
            lines.append(line)
            if len(lines) >= buffer_size:
                random.shuffle(lines)
                with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                    tmp.writelines(lines)
                    temp_files.append(tmp.name)
                lines = []
        if lines:
            random.shuffle(lines)
            with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as tmp:
                tmp.writelines(lines)
                temp_files.append(tmp.name)
    
    # 第二阶段：合并打乱
    file_handles = [open(tf, 'r', encoding='utf-8') for tf in temp_files]
    with open(output_path, 'w', encoding='utf-8') as out_f:
        while file_handles:
            idx = random.randint(0, len(file_handles) - 1)
            line = file_handles[idx].readline()
            if not line:
                file_handles[idx].close()
                file_handles.pop(idx)
            else:
                out_f.write(line)
    
    # 清理临时文件
    for tf in temp_files:
        os.unlink(tf)

if __name__ == "__main__":
    print(f"开始打乱文件: {INPUT_FILE}")
    print(f"输出路径: {OUTPUT_FILE}")
    print(f"缓冲行数: {BUFFER_SIZE}, 随机种子: {RANDOM_SEED}")
    
    random.seed(RANDOM_SEED)
    split_and_shuffle(INPUT_FILE, OUTPUT_FILE, BUFFER_SIZE)
    
    print("✅ 打乱完成！")