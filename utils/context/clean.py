import json
import hashlib
import re
import random
from pathlib import Path
from typing import Any, Set, List, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    print("⚠️ 未安装 tqdm，进度条不可用。请运行：pip install tqdm")

# ============ 1. 清理字符串中的 BiDi 控制字符 ============
def clean_bidi_from_string(s: str) -> str:
    bidi_pattern = r'[\u202A-\u202E\u2066-\u2069\u200E\u200F]'
    return re.sub(bidi_pattern, '', s)

# ============ 2. 递归清理 JSON 中所有字符串 + 检测是否被清洗 ============
def clean_bidi_in_json_and_detect(obj: Any) -> Tuple[Any, bool]:
    if isinstance(obj, str):
        cleaned = clean_bidi_from_string(obj)
        was_cleaned = cleaned != obj
        return cleaned, was_cleaned
    elif isinstance(obj, dict):
        new_dict = {}
        was_cleaned = False
        for k, v in obj.items():
            cleaned_v, child_cleaned = clean_bidi_in_json_and_detect(v)
            new_dict[k] = cleaned_v
            if child_cleaned:
                was_cleaned = True
        return new_dict, was_cleaned
    elif isinstance(obj, list):
        new_list = []
        was_cleaned = False
        for item in obj:
            cleaned_item, child_cleaned = clean_bidi_in_json_and_detect(item)
            new_list.append(cleaned_item)
            if child_cleaned:
                was_cleaned = True
        return new_list, was_cleaned
    else:
        return obj, False

# ============ 3. 单文件处理函数（Worker） ============
def process_single_file(file_path: Path, skip_invalid: bool = True):
    results = []
    total_lines_in_file = 0
    skipped_lines = 0
    cleaned_count = 0

    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            total_lines_in_file += 1
            stripped = line.rstrip('\r\n')
            if not stripped.strip():
                skipped_lines += 1
                continue

            try:
                obj = json.loads(stripped)
                cleaned_obj, was_cleaned = clean_bidi_in_json_and_detect(obj)
                if was_cleaned:
                    cleaned_count += 1
                results.append(cleaned_obj)
            except json.JSONDecodeError:
                if skip_invalid:
                    skipped_lines += 1
                    continue
                else:
                    raise

    return file_path.name, results, total_lines_in_file, skipped_lines, cleaned_count

# ============ 4. 六合一终极函数（训练集三切分） ============
def merge_clean_dedup_shuffle_split_multi_train(
    output_dir: str = ".",
    train_main_name: str = "train_main.jsonl",   # 60%
    train_aux_name: str = "train_aux.jsonl",     # 25%
    train_debug_name: str = "train_debug.jsonl", # 15%
    valid_name: str = "valid.jsonl",
    valid_size: int = 10000,
    train_ratios: tuple = (0.6, 0.25, 0.15),  # 训练集内部分割比例
    dir_path: str = ".",
    file_pattern: str = "*.jsonl",
    hash_algo: str = 'md5',
    skip_invalid: bool = True,
    sort_keys_for_dedup: bool = True,
    max_workers: int = 8,
    seed: int = 42
):
    """
    🚀🚀🚀 六合一：合并 + 清理 BiDi + 去重 + 打乱 + 划分验证集 + 训练集三切分

    输出文件：
      - {output_dir}/train_main.jsonl    → 60% of train
      - {output_dir}/train_aux.jsonl     → 25% of train
      - {output_dir}/train_debug.jsonl   → 15% of train
      - {output_dir}/valid.jsonl         → 最后 valid_size 条

    新增：训练集内部按比例切割
    """
    if max_workers is None:
        import os
        max_workers = min(32, (os.cpu_count() or 1) * 2)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    valid_path = Path(output_dir) / valid_name

    # 1️⃣ 找文件
    input_files = list(Path(dir_path).glob(file_pattern))
    if not input_files:
        print("❌ 未找到任何 .jsonl 文件")
        return

    input_files.sort(key=lambda x: x.name)
    print(f"📁 找到 {len(input_files)} 个文件，使用 {max_workers} 线程处理")

    # 2️⃣ 预估总行数（用于进度条）
    total_lines_estimate = 0
    for fp in input_files:
        try:
            with open(fp, 'r', encoding='utf-8') as f:
                total_lines_estimate += sum(1 for _ in f)
        except Exception as e:
            print(f"⚠️ 无法统计 {fp.name} 行数: {e}")

    # 3️⃣ 初始化统计
    stats = {
        'total_raw_lines': 0,
        'total_skipped': 0,
        'total_cleaned_objects': 0,
        'total_after_clean': 0,
        'total_duplicates': 0,
        'final_unique': 0
    }

    # 4️⃣ 并行处理 + 去重
    seen_hashes: Set[str] = set()
    all_cleaned_items: List[Any] = []

    pbar = None
    if tqdm and total_lines_estimate > 0:
        pbar = tqdm(total=total_lines_estimate, desc="🧵 多线程处理中", unit="行", smoothing=0.1)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_file = {
            executor.submit(process_single_file, fp, skip_invalid): fp.name
            for fp in input_files
        }

        for future in as_completed(future_to_file):
            filename = future_to_file[future]
            try:
                fname, results, raw_count, skipped, cleaned_cnt = future.result()

                stats['total_raw_lines'] += raw_count
                stats['total_skipped'] += skipped
                stats['total_cleaned_objects'] += cleaned_cnt
                stats['total_after_clean'] += len(results)

                for cleaned_obj in results:
                    if pbar:
                        pbar.update(1)

                    canonical_str = json.dumps(
                        cleaned_obj,
                        sort_keys=sort_keys_for_dedup,
                        ensure_ascii=False,
                        separators=(',', ':')
                    )
                    hasher = hashlib.new(hash_algo)
                    hasher.update(canonical_str.encode('utf-8'))
                    h = hasher.hexdigest()

                    if h in seen_hashes:
                        stats['total_duplicates'] += 1
                        continue

                    seen_hashes.add(h)
                    all_cleaned_items.append(cleaned_obj)

            except Exception as e:
                print(f"❌ 处理文件 {filename} 时出错: {e}")

    if pbar:
        pbar.close()

    stats['final_unique'] = len(all_cleaned_items)
    total_before_shuffle = stats['final_unique']

    if total_before_shuffle == 0:
        print("❌ 没有有效数据")
        return

    # 5️⃣ 打乱顺序
    print(f"🔀 正在打乱 {total_before_shuffle} 条数据 (seed={seed})...")
    random.seed(seed)
    random.shuffle(all_cleaned_items)

    # 6️⃣ 划分验证集
    if total_before_shuffle <= valid_size:
        print(f"⚠️ 数据总量 {total_before_shuffle} <= 验证集大小 {valid_size}，全部作为验证集")
        train_items = []
        valid_items = all_cleaned_items
    else:
        split_point = total_before_shuffle - valid_size
        train_items = all_cleaned_items[:split_point]
        valid_items = all_cleaned_items[split_point:]

    # 7️⃣ 划分训练子集（按比例）
    train_total = len(train_items)
    if train_total == 0:
        print("⚠️ 训练集为空，无法切分")
        train_main_items = []
        train_aux_items = []
        train_debug_items = []
    else:
        r1, r2, r3 = train_ratios
        # 校验比例和是否为1
        if abs(r1 + r2 + r3 - 1.0) > 1e-5:
            print(f"⚠️ 比例和不为1 ({r1}+{r2}+{r3}={r1+r2+r3})，已自动归一化")
            total_r = r1 + r2 + r3
            r1, r2, r3 = r1/total_r, r2/total_r, r3/total_r

        n1 = int(train_total * r1)
        n2 = int(train_total * r2)
        n3 = train_total - n1 - n2  # 确保总数不变（避免浮点误差）

        train_main_items = train_items[:n1]
        train_aux_items = train_items[n1:n1+n2]
        train_debug_items = train_items[n1+n2:]

    # 8️⃣ 写入所有文件
    def write_jsonl(filepath: Path, items: List[Any], desc: str = ""):
        with open(filepath, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False, separators=(',', ':')) + '\n')
        count = len(items)
        if desc:
            print(f"   - {desc}: {count:,} 条 → {filepath}")
        return count

    print("\n💾 正在写入输出文件...")
    write_jsonl(Path(output_dir) / train_main_name, train_main_items, "主训练集 (60%)")
    write_jsonl(Path(output_dir) / train_aux_name, train_aux_items, "辅助训练集 (25%)")
    write_jsonl(Path(output_dir) / train_debug_name, train_debug_items, "调试训练集 (15%)")
    write_jsonl(valid_path, valid_items, "验证集")

    # 🎉 最终报告
    print("\n" + "="*70)
    print("✅ 六合一处理完成！终极统计报告")
    print("="*70)
    print(f"📁 输入文件数: {len(input_files)}")
    print(f"🧵 使用线程数: {max_workers}")
    print()
    print("📊 原始数据统计:")
    print(f"   - 未清洗前总行数: {stats['total_raw_lines']:,}")
    print(f"   - 被跳过的行数: {stats['total_skipped']:,}")
    print(f"   - 有效 JSON 对象数: {stats['total_after_clean']:,}")
    print()
    print("🧽 清洗统计:")
    print(f"   - 包含 BiDi 控制字符的对象数: {stats['total_cleaned_objects']:,}")
    print()
    print("♻️  去重统计:")
    print(f"   - 重复对象数: {stats['total_duplicates']:,}")
    print(f"   - 去重后唯一对象数: {stats['final_unique']:,}")
    print()
    print("✂️  数据划分:")
    print(f"   - 验证集大小: {len(valid_items):,}")
    print(f"   - 训练集总大小: {train_total:,}")
    print(f"     ├─ 训练集 256 (60%): {len(train_main_items):,}")
    print(f"     ├─ 训练集 512 (25%): {len(train_aux_items):,}")
    print(f"     └─ 训练集 1024 (15%): {len(train_debug_items):,}")
    print("="*70)


# ===== 使用示例 =====
if __name__ == "__main__":
    merge_clean_dedup_shuffle_split_multi_train(
        output_dir="output",
        train_main_name="train_256.jsonl",
        train_aux_name="train_512.jsonl",
        train_debug_name="train_1024.jsonl",
        valid_name="valid.jsonl",
        valid_size=10000,
        train_ratios=(0.6, 0.25, 0.15),
        dir_path=".",
        file_pattern="*.jsonl",
        hash_algo="md5",
        skip_invalid=True,
        sort_keys_for_dedup=True,
        max_workers=8,
        seed=42
    )