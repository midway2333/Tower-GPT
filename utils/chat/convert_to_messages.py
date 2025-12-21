import json
import os
from pathlib import Path
from typing import List, Dict, Any

def is_single_turn_record(record: Dict[str, Any]) -> bool:
    """判断是否为单轮问答格式（含 prompt + response）"""
    return "prompt" in record and "response" in record

def is_multi_turn_record(record: Dict[str, Any]) -> bool:
    """判断是否为多轮对话格式（含 conversation 列表）"""
    conv = record.get("conversation")
    return isinstance(conv, list) and len(conv) > 0

def convert_single_turn_to_messages(record: Dict[str, Any]) -> List[Dict[str, str]]:
    """将单轮问答转为 messages"""
    return [
        {"role": "user", "content": record["prompt"]},
        {"role": "assistant", "content": record["response"]}
    ]

def convert_multi_turn_to_messages(
    record: Dict[str, Any],
    user_key: str = "human",
    assistant_key: str = "assistant"
) -> List[Dict[str, str]]:
    """将多轮对话转为 messages"""
    messages = []
    for turn in record["conversation"]:
        if user_key in turn:
            messages.append({"role": "user", "content": turn[user_key]})
        if assistant_key in turn:
            messages.append({"role": "assistant", "content": turn[assistant_key]})
    return messages

def process_file(file_path: Path, output_lines: List[str]) -> int:
    """处理单个文件，将有效行追加到 output_lines，返回处理的行数"""
    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            messages = None
            if is_single_turn_record(record):
                messages = convert_single_turn_to_messages(record)
            elif is_multi_turn_record(record):
                messages = convert_multi_turn_to_messages(record)
            else:
                continue  # 不符合任一格式，跳过

            if messages:
                output_lines.append(json.dumps({"messages": messages}, ensure_ascii=False))
                count += 1
    return count

def batch_convert_folder_to_messages(
    input_folder: str,
    output_folder: str,
    base_name: str = "chat-text",
    user_key: str = "human",
    assistant_key: str = "assistant"
) -> None:
    """
    批量转换文件夹内所有 JSONL 文件为统一 messages 格式。
    
    参数:
        input_folder: 输入文件夹路径（包含多个 .jsonl 文件）
        output_folder: 输出文件夹路径
        base_name: 输出文件基础名，默认 "chat-text"
        user_key: 多轮对话中用户消息的键名（默认 "human"）
        assistant_key: 多轮对话中助手消息的键名（默认 "assistant"）
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    # 收集所有 .jsonl 文件（不递归子目录）
    jsonl_files = sorted([f for f in input_path.iterdir() if f.is_file() and f.suffix.lower() == '.jsonl'])
    
    if not jsonl_files:
        print(f"⚠️  在 {input_folder} 中未找到 .jsonl 文件")
        return

    all_output_lines = []
    total_files = len(jsonl_files)
    total_lines = 0

    print(f"📁 正在处理 {total_files} 个文件...")
    for file in jsonl_files:
        count = process_file(file, all_output_lines)
        total_lines += count
        print(f"  ✅ {file.name} → {count} 条对话")

    if not all_output_lines:
        print("❌ 未找到任何有效对话数据")
        return

    # 按每文件 100,000 行分片（可调整）
    MAX_LINES_PER_FILE = 300000
    num_output_files = (len(all_output_lines) + MAX_LINES_PER_FILE - 1) // MAX_LINES_PER_FILE

    for i in range(num_output_files):
        start_idx = i * MAX_LINES_PER_FILE
        end_idx = min(start_idx + MAX_LINES_PER_FILE, len(all_output_lines))
        chunk = all_output_lines[start_idx:end_idx]

        output_file = output_path / f"{base_name}{i+1}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(chunk) + "\n")
        print(f"💾 已保存: {output_file.name} ({len(chunk)} 行)")

    print(f"\n🎉 共处理 {total_lines} 条对话，输出到 {num_output_files} 个文件")

if __name__ == "__main__":
    batch_convert_folder_to_messages(
        input_folder="./back/",              # 原始 JSONL 文件所在文件夹
        output_folder="./cleaned_data/",     # 输出文件夹
        base_name="chat-text",               # 输出文件名前缀
        user_key="human",                    # 多轮对话的用户键（可选）
        assistant_key="assistant"            # 多轮对话的助手键（可选）
    )
