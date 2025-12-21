import json
from typing import Optional

def convert_conversations_to_messages(
    input_path: str,
    output_path: str,
    user_key: str = "human",
    assistant_key: str = "assistant"
) -> None:
    """
    将原始对话 JSONL 文件转换为仅包含 messages 字段的新 JSONL 文件。
    支持自定义用户和助手消息的字段名。

    每行输入格式示例:
        {"conversation": [{"human": "...", "assistant": "..."}, ...]}

    每行输出格式:
        {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]}

    参数:
        input_path (str): 原始 JSONL 文件路径
        output_path (str): 转换后 JSONL 文件保存路径
        user_key (str): 对话中表示用户消息的字段名，默认为 "human"
        assistant_key (str): 对话中表示助手消息的字段名，默认为 "assistant"
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue  # 跳过无效 JSON 行
            
            conversation = record.get("conversation")
            if not isinstance(conversation, list):
                continue  # 跳过无有效对话的记录

            messages = []
            for turn in conversation:
                # 添加用户消息（如果存在）
                if user_key in turn:
                    messages.append({"role": "user", "content": turn[user_key]})
                # 添加助手消息（如果存在）
                if assistant_key in turn:
                    messages.append({"role": "assistant", "content": turn[assistant_key]})
            
            # 只有当 messages 非空时才写入（避免空对话）
            if messages:
                fout.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    convert_conversations_to_messages("back\\common_chn_70k.jsonl", "zh_70k.jsonl")
