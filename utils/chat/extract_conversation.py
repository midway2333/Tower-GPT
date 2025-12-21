import json

def extract_conversation_turns_to_jsonl(
    input_jsonl_path: str,
    output_jsonl_path: str,
    human_field: str = "human",
    assistant_field: str = "assistant"
) -> None:
    """
    从 JSONL 文件中读取多个对话记录（每行一个含 'conversation' 的对象），
    提取每轮 human-assistant 对话，写入新的 JSONL 文件，每行一个对话轮次。

    参数:
        input_jsonl_path: 输入的 JSONL 文件路径（每行是一个完整对话记录）
        output_jsonl_path: 输出的 JSONL 文件路径（每行是一个 {"human": ..., "assistant": ...}）
        human_field: 用户消息字段名，默认 "human"
        assistant_field: 助手回复字段名，默认 "assistant"
    """
    total_turns = 0

    with open(input_jsonl_path, 'r', encoding='utf-8') as in_f, \
         open(output_jsonl_path, 'w', encoding='utf-8') as out_f:

        for line_num, line in enumerate(in_f, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行

            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"⚠️ 第 {line_num} 行 JSON 解析失败: {e}")
                continue

            conversations = record.get("conversation", [])
            if not isinstance(conversations, list):
                print(f"⚠️ 第 {line_num} 行 'conversation' 不是列表，跳过")
                continue

            for turn in conversations:
                if not isinstance(turn, dict):
                    continue

                # 提取并转为字符串（防止 None）
                human_msg = str(turn.get(human_field, "")).strip()
                assistant_msg = str(turn.get(assistant_field, "")).strip()

                # 可选：跳过空对话（根据需求决定是否保留）
                # if not human_msg and not assistant_msg:
                #     continue

                out_turn = {
                    "human": human_msg,
                    "assistant": assistant_msg
                }
                out_f.write(json.dumps(out_turn, ensure_ascii=False) + '\n')
                total_turns += 1

    print(f"✅ 成功处理 {total_turns} 轮对话，已写入 {output_jsonl_path}")

extract_conversation_turns_to_jsonl(
    input_jsonl_path="gen\\common_en_70k.jsonl",
    output_jsonl_path="gen\\common_en.jsonl",
    human_field="human",
    assistant_field="assistant"
)

extract_conversation_turns_to_jsonl(
    input_jsonl_path="gen\\common_chn_70k.jsonl",
    output_jsonl_path="gen\\common_en.jsonl",
    human_field="human",
    assistant_field="assistant"
)
