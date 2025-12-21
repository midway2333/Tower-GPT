def clean_text_file(input_path, output_path=None):
    """
    清理文本文件：
    - 删除所有空行
    - 去除每行开头的空白字符（保留行内空格）
    
    参数:
        input_path (str): 输入文件路径
        output_path (str): 输出文件路径。若为 None，则覆盖原文件
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    cleaned_lines = []
    for line in lines:
        # 去除行首空白（rstrip 不动，因为要保留行尾换行符由 writelines 处理）
        stripped_line = line.lstrip()
        # 如果去除开头空白后不是空行，则保留
        if stripped_line:  # 非空行
            cleaned_lines.append(stripped_line)

    # 决定输出路径
    out_path = output_path if output_path else input_path

    with open(out_path, 'w', encoding='utf-8') as f:
        f.writelines(cleaned_lines)

# ===== 使用示例 =====
if __name__ == "__main__":
    # 修改这里的文件路径为你自己的
    input_file = "merged_output.txt"       # 原始文件
    output_file = "corpus.txt"     # 可选：设为 None 则覆盖原文件

    clean_text_file(input_file, output_file)
    print(f"文件已清理完成！输出路径: {output_file if output_file else input_file}")
    