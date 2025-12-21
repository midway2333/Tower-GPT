import sentencepiece as spm

# 加载训练好的模型
sp = spm.SentencePieceProcessor(model_file='tokenizer/tower_dict_v2.4_32768.model')   # type: ignore

# 测试字符串
test_text = """
\n头发
"""

# 分词（返回 token 列表）
pieces = sp.encode(test_text, out_type=str)   # type: ignore
print("Tokens:", pieces)

# 也可以返回 ID
ids = sp.encode(test_text, out_type=int)   # type: ignore
print("IDs:", ids)

# 逆操作：从 ID 或 token 重建文本
reconstructed = sp.decode(pieces)   # type: ignore
print("Reconstructed:", repr(reconstructed))