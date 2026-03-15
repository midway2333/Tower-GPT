import sentencepiece as spm

# 加载训练好的模型
sp = spm.SentencePieceProcessor(model_file='tower_dict_v3.0_7368.model')   # type: ignore

# 测试字符串
test_text = """如果一个电动机在运转时发出嘶嘶声，这可能是由于什么原因？可能是电动机的轴承磨损或齿轮间隙过大导致的。<|im_end|> 给定一段文字，请为标题写一个恰当而吸引人的描述。\n这是一篇介绍天然气的文章。天然气：解密其优点和挑战<|im_end|> 为什么一些元素会自然辐射？这种辐射是由于元素的原子核不稳定，释放出高能粒子或电磁辐射以达到稳定状态。<|im_end|> 谢谢，再将这段文字转换成中文翻译一下。没问题，以下是中文翻译： 今天天气真好，出去散步真是太惬意了。<|im_end|> 请给我5个爬虫类的动物名字。爬虫类的动物很多，以下是5个比较常见的名字：蛇、鳄鱼、蜥蜴、龟、变色龙。<|im_end|> 你好！我想找一本好看的小说，你有什么推荐吗？你好！当然有了。你喜欢什么类型的小说？爱情、悬疑、历史？<|im_end|> 问答系统 - 回答有关特定主题的问题。\n谁是美国第一位女法官？美国第一位女法官是苏珊·贝斯·格里尔。<|im_end|> 请将购物清单发送至我的邮箱。非常好的主意。请问你的邮箱是什么呢？我将会立即将购物清单发送至你的邮箱。<|im_end|>
"""

# 分词（返回 token 列表）
pieces = sp.encode(test_text, out_type=str)   # type: ignore
print("Tokens:", pieces)
print("Number of tokens:", len(pieces))

# 也可以返回 ID
ids = sp.encode(test_text, out_type=int)   # type: ignore
print("IDs:", ids)

# 逆操作：从 ID 或 token 重建文本
reconstructed = sp.decode(pieces)   # type: ignore
print("Reconstructed:", repr(reconstructed))
