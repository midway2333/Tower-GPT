import sentencepiece as spm
import os

if __name__ == '__main__':
    # 定义语料文件路径
    corpus = 'data\\pretrain_hq.txt,code_2mb_simplified.txt'

    # 定义分词器的模型名称
    model_prefix = "tower_dict_v3.0_7368"

    # 训练分词器
    spm.SentencePieceTrainer.train(     # type: ignore
        input=corpus,                   # 输入语料文件
        model_prefix=model_prefix,      # 输出模型前缀
        vocab_size=7368,                # 词汇表大小
        model_type='bpe',               # 分词模型类型
        max_sentencepiece_length=16,    # 最长单分词字数
        input_format="text",            # 输入纯文本文件
        max_sentence_length=4192,       # 训练时最大允许的句子长度
        byte_fallback=True,             # 启用字节回退机制
        character_coverage=0.999,       # 字符覆盖率
        split_digits=True,              # 将数字拆分为单个token
        num_threads=os.cpu_count(),     # 使用的线程数
        self_test_sample_size=0,        # 自测样本的大小
        unk_id=0,                       # 设置unk_id
        bos_id=1,                       # 设置bos_id
        eos_id=2,                       # 设置eos_id
        pad_id=3,                       # 设置pad_id
        bos_piece="<|im_start|>",       # 设置bos_token
        eos_piece="<|im_end|>",         # 设置eos_token
        
        user_defined_symbols=[    # 自定义标记
            '<user>', '<bot>', '<system>',

            # 思考标记
            '<think>', '</think>',
            '<brief_think>', '</brief_think>',

            # 以下为 Qwen3 Omni 模型的特殊标记, 可能在训练中不完全适用, 但保留以防未来使用
            '<|endoftext|>',
            '<|object_ref_start|>',
            '<|object_ref_end|>',
            '<|box_start|>',
            '<|box_end|>',
            '<|quad_start|>',
            '<|quad_end|>',
            '<|vision_start|>',
            '<|vision_end|>',
            '<|vision_pad|>',
            '<|image_pad|>',
            '<|video_pad|>',
            '<tool_call>',
            '</tool_call>',
            '<|fim_prefix|>',
            '<|fim_middle|>',
            '<|fim_suffix|>',
            '<|fim_pad|>',
            '<|repo_name|>',
            '<|file_sep|>',
            '<tool_response>',
            '</tool_response>',
            '<|audio_start|>',
            '<|audio_end|>',
            '<tts_pad>',
            '<tts_text_bos>',
            '<tts_text_eod>',
            '<tts_text_bos_single>',
            '<|audio_pad|>',

            # 情感符号
            '😀','😃','😄','😅','😂','😗','😙','😚','😊','🙃',
            '🙂','😑','😐','🫢','🤭','🥱','🤗','🫣','😱','🤨',
            '🧐','😒','😧','😟','🙄','😥','😦','😮‍💨','😢','😮',
            '😤','☹️','😯','😠','🙁','😲','😡','🫤','😳','🤬',
            '😕','🤯','🥵','🤢','🤮','😖','😣','😰','😨','😞',
            '😓','😋','✌️','🫲','🫱','💪','👏','👍','👎','🤜',
            '🤛','✊','👊','🥰','😍','🤪','🤓','🤖','👾','👻',
            '🙌','👋','✋','🤲','🤝','🙏','💢','💥','💯','❤️',
            '🐶','🐱','🐭','🐹','🐰','🦊','🐻','🐼','🐨','🐯',
            '🦁','🐮','🐷','🐽','🐸','🐵','🐔','🐧','🐦','🍎',
            '🍌','🍉','🍓','🍒','🍇','🍊','🍋','🍈','🥭','🥥',
            '🥝','🍍','🍠','🍆','🥕','🥦','💔','💌','💕','💞',
            '💓','💗','💖','💘','💝','⚠️','✅','❌','❓','⭐',

            # 特殊词汇
            'Tower','tower','GPT','Midway', '\n', '\t'
        ]
        # 添加格外分词
    )

    print(f"模型已训练完成,并保存为 {model_prefix}.model和{model_prefix}.vocab")

