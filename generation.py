import torch
import torch.nn.functional as fc
from torch import Tensor
from typing import Optional, List, Tuple, Union
import sentencepiece as spm

class Generation:
    def __init__(self, model: torch.nn.Module, tokenizer_path: str, device: torch.device | str):
        """初始化模型生成器

        参数:
        - model: 模型
        - tokenizer: 用于编码和解码文本的分词器
        - device: 运行模型的设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()

        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(tokenizer_path)
        # 加载 sentencepiece 分词器

        self.bos_id = self.tokenizer.bos_id()
        self.eos_id = self.tokenizer.eos_id()
        self.user_id = self.tokenizer.PieceToId('<user>')
        self.bot_id = self.tokenizer.PieceToId('<bot>')
        # 获取特殊token ID

        self.STOP_TOKENS = [self.eos_id, self.user_id, self.bot_id]
        # 停止标记列表

    def _apply_repetition_penalty(
        self, 
        logits: Tensor, 
        input_ids: Tensor, 
        penalty: float
    ) -> Tensor:
        """应用重复惩罚"""
        if penalty != 1.0:
            for token in torch.unique(input_ids):
                logits[..., token] = torch.where(
                    logits[..., token] > 0,
                    logits[..., token] / penalty,
                    logits[..., token] * penalty,
                )
        return logits

    def _top_k_top_p_filtering(
        self,
        logits: Tensor,
        top_k: int = 0,
        top_p: float = 0.0,
        filter_value: float = -float('inf')
    ) -> Tensor:
        """
        使用 top-k 和 top-p (nucleus sampling) 过滤 logits
        """
        # top-k 过滤
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        # top-p 过滤
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(fc.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            # 移除累积概率超过 top_p 的标记

            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            # 保留第一个超过阈值的标记

            indices_to_remove = sorted_indices_to_remove.scatter(
                1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = filter_value

        return logits

    def _sample_next_token(
        self,
        logits: Tensor,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0
    ) -> Tensor:
        """
        从 logits 中采样下一个 token
        """

        if temperature != 1.0:
            logits = logits / temperature
        # 应用温度

        logits = self._top_k_top_p_filtering(logits, top_k, top_p)
        # 应用 top-k 和 top-p 过滤

        probs = fc.softmax(logits, dim=-1)
        # 转换为概率

        next_token = torch.multinomial(probs, num_samples=1)
        # 采样

        return next_token

    def generate(
        self,
        input_text: list[int],
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0
    ) -> int:
        """
        生成文本
        
        参数:
        - input_text: 输入文本
        - temperature: 温度参数
        - top_k: top-k 采样参数
        - top_p: top-p 采样参数
        - repetition_penalty: 重复惩罚参数
        
        返回:
        - 生成的文本 token ID
        """
        input_ids = torch.tensor(input_text).unsqueeze(0).to(self.device)
        # 准备输入

        with torch.no_grad():
            model_outputs: Tensor = self.model(input_ids)
            logits: Tensor = model_outputs[:, -1, :]
            # 选择最后一个时间步的输出, 形状为 [batch_size, vocab_size]
            # 第一个冒号: 选择所有 batch
            # 第二个冒号: 选择所有 vocab

        logits = self._apply_repetition_penalty(
            logits,
            input_ids,
            repetition_penalty,
        )   # 应用重复惩罚

        output_token_id = self._sample_next_token(
            logits,
            temperature,
            top_k,
            top_p,
        )   # 采样下一个 token

        return output_token_id.item()   # type: ignore

    def check_and_cut_length(
        self,
        input_ids: List[int],
        max_length: int
    ) -> List[int]:
        """
        检查输入长度是否超过最大长度

        参数:
        - input_ids: 输入 token ID 列表
        - max_length: 最大长度

        返回:
        - 截断后的输入 token ID 列表
        """
        if len(input_ids) > max_length:
            input_ids = input_ids[:max_length]

        return input_ids

    def chat(
        self,
        max_length: int = 128,
        max_generate_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0
    ):
        """
        聊天生成文本

        参数:
        - max_length: 最大生成长度
        - temperature: 温度参数
        - top_k: top-k 采样参数
        - top_p: top-p 采样参数
        - repetition_penalty: 重复惩罚参数

        """

        history = [self.bos_id]   # 初始化对话历史

        while True:
            input_text = input('user: ')

            if input_text.lower() in ['exit', 'quit']:
                print("Exiting chat.")
                break

            history.extend(
                  [self.user_id]
                + self.tokenizer.EncodeAsIds(input_text)
                + [self.eos_id]
                + [self.bot_id]
            )   # 添加输入文本

            history = self.check_and_cut_length(history, max_length)
            # 长度检查

            generate_length = 0
            # 初始化生成长度

            print('bot: ', end='')
            # 打印提示符

            while generate_length < max_generate_length:
                output_token_id = self.generate(
                    history,
                    temperature,
                    top_k,
                    top_p,
                    repetition_penalty,
                )   # 生成下一个 token

                if output_token_id in self.STOP_TOKENS:
                    history.append(self.eos_id)
                    generate_length = 0
                    print()   # 换行
                    break

                else:
                    history.append(output_token_id)
                    generate_length += 1
                    # 更新生成长度

                    history = self.check_and_cut_length(history, max_length)
                    # 长度检查

                    word = self.tokenizer.DecodeIds([output_token_id])   # 预测的 token 解码为字词
                    print(word, end='')    # 打印字词

                    if generate_length % 20 == 0:   # 打印换行符
                        print()                     # 方便阅读

    def generate_text(
        self,
        prompt: str,
        max_length: int = 128,
        max_generate_length: int = 64,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 0.0,
        repetition_penalty: float = 1.0
    ) -> str:
        """
        文本续写生成
        
        参数:
        - prompt: 输入提示文本
        - max_length: 最大输入长度
        - max_generate_length: 最大生成长度
        - temperature: 温度参数
        - top_k: top-k 采样参数
        - top_p: top-p 采样参数
        - repetition_penalty: 重复惩罚参数
        
        返回:
        - 生成的文本
        """
        input_ids = self.tokenizer.EncodeAsIds(prompt)
        input_ids = [self.bos_id] + input_ids
        # 将提示文本转换为token

        input_ids = self.check_and_cut_length(input_ids, max_length)
        # 检查并截断输入长度

        generated_tokens = []
        generate_length = 0
        # 初始化生成长度
        
        while generate_length < max_generate_length:
            output_token_id = self.generate(
                input_ids + generated_tokens,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
            )   # 生成下一个 token

            if output_token_id in self.STOP_TOKENS:
                break
            # 检查是否遇到停止token

            generated_tokens.append(output_token_id)
            generate_length += 1
            # 将生成的token添加到列表

        generated_text = self.tokenizer.DecodeIds(generated_tokens)
        # 解码生成的文本

        return generated_text

    def reload_model(self, model: torch.nn.Module):
        """
        重新加载模型
        
        参数:
        - model_path: 模型文件路径
        """
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        # 加载模型权重并设置为评估模式

if __name__ == '__main__':
    from model import Tower_GPT
    
    # 加载模型
    model = Tower_GPT(
        decoder_num=4,
        head_num=4,
        d=1024,
        dk=128,
        dff=4096,
        vocab_size=32768,
        padding_idx=3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        init=False
    )
    
    # 创建生成器
    generator = Generation(
        model=model,
        tokenizer_path='tokenizer\\tower_dict_v2.4_32768.model',  # 替换为你的分词器路径
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # 示例1: 文本续写
    print("=== 文本续写示例 ===")
    result = generator.generate_text(
        prompt="人工智能是",
        max_generate_length=50,
        temperature=0.8,
        top_k=30,
        top_p=0.9
    )
    print(f"输入: 人工智能是")
    print(f"续写: {result}")
    print()
    
    # 示例2: 启动聊天模式
    print("=== 启动聊天模式 ===")
    generator.chat(
        max_length=256,
        max_generate_length=128,
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.1
    )
