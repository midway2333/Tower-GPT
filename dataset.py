from torch.utils.data import Dataset, IterableDataset
from numpy.random import shuffle
import random, torch, json
import sentencepiece as spm
from typing import Optional, Tuple


class TextDataProcessor:
    def __init__(self, json_file, sp_model_path, block_size, buffer_size=32768, field='text'):
        """
        初始化 TextDataProcessor 类的实例

        参数:
        - json_file (str): 包含对话数据的 JSON 文件路径
        - sp_model_path (str): SentencePiece 模型文件路径
        - block_size (int): 单个输入中的最大 token 数量
        - buffer_size (int): 缓冲区大小, 默认为 32768
        - field (str): 对话数据中包含输入文本的字段名, 默认值为'text'
        """

        self.json_file = json_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)   # type: ignore
        self.block_size = block_size
        self.padding_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.buffer_size: int = buffer_size   # 缓冲区大小
        self.field: str = field   # 对话数据中包含输入文本的字段名

    def load_and_encode_data(self):
        """
        加载并编码对话数据
        使用 jsonl 文件

        返回:
        - inputs (list): 编码后的用户输入列表
        - targets (list): 编码后的助手响应列表
        """

        inputs = []
        targets = []

        with open(self.json_file, 'r', encoding='utf-8') as f:   # 打开 jsonl 文件
            for line in f:
                line = line.strip()
                if not line:   # 跳过空行
                    continue

                dialogue = json.loads(line)
                input_text = dialogue[self.field]   # 假设每行都有 'text' 字段

                input_ids = self.sp.encode(input_text, out_type=int)   # type: ignore
                input_ids = [self.bos_id] + input_ids + [self.eos_id]

                if len(input_ids) > self.block_size:
                    i = random.randint(0, len(input_ids) - self.block_size - 1)
                    x_data = input_ids[i:i + self.block_size]
                    y_data = input_ids[i + 1:i + 1 + self.block_size]
                    # 随机截取一个长度为 block_size 的片段

                else:   # 短序列: 填充到 block_size
                    x_data = input_ids[:-1] + [self.padding_id] * (self.block_size - len(input_ids) + 1)
                    y_data = input_ids[1:] + [self.padding_id] * (self.block_size - len(input_ids) + 1)

                x_data = torch.tensor(x_data, dtype=torch.int32)
                y_data = torch.tensor(y_data, dtype=torch.int32)

                inputs.append(x_data)
                targets.append(y_data)

        return inputs, targets

    def data_generator(self):

        """
        使用生成器加载并编码对话数据,适用于大数据集加载
        使用 jsonl 文件

        返回:
        - inputs (list): 编码后的用户输入列表
        - targets (list): 编码后的助手响应列表
        - None: 占位符, 用于与对话数据加载模式保持一致
        """

        with open(self.json_file, 'r', encoding='utf-8') as f:
            buffer_list = []   # 缓冲区

            for line in f:   # 加载jsonl文件
                dialogue = json.loads(line.strip())

                input = dialogue[self.field]   # 获取用户输入文本
                input_ids = self.sp.encode(input, out_type=int)   # type: ignore
                input_ids = [self.bos_id] + input_ids + [self.eos_id]

                if len(input_ids) > self.block_size:   # 随机选择一个起始索引

                    i = random.randint(0, len(input_ids) - self.block_size - 1)
                    x_data = input_ids[i:i+self.block_size]
                    y_data = input_ids[i+1:i+1+self.block_size]

                    x_data = torch.tensor(x_data, dtype=torch.int32)
                    y_data = torch.tensor(y_data, dtype=torch.int32)
                    # 将编码信息转换为tensor    

                else:   # 短序列: 填充
                    x_data = input_ids[:-1] + [self.padding_id] * (self.block_size - len(input_ids) + 1)
                    y_data = input_ids[1:] + [self.padding_id] * (self.block_size - len(input_ids) + 1)

                    x_data = torch.tensor(x_data, dtype=torch.int32)
                    y_data = torch.tensor(y_data, dtype=torch.int32)
                    # 将编码信息转换为tensor

                buffer_list.append((x_data, y_data))   # 添加到缓冲区
                
                # 检查缓冲区是否已满
                if len(buffer_list) >= self.buffer_size:
                    random.shuffle(buffer_list)
                    for inputs, targets in buffer_list:
                        yield inputs, targets, None
                    buffer_list = []   # 清空缓冲区
            
            # 处理剩余的缓冲区数据
            if buffer_list:
                random.shuffle(buffer_list)
                for inputs, targets in buffer_list:
                    yield inputs, targets, None
    
    def data_length(self):
        """返回数据集的长度"""
        with open(self.json_file, 'r', encoding='utf-8') as file:
            return sum(1 for line in file)


class TextDataset(Dataset):   # 负责加载和编码数据的实例
    def __init__(self, processor: TextDataProcessor):
        """初始化 TextDataset 类的实例"""
        self.inputs, self.targets = processor.load_and_encode_data()

    def __len__(self):   # 返回数据集的大小
        return len(self.inputs)

    def __getitem__(self, idx):   # 根据索引获取数据集中的样本
        return self.inputs[idx], self.targets[idx], None

class GeneratorTextDataset(IterableDataset):   # 负责生成器模式下加载和编码数据的实例
    def __init__(self, processor: TextDataProcessor):
        """初始化 GeneratorTextDataset 类的实例"""
        super().__init__()
        self.processor = processor

    def __iter__(self):   # 返回一个迭代器对象,每次迭代时从生成器中获取下一个样本
        return iter(self.processor.data_generator())
    
def text_collate_fn(batch):
    """
    处理 (x, y, None) 的 batch
    返回:
    - (x_stacked, y_stacked, None)
    """
    xs, ys, _ = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.stack(ys)
    # 堆叠 x 和 y

    return xs, ys, None

""" ------------------------------------- 以上对话文本dataset ------------------------------------- """

""" ------------------------------------- 以下多轮对话dataset ------------------------------------- """

class MultiTurn_DialogueDataProcessor:
    """多轮对话数据处理器"""
    def __init__(self, json_file, sp_model_path, block_size, buffer_size=32768):
        """
        初始化 MultiTurn_DialogueDataProcessor 类的实例, 接受标准格式的多轮对话数据

        接受格式:
        - {"messages": [{"role": "user", "content": "用户输入"}, {"role": "assistant", "content": "助手回复"}]}

        字段:
        - messages (list): 包含多轮对话消息的列表,每个消息是一个字典,包含"role"和"content"键

        参数:
        - json_file (str): 包含对话数据的 jsonl 文件路径
        - sp_model_path (str): SentencePiece 模型文件路径
        - block_size (int): 单个输入中的最大 token 数量
        - buffer_size (int): 缓冲区大小, 用于批量加载数据, 默认值为 32768
        """

        self.json_file = json_file
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(sp_model_path)    # type: ignore
        self.block_size = block_size
        self.padding_id = self.sp.pad_id()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        self.buffer_size: int = buffer_size   # 添加缓冲区大小

        self.user_id = [self.sp.PieceToId('<user>')]
        self.bot_id = [self.sp.PieceToId('<bot>')]
        # 获得user_id与bot_id

    def _process_single_dialogue(self, record: dict) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """处理单个多轮对话数据"""
        messages = record.get("messages")

        if not isinstance(messages, list) or len(messages) == 0:
            return None   # 输入验证检查

        token_ids = [self.bos_id]   # 整个序列开头只加一次 <s>
        loss_mask = [0]             # 损失掩码, 0表示忽略loss, 1表示计算loss

        for msg in messages:   # 检查消息结构
            if not (isinstance(msg, dict) and "role" in msg and "content" in msg):
                continue

            role = msg["role"]
            content = msg["content"]
            # 提取数据

            if not isinstance(content, str):   # 检查内容类型
                continue

            if role == "user":
                token_ids += self.user_id + self.sp.encode(content, out_type=int) + [self.eos_id]   # type: ignore
                loss_mask += [0] * (len(self.user_id) + len(self.sp.encode(content, out_type=int)) + 1)    # type: ignore

            elif role == "assistant":
                token_ids += self.bot_id + self.sp.encode(content, out_type=int) + [self.eos_id]   # type: ignore
                loss_mask += [1] * (len(self.bot_id) + len(self.sp.encode(content, out_type=int)) + 1)   # type: ignore
                # 助手回复部分计算loss

        if len(token_ids) < 2:   # 至少要有 <s> + 一个 token
            return None

        if len(token_ids) > self.block_size:
            input_ids = token_ids[:self.block_size]
            target_ids = token_ids[1:self.block_size+1]
            loss_mask = loss_mask[1:self.block_size+1]   # 目标序列对应的掩码
            # 构建 input / target

        else:   # 短序列: 填充
            input_ids = token_ids[:-1] + [self.padding_id] * (self.block_size - len(token_ids) + 1)
            target_ids = token_ids[1:] + [self.padding_id] * (self.block_size - len(token_ids) + 1)
            loss_mask = loss_mask[1:] + [0] * (self.block_size - len(token_ids) + 1)  # 填充部分不计算loss

        return (
            torch.tensor(input_ids, dtype=torch.int32), 
            torch.tensor(target_ids, dtype=torch.int32),
            torch.tensor(loss_mask, dtype=torch.int32)  # 返回损失掩码
        )

    def load_and_encode_data(self):
        """小数据集, 一次性加载全部"""
        inputs, targets, loss_masks = [], [], []
        with open(self.json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                record = json.loads(line)
            # 读取jsonl文件

                result = self._process_single_dialogue(record)
                # 处理单轮对话数据
                
                if result:
                    inputs.append(result[0])
                    targets.append(result[1])
                    loss_masks.append(result[2])

        return inputs, targets, loss_masks

    def data_generator(self):
        """用于大数据集：生成器 + 缓冲打乱"""
        buffer = []
        with open(self.json_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                record = json.loads(line)
                # 读取jsonl文件

                result = self._process_single_dialogue(record)
                # 处理单轮对话数据

                if not result:
                    continue

                if len(buffer) < self.buffer_size:
                    buffer.append(result)
                # 缓冲区未满, 继续添加

                else:
                    shuffle(buffer)
                    yield from buffer
                    buffer = [result]
                # 缓冲区满, 打乱并 yield, 清空

        if buffer:
            shuffle(buffer)
            yield from buffer
        # 处理剩余数据

    def data_length(self):
        """返回文件行数"""
        with open(self.json_file, 'r', encoding='utf-8') as f:
            return sum(1 for line in f if line.strip())


class Talk_DialogueDataset(Dataset):
    """用于小数据集的数据加载器"""
    def __init__(self, processor: MultiTurn_DialogueDataProcessor):
        self.inputs, self.targets, self.loss_masks = processor.load_and_encode_data()
    
    def __len__(self):   # 返回数据集的大小
        return len(self.inputs)
    
    def __getitem__(self, idx):   # 根据索引获取数据集中的样本
        return self.inputs[idx], self.targets[idx], self.loss_masks[idx]

class Talk_GeneratorDialogueDataset(IterableDataset):
    """用于大数据集的数据加载器"""
    def __init__(self, processor: MultiTurn_DialogueDataProcessor):
        super().__init__()
        self.processor = processor

    def __iter__(self):   # 返回一个迭代器对象,每次迭代时从生成器中获取下一个样本
        return iter(self.processor.data_generator())
