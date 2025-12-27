from dataclasses import dataclass
from typing import Literal

@dataclass
class TrainerConfig():
    """训练器配置
    Attributes:
        decoder_num (int): 解码器数量
        head_num (int): 注意力头数
        d (int): 隐藏层维度
        dk (int): KV 维度
        dff (int): 前馈网络维度
        vocab_size (int): 词表大小
        dropout (float): dropout 概率

        train_method (str): 训练方式, 可选值为 "text" 或 "chat"
        keep_train (bool): 是否从检查点续训练
        ckpt_path (str | None): 检查点路径, 可选值为 None 或 检查点路径
        finetune (bool): 是否微调模型
        compile (bool): 是否使用 torch.compile 编译模型加速训练
        load_optimizer (bool): 是否加载优化器, 用于续训练

        device (str): 加载设备, 可选值为 "cpu" "cuda" "mps" "xpu" 或 "auto"
        mixed_precision (str): 混合精度训练, 可选值为 "full" "fp16" "bf16"

        train_model_dir (str | None): 预训练模型目录, None 表示不使用预训练模型
        train_model_name (str | None): 预训练模型名, None 表示不使用预训练模型
        output_dir (str | None): 输出目录, 用于保存模型和日志
        output_model_name (str | None): 输出模型名
        model_suffix (str): 模型文件后缀名
        optimizer_suffix (str): 优化器文件后缀名
        scheduler_suffix (str | None): 调度器文件后缀名
        max_checkpoints (int): 最大保存检查点数量; 按时间排序, 保留最新的 `max_checkpoints` 个检查点
        save_best_checkpoint (bool): 是否保存最佳检查点, 仅存在验证集时有效, 与 `max_checkpoints` 不冲突

        train_data_path (str): 训练数据路径
        valid_data_path (str | None): 验证数据路径, 可选值为 None 或 验证数据路径
        test_data_path (str | None): 测试数据路径, 可选值为 None 或 测试数据路径
        tokenizer_path (str): 分词器路径
        num_workers (int): 数据加载器工作线程数
        pin_memory (bool): 是否将数据加载到 CUDA 固定内存中, 加速数据传输
        yield_load (bool): 是否使用 yield 加载数据, 用于处理大文件, 此时 `num_workers` 与 `pin_memory` 无效

        all_epochs (int): 总训练轮数
        batch_size (int): 批次大小
        block_size (int): 输入序列长度
        accumulation_steps (int): 梯度累计步数
        info_update_interval (int): 信息更新间隔, 单位为 backward 步数; 此时会验证并保存模型, 记录日志

        optimizer (str): 优化器, 可选值为 "adamw" "adafactor" "galore_adamw" 或 "galore_adafactor"
        learning_rate (float): 学习率
        betas (tuple[float, float] | float): beta 参数, adamw 下为 tuple[float, float], adafactor 下为 float
        eps (float | tuple[float, float]): 优化器 epsilon 参数, 用于训练稳定性; adamw 下为 float, adafactor 下为 tuple[float, float]
        weight_decay (float): 权重衰减系数, 正则化项, 防止过拟合
        lr_scheduler (bool): 是否使用学习率调度器
        pct_start (float): 学习率调度器的预热比例
        max_lr_rate (float): 学习率调度器的最大学习率倍率
        div_factor (float): 学习率调度器的初始学习率倍率 (初始学习率 = 最大学习率 / div_factor)
        anneal_strategy (str): 学习率调度器的退火策略, 可选值为 "linear" 或 "cos"

        grad_clip (float | None): 梯度裁剪值, 可选值为 None 或 梯度裁剪值
        dropout (float):  dropout 概率, 正则化选项

        ppl_eval (bool): 是否在验证集上评估困惑度
        bleu_eval (bool): 是否在验证集上评估 BLEU-4 分数

        tensorboard (bool): 是否使用 tensorboard 可视化训练过程
        tensorboard_dir (str | None): tensorboard 日志目录, 可选值为 None 或 tensorboard 日志目录
        writer_name (str | None): tensorboard 日志文件名, 可选值为 None 或 tensorboard 日志文件名

        logger_name (str): 日志名称, 用于区分不同模块
        level (str): 日志级别, 可选值为 "debug" "info" "warning" 或 "error"
        std_out (bool): 是否输出到控制台, 默认为 True
        save_info (bool): 是否保存日志到文件, 默认为 True
        file_name (str | None): 日志文件名, 默认为 None, 此时使用日期记录日志
    """
    # 模型配置
    decoder_num: int = 6
    """解码器数量"""
    head_num: int = 6
    """注意力头数"""
    d: int = 384
    """隐藏层维度"""
    dk: int = 64
    """KV 维度"""
    dff: int = 1536
    """前馈网络维度"""
    vocab_size: int = 32768
    """词表大小"""

    # 训练器配置
    train_method: Literal["text", "chat"] = "text"
    """训练方式, 可选值为 "text" 或 "chat" """
    keep_train: bool = False
    """是否从检查点续训练"""
    ckpt_path: str | None = None
    """检查点路径, 可选值为 None 或 检查点路径"""
    finetune: bool = False
    """是否微调模型"""
    compile: bool = False
    """是否使用 torch.compile 编译模型加速训练"""
    load_optimizer: bool = False
    """是否加载优化器, 用于续训练"""

    # 设备配置
    device: Literal["cpu", "cuda", "xpu", "mps","auto"] = "auto"
    """加载设备, 可选值为 "cpu" "cuda" "xpu" "mps" 或 "auto" """
    mixed_precision: Literal["full", "fp16", "bf16"] = "full"
    """混合精度训练, 可选值为 "full" "fp16" "bf16" """

    # 模型配置
    train_model_dir: str | None = None
    """预训练模型目录, None 表示不使用预训练模型"""
    train_model_name: str | None = None
    """预训练模型名, None 表示不使用预训练模型"""
    output_dir: str = "./output"
    """输出目录, 用于保存模型和日志"""
    output_model_name: str = "llm_model"
    """输出模型名"""
    model_suffix: str = ".bin"
    """模型文件后缀名"""
    optimizer_suffix: str = ".pt"
    """优化器文件后缀名"""
    scheduler_suffix: str | None = ".sdl"
    """调度器文件后缀名"""
    max_checkpoints: int = 3
    """最大保存检查点数量; 按时间排序, 保留最新的 `max_checkpoints` 个检查点"""
    save_best_checkpoint: bool = True
    """是否保存最佳检查点, 仅存在验证集时有效, 与 `max_checkpoints` 不冲突"""

    # 数据集配置
    train_data_path: str = ""
    """训练数据路径"""
    valid_data_path: str | None = None
    """验证数据路径, 可选值为 None 或 验证数据路径"""
    test_data_path: str | None = None
    """测试数据路径, 可选值为 None 或 测试数据路径"""
    tokenizer_path: str = ""
    """分词器路径"""
    num_workers: int = 4
    """数据加载器工作线程数"""
    pin_memory: bool = True
    """是否将数据加载到 CUDA 固定内存中, 加速数据传输"""
    yield_load: bool = False
    """是否使用 yield 加载数据, 用于处理大文件, 此时 `num_workers` 与 `pin_memory` 无效"""

    # 训练参数配置
    all_epochs: int = 2
    """总训练轮数"""
    batch_size: int = 16
    """批次大小"""
    block_size: int = 512
    """输入序列长度"""
    accumulation_steps: int = 8
    """梯度累计步数"""
    info_update_interval: int = 2048
    """信息更新间隔, 单位为 backward 步数; 此时会验证并保存模型, 记录日志"""

    # 优化器配置
    optimizer: Literal["adamw", "adafactor", "galore_adamw", "galore_adafactor"] = "adamw"
    """优化器, 可选值为 "adamw" "adafactor" "galore_adamw" 或 "galore_adafactor" """
    learning_rate: float = 1e-4
    """学习率"""
    betas: tuple[float, float] | float = (0.9, 0.999)
    """beta 参数, adamw 下为 tuple[float, float], adafactor 下为 float"""
    eps: float | tuple[float, float] = 1e-8
    """优化器 epsilon 参数, 用于训练稳定性; adamw 下为 float, adafactor 下为 tuple[float, float]"""
    weight_decay: float = 0.001
    """权重衰减系数, 正则化项, 防止过拟合"""
    lr_scheduler: bool = True
    """是否使用学习率调度器"""
    pct_start: float | None = 0.1
    """学习率调度器的预热比例"""
    max_lr_rate: float | None = 35.0
    """学习率调度器的最大学习率倍率"""
    div_factor: float | None = 35.0
    """学习率调度器的初始学习率倍率 (初始学习率 = 最大学习率 / div_factor)"""
    anneal_strategy: Literal["cos", "linear"] | None = "cos"
    """学习率调度器的退火策略, 可选值为 "linear" 或 "cos" """

    # 模型技术参数配置
    grad_clip: float | None = None
    """梯度裁剪值, 可选值为 None 或 梯度裁剪值"""
    grad_checkpoint: bool = False
    """是否使用梯度检查点技术节省显存"""
    dropout: float = 0.1
    """dropout 概率, 正则化选项"""

    # 评估配置
    ppl_eval: bool = True
    """是否在验证集上评估困惑度"""
    bleu_eval: bool = True
    """是否在验证集上评估 BLEU-4 分数"""

    # 可视化配置
    tensorboard: bool = True
    """是否使用 tensorboard 可视化训练过程"""
    tensorboard_dir: str | None = None
    """tensorboard 日志目录, 可选值为 None 或 tensorboard 日志目录"""
    writer_name: str | None = None
    """tensorboard 日志文件名, 可选值为 None 或 tensorboard 日志文件名"""

    # 日志配置
    logger_name: str = "trainer"
    """日志名称, 用于区分不同模块"""
    std_level: Literal["debug", "info", "warning", "error"] = "info"
    """控制台输出日志级别, 可选值为 "debug" "info" "warning" 或 "error" """
    file_level: Literal["debug", "info", "warning", "error"] = "debug"
    """文件输出日志级别, 可选值为 "debug" "info" "warning" 或 "error" """
    std_out: bool = True
    """是否输出到控制台, 默认为 True"""
    save_info: bool = False
    """是否保存日志到文件, 默认为 False"""
    file_name: str | None = None
    """日志文件名, 默认为 None, 此时使用日期记录"""
