# 基本库
import torch
import json
import numpy as np
import os, shutil
import math
import copy

# 用于可视化的库
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn, TimeElapsedColumn
from torch.utils.tensorboard import SummaryWriter   # type: ignore

# tokenizer
import sentencepiece as spm
from transformers import AutoTokenizer

# 优化器
from torch.optim.adamw import AdamW
from transformers import Adafactor
from galore_torch import GaLoreAdamW8bit
from galore_torch import GaLoreAdafactor

# 模块库
from torch import GradScaler, autocast   # type: ignore
from torch.utils.data import DataLoader
from torch.nn import functional as fc
from torch import nn, Tensor

# 其他库
from datetime import datetime
from typing import Optional, Union
import signal
import sys

# Dataset & DataLoader
from dataset import TextDataProcessor
from dataset import TextDataset, GeneratorTextDataset
from dataset import MultiTurn_DialogueDataProcessor
from dataset import Talk_DialogueDataset, Talk_GeneratorDialogueDataset
from dataset import RLDataProcessor
from dataset import RLDataset, RLGeneratorDataset

from dataset import text_collate_fn
from dataset import rl_collate_fn

# 模型类
from re_model import Tower_GPT

# 配置文件
from train_config import TrainerConfig

# 日志类
import logging
from log_utils import TrainLogger


class Trainer():
    def __init__(self, config: TrainerConfig):
        """初始化训练器

        参数:
        - config (TrainerConfig): 训练配置
        """
        # 模型配置
        self.decoder_num = config.decoder_num
        """解码器数量"""
        self.head_num = config.head_num
        """注意力头数"""
        self.d = config.d
        """隐藏层维度"""
        self.dk = config.dk
        """KV 维度"""
        self.dff = config.dff
        """前馈网络维度"""
        self.vocab_size = config.vocab_size
        """词表大小"""

        # 训练器配置
        self.train_method = config.train_method
        """训练方式"""
        self.keep_train = config.keep_train
        """是否从最近检查点续训练"""
        self.ckpt_path = config.ckpt_path
        """检查点路径"""
        self.finetune = config.finetune
        """是否微调模型"""
        self.compile = config.compile
        """是否使用 torch.compile 编译模型加速训练"""
        self.load_optimizer = config.load_optimizer
        """是否加载优化器"""

        # 设备配置
        self.device = config.device
        """加载设备"""
        self.mixed_precision = config.mixed_precision
        """混合精度训练方式"""

        # 模型配置
        self.train_model_dir = config.train_model_dir
        """预训练模型目录"""
        self.train_model_name = config.train_model_name
        """预训练模型名"""
        self.output_dir = config.output_dir
        """输出目录"""
        self.output_model_name = config.output_model_name
        """输出模型名"""
        self.model_suffix = config.model_suffix
        """模型文件后缀名"""
        self.optimizer_suffix = config.optimizer_suffix
        """优化器文件后缀名"""
        self.scheduler_suffix = config.scheduler_suffix
        """调度器文件后缀名"""
        self.max_checkpoints = config.max_checkpoints
        """最大保存检查点数量"""
        self.save_best_checkpoint = config.save_best_checkpoint
        """是否保存最佳检查点"""

        # 数据集配置
        self.train_data_path = config.train_data_path
        """训练数据路径"""
        self.valid_data_path = config.valid_data_path
        """验证数据路径"""
        self.test_data_path = config.test_data_path
        """测试数据路径"""
        self.tokenizer_path = config.tokenizer_path
        """分词器路径"""
        self.num_workers = config.num_workers
        """数据加载器工作线程数"""
        self.pin_memory = config.pin_memory
        """是否将数据加载到 CUDA 固定内存中"""
        self.yield_load = config.yield_load
        """是否使用 yield 加载数据"""

        # 训练参数配置
        self.all_epochs = config.all_epochs
        """总训练轮数"""
        self.batch_size = config.batch_size
        """批次大小"""
        self.block_size = config.block_size
        """输入序列长度"""
        self.accumulation_steps = config.accumulation_steps
        """梯度累计步数"""
        self.info_update_interval = config.info_update_interval
        """信息更新间隔"""

        # 优化器配置
        self.optimizer_name = config.optimizer
        """优化器"""
        self.learning_rate = config.learning_rate
        """学习率"""
        self.betas = config.betas
        """beta 参数"""
        self.eps = config.eps
        """优化器 epsilon 参数"""
        self.weight_decay = config.weight_decay
        """权重衰减系数"""
        self.lr_scheduler = config.lr_scheduler
        """是否使用学习率调度器"""
        self.pct_start = config.pct_start
        """学习率调度器的预热比例"""
        self.max_lr_rate = config.max_lr_rate
        """学习率调度器的最大学习率倍率"""
        self.div_factor = config.div_factor
        """学习率调度器的初始学习率倍率"""
        self.anneal_strategy = config.anneal_strategy
        """学习率调度器的退火策略"""

        # 模型技术参数配置
        self.grad_clip = config.grad_clip
        """梯度裁剪值"""
        self.grad_checkpoint = config.grad_checkpoint
        """是否使用梯度检查点技术节省显存"""
        self.dropout = config.dropout
        """dropout 概率"""

        # RL 训练配置
        self.rl_beta = config.rl_beta
        """RL 训练的超参数 beta"""
        self.gamma = config.gamma
        """SimPO 训练的超参数 gamma"""

        # 评估配置
        self.ppl_eval = config.ppl_eval
        """是否在验证集上评估困惑度"""
        self.bleu_eval = config.bleu_eval
        """是否在验证集上评估 BLEU-4 分数"""

        # 可视化配置
        self.tensorboard = config.tensorboard
        """是否使用 tensorboard 可视化训练过程"""
        self.tensorboard_dir = config.tensorboard_dir
        """tensorboard 日志目录"""
        self.writer_name = config.writer_name
        """tensorboard 日志文件名"""

        # 日志配置
        self.logger_name = config.logger_name
        """日志名称"""
        self.std_level = config.std_level
        """控制台输出日志级别"""
        self.file_level = config.file_level
        """文件输出日志级别"""
        self.std_out = config.std_out
        """是否输出到控制台"""
        self.save_info = config.save_info
        """是否保存日志到文件"""
        self.file_name = config.file_name
        """日志文件名"""

        self._build_logger()   # 构建日志记录器
        self._device_check()   # 检查设备信息

        self.best_blue: float = -float("inf")
        """最佳 BLUE 分数"""
        self.best_loss: float = float("inf")
        """最佳损失值"""

        self.train_signal = False
        """用于判断模型是否进入训练流程"""

        sp = spm.SentencePieceProcessor()
        self.tokenizer = sp.Load(self.tokenizer_path)
        """分词器"""
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=sp.pad_id(), reduction='none')
        """损失函数"""
        self.scaler = GradScaler() if self.mixed_precision == "fp16" else None
        """混合精度训练的梯度缩放器"""

        self.now_epoch = 0
        """当前轮数, 从 0 开始计数"""
        self.train_steps = 0
        """当前训练步数, 每次梯度更新时增加"""
        self.local_steps = 0
        """当前数据加载步数"""
        self.info_steps = 0
        """当前信息更新步数"""
        self.skip_steps = 0
        """需要跳过的批次数量"""
        self.need_skip = False
        """是否需要跳过批次"""

        self.print_info()           # 打印训练配置信息
        self._init_model()          # 初始化模型
        self._init_tensorboard()    # 初始化 TensorBoard
        self._init_optimizer()      # 初始化优化器
        self._init_dataloader()     # 初始化数据加载器
        self._init_lr_scheduler()   # 初始化学习率调度器

        if self.train_model_dir and self.train_model_name is not None and self.keep_train is not True:
            self._load_from_base_model()   # 加载基础模型

        if self.keep_train:
            self._load_resume_page()   # 加载恢复页面
            self._load_from_resume()   # 从恢复页面加载模型

        self._is_finetune()   # 微调判定
        # load_state_dict 会使 FT 失效, 这里最后加载 FT 状态

        signal.signal(signal.SIGINT, self.exit)

    def _build_logger(self):
        """构建日志记录器"""
        assert self.std_level in ["debug", "info", "warning", "error"], f"日志级别必须是 'debug', 'info', 'warning', 或 'error', 但得到 {self.std_level}"
        assert self.file_level in ["debug", "info", "warning", "error"], f"日志级别必须是 'debug', 'info', 'warning', 或 'error', 但得到 {self.file_level}"

        # level = logging._nameToLevel[self.level.upper()]
        LEVEL_MAP = {
            "debug": logging.DEBUG,
            "info": logging.INFO,
            "warning": logging.WARNING,
            "error": logging.ERROR,
        }
        std_level = LEVEL_MAP[self.std_level]
        file_level = LEVEL_MAP[self.file_level]
        # 映射 log 级别

        self.logger = TrainLogger(
            logger_name=self.logger_name,
            std_level=std_level,
            file_level=file_level,
            std_out=self.std_out,
            file_out=self.save_info,
            output_dir=self.output_dir,
            file_name=self.file_name,
        )   # 构建日志记录器

    def _device_check(self):
        """检查设备信息"""
        assert self.device in ["cpu", "cuda", "xpu", "mps","auto"], f"设备必须是 'cpu', 'cuda', 'xpu', 'mps' 或 'auto', 但得到 {self.device}"
        # 断言设备信息无误

        if self.device == "cuda":   # 手动选择 CUDA
            if torch.cuda.is_available():
                self.logger.info(f"Manual: 成功加载 Nvidia CUDA")

            else:
                self.logger.error("Manual: Nvidia CUDA 加载失败; 请检查设备")
                raise RuntimeError("Nvidia CUDA 加载失败; 请检查设备")
            
        if self.device == "xpu":   # 手动选择 XPU
            if torch.xpu.is_available():
                self.logger.info(f"Manual: 成功加载 Intel XPU")

            else:
                self.logger.error("Manual: Intel XPU 加载失败; 请检查设备")
                raise RuntimeError("Intel XPU 加载失败; 请检查设备")
            
        if self.device == "mps":   # 手动选择 MPS
            if torch.backends.mps.is_available():
                self.logger.info(f"Manual: 成功加载 Apple MPS")

            else:
                self.logger.error("Manual: Apple MPS 加载失败; 请检查设备")
                raise RuntimeError("Apple MPS 加载失败; 请检查设备")
            
        if self.device == "cpu":   # 手动选择 CPU
            if torch.cpu.is_available():
                self.logger.info(f"Manual: 成功加载 CPU")
                self.logger.warning("Manual: 手动选择 CPU; 训练可能受影响")

            else:
                self.logger.error("Manual: CPU 加载失败; 请检查设备")
                raise RuntimeError("CPU 加载失败; 请检查设备")
            # 尽管 torch.cpu.is_available() 在一般情况下恒为 True
            # 此处为保持设备检查逻辑统一仍使用该判断

        if self.device == "auto":   # 自动选择设备
            if torch.cuda.is_available():
                self.device = "cuda"
                self.logger.info("AUTO: 成功加载 Nvidia CUDA")

            elif torch.xpu.is_available():
                self.device = "xpu"
                self.logger.info("AUTO: 成功加载 Intel XPU")

            elif torch.backends.mps.is_available():
                self.device = "mps"
                self.logger.info("AUTO: 成功加载 Apple MPS")

            else:
                if torch.cpu.is_available():
                    self.device = "cpu"
                    self.logger.info("AUTO: 成功加载 CPU")
                    self.logger.warning("AUTO: 未检测到任何可行的加速方式, 自动选择 CPU; 训练可能受影响")

                else:
                    raise RuntimeError("AUTO: 未检测到可用设备; 请检查设备配置")

    def _init_model(self):
        """初始化模型"""
        self.model = Tower_GPT(
            decoder_num=self.decoder_num,
            head_num=self.head_num,
            d=self.d,
            dk=self.dk,
            dff=self.dff,
            vocab_size=self.vocab_size,
            device=self.device,
            dropout=self.dropout,
        )   # 构建模型

        if self.compile:   # 编译模型
            try:
                self.model = torch.compile(self.model)
                self.logger.debug("模型编译完成")

            except Exception as e:
                self.logger.error(f"模型编译失败: {e}")
                raise RuntimeError(f"模型编译失败: {e}")
            
        self.model.to(self.device)   # 模型移动到指定设备
        self.logger.debug(f"模型已加载到设备: {self.device}")

        if self.train_method in ["dpo", "ipo"]:
            self._init_ref_model()

    def _init_ref_model(self):
            """初始化 DPO 所需的参考模型"""
            self.logger.info("正在初始化参考模型 (Ref Model)...")
            self.ref_model = copy.deepcopy(self.model)
            self.ref_model.eval()
            self.ref_model.to(self.device)
            # 深拷贝当前模型

            for param in self.ref_model.parameters():
                param.requires_grad = False
                # 冻结参数

            self.logger.info("参考模型初始化完成")

    def _init_dataloader(self):
        """初始化数据加载器"""
        assert self.train_method in ["text", "chat", "dpo", "ipo", "simpo"], f"不支持的训练方法: {self.train_method}"
        # 断言训练方法信息无误

        if self.train_method == "text":
            self.processor = TextDataProcessor(
                json_file=self.train_data_path,
                sp_model_path=self.tokenizer_path,
                block_size=self.block_size,
            )

        elif self.train_method == "chat":
            self.processor = MultiTurn_DialogueDataProcessor(
                json_file=self.train_data_path,
                sp_model_path=self.tokenizer_path,
                block_size=self.block_size,
            )

        elif self.train_method in ["dpo", "ipo", "simpo"]:
            self.processor = RLDataProcessor(
                json_file=self.train_data_path,
                sp_model_path=self.tokenizer_path,
                block_size=self.block_size,
            )

        self.logger.debug(f"Train DataProcessor 加载完成")

        if self.train_method == "text":
            collate = text_collate_fn
        elif self.train_method in ["dpo", "ipo", "simpo"]:
            collate = rl_collate_fn
        else:
            collate = None
        # 选择对应的 collate_fn

        if self.yield_load:
            self.logger.debug(f"使用 Yield 方式加载数据")

            if self.train_method == "text":
                self.train_dataset = GeneratorTextDataset(self.processor)   # type: ignore
            elif self.train_method == "chat":
                self.train_dataset = Talk_GeneratorDialogueDataset(self.processor)   # type: ignore
            elif self.train_method in ["dpo", "ipo", "simpo"]:
                self.train_dataset = RLGeneratorDataset(self.processor)   # type: ignore

            self.logger.debug(f"Train Dataset 加载完成")

            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,   # yield 下必须为 0
                collate_fn=collate,
            )   # 初始化训练数据加载器

        else:
            self.logger.debug(f"使用普通方式加载数据")

            if self.train_method == "text":
                self.train_dataset = TextDataset(self.processor)   # type: ignore
            elif self.train_method == "chat":
                self.train_dataset = Talk_DialogueDataset(self.processor)   # type: ignore
            elif self.train_method in ["dpo", "ipo", "simpo"]:
                self.train_dataset = RLDataset(self.processor)   # type: ignore

            self.logger.debug(f"Train Dataset 加载完成")

            self.train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate,
            )   # 初始化训练数据加载器

        self.logger.debug(f"Train DataLoader 加载完成")

        if self.valid_data_path:   # 初始化验证数据加载器
            if self.train_method == "text":
                valid_processor = TextDataProcessor(
                    json_file=self.valid_data_path,
                    sp_model_path=self.tokenizer_path,
                    block_size=self.block_size,
                )
                valid_dataset = TextDataset(valid_processor)

            elif self.train_method == "chat":
                valid_processor = MultiTurn_DialogueDataProcessor(
                    json_file=self.valid_data_path,
                    sp_model_path=self.tokenizer_path,
                    block_size=self.block_size,
                )
                valid_dataset = Talk_DialogueDataset(valid_processor)

            elif self.train_method in ["dpo", "ipo", "simpo"]:
                valid_processor = RLDataProcessor(
                    json_file=self.valid_data_path,
                    sp_model_path=self.tokenizer_path,
                    block_size=self.block_size,
                )
                valid_dataset = RLDataset(valid_processor)

            self.logger.debug(f"Valid Dataset 加载完成")

            self.valid_dataloader = DataLoader(
                valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate,
            )

            self.logger.debug(f"Valid DataLoader 加载完成")

        else:
            self.valid_dataloader = None

        if self.test_data_path:   # 初始化测试数据加载器
            if self.train_method == "text":
                test_processor = TextDataProcessor(
                    json_file=self.test_data_path,
                    sp_model_path=self.tokenizer_path,
                    block_size=self.block_size,
                )
                test_dataset = TextDataset(test_processor)

            elif self.train_method == "chat":
                test_processor = MultiTurn_DialogueDataProcessor(
                    json_file=self.test_data_path,
                    sp_model_path=self.tokenizer_path,
                    block_size=self.block_size,
                )
                test_dataset = Talk_DialogueDataset(test_processor)

            elif self.train_method in ["dpo", "ipo", "simpo"]:
                test_processor = RLDataProcessor(
                    json_file=self.test_data_path,
                    sp_model_path=self.tokenizer_path,
                    block_size=self.block_size,
                )
                test_dataset = RLDataset(test_processor)

            self.logger.debug(f"Test Dataset 加载完成")

            self.test_dataloader = DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                collate_fn=collate,
            )

            self.logger.debug(f"Test DataLoader 加载完成")

        else:
            self.test_dataloader = None

        self.data_length = self.processor.data_length()   # 训练数据集长度
        self.logger.info(f"数据集加载完成")

    def _init_optimizer(self):
        """初始化优化器"""
        OPTIMIZER_MAP = {
            "adamw": AdamW,
            "adafactor": Adafactor,
            "galore_adamw": GaLoreAdamW8bit,
            "galore_adafactor": GaLoreAdafactor,
        }

        assert self.optimizer_name in OPTIMIZER_MAP, f"优化器必须是 {list(OPTIMIZER_MAP.keys())}, 但得到 {self.optimizer_name}"
        # 断言优化器信息无误

        self.optimizer: AdamW | GaLoreAdamW8bit | Adafactor | GaLoreAdafactor
        if self.optimizer_name in ["adamw", "galore_adamw"]:
            self.optimizer = OPTIMIZER_MAP[self.optimizer_name](
                self.model.parameters(),
                lr=self.learning_rate,
                betas=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )   # 初始化 AdamW系 优化器

        else:
            self.optimizer = OPTIMIZER_MAP[self.optimizer_name](
                self.model.parameters(),
                lr=self.learning_rate,
                beta1=self.betas,
                eps=self.eps,
                weight_decay=self.weight_decay,
            )   # 初始化 Adafactor系 优化器

        self.logger.debug(f"优化器初始化完成: {self.optimizer}")

    def _init_lr_scheduler(self):
        """初始化学习率调度器"""
        if not self.lr_scheduler:
            self.rate_scheduler = None
            return

        assert self.max_lr_rate is not None, "max_lr_rate 必须指定"
        assert self.div_factor is not None, "div_factor 必须指定"
        assert self.pct_start is not None, "pct_start 必须指定"
        assert self.anneal_strategy in ["linear", "cos"], f"anneal_strategy 必须指定为 linear 或 cos, 但得到 {self.anneal_strategy}"
        last_step = -1 if self.train_steps == 0 else self.train_steps - 1
        self.rate_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            max_lr=self.max_lr_rate * self.learning_rate,
            epochs=self.all_epochs,
            cycle_momentum=False,
            steps_per_epoch=int(np.ceil(self.data_length / (self.batch_size * self.accumulation_steps))),
            div_factor=self.div_factor,
            last_epoch=last_step,
            pct_start=self.pct_start,
            anneal_strategy=self.anneal_strategy,   # type: ignore
        )   # 创建学习率调度器

    def _init_tensorboard(self):
        """初始化 TensorBoard"""
        if self.tensorboard:
            self.writer = SummaryWriter(self.tensorboard_dir)
            # 初始化 TensorBoard 写入器

        else:
            self.writer = None

    def _save_resume_page(self, dir: str, suffix: str | int, now_time: str):
        """保存恢复页面
        
        参数:
        - dir (str): 保存目录
        - suffix (str | int): 保存后缀
        - now_time (str): 当前时间
        """
        os.makedirs(os.path.join(self.output_dir, dir), exist_ok=True)
        # 确保目录存在

        save_path = os.path.join(
            self.output_dir,
            dir,
            f"{self.output_model_name}_{suffix}.log",
        )   # 构建保存路径

        resume_page = {
            "model_file": f"{self.output_model_name}_{suffix}{self.model_suffix}",
            "optimizer_file": f"{self.output_model_name}_{suffix}{self.optimizer_suffix}",
            "scheduler_file": f"{self.output_model_name}_{suffix}{self.scheduler_suffix}" if self.scheduler_suffix else None,
            "train_data_path": self.train_data_path,
            "valid_data_path": self.valid_data_path,
            "test_data_path": self.test_data_path,
            "tokenizer_path": self.tokenizer_path,
            "train_method": self.train_method,
            "finetune": self.finetune,
            "batch_size": self.batch_size,
            "accumulation_steps": self.accumulation_steps,
            "block_size": self.block_size,
            "rl_beta": self.rl_beta,
            "gamma": self.gamma,
            "time": now_time,
            "all_epochs": self.all_epochs,
            "now_epoch": self.now_epoch,
            "train_steps": self.train_steps,
            "tensorboard": self.tensorboard,
            "tensorboard_dir": self.tensorboard_dir,
            "writer_name": self.writer_name,
            "skip_steps": self.local_steps + 1,   # 已经加载多少, 就跳过多少
        }   # 恢复页面内容

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(resume_page, f, ensure_ascii=False, indent=4)
        # 保存恢复页面

        self.logger.info(f"恢复页面已保存: {save_path}")   # logger

    def _load_resume_page(self):
        """加载恢复页面"""
        if self.ckpt_path is None:
            self.logger.warning("检查点路径为空, 无法加载恢复页面")
            raise RuntimeError("检查点路径为空, 无法加载恢复页面")
        # 检查点路径为空, 直接返回 None

        try:
            with open(self.ckpt_path, "r", encoding="utf-8") as f:
                resume_page = json.load(f)
            # 加载恢复页面

            self.resume_model_path = os.path.join(os.path.dirname(self.ckpt_path), resume_page["model_file"])
            # 从恢复页面中提取模型子路径

            self.resume_optimizer_path = os.path.join(os.path.dirname(self.ckpt_path), resume_page["optimizer_file"])
            # 从恢复页面中提取优化器子路径

            self.resume_scheduler_path = os.path.join(os.path.dirname(self.ckpt_path), resume_page["scheduler_file"]) if resume_page["scheduler_file"] else None
            # 从恢复页面中提取调度器子路径, 若不存在则为 None
            
            self.train_data_path = resume_page["train_data_path"]
            # 从恢复页面中提取训练数据路径
            
            self.valid_data_path = resume_page["valid_data_path"]
            # 从恢复页面中提取验证数据路径
            
            self.test_data_path = resume_page["test_data_path"]
            # 从恢复页面中提取测试数据路径
            
            self.tokenizer_path = resume_page["tokenizer_path"]
            # 从恢复页面中提取分词器路径
            
            self.train_method = resume_page["train_method"]
            # 从恢复页面中提取训练方法
            
            self.finetune = resume_page["finetune"]
            # 从恢复页面中提取微调标志
            
            self.batch_size = resume_page["batch_size"]
            # 从恢复页面中提取批量大小
            
            self.accumulation_steps = resume_page["accumulation_steps"]
            # 从恢复页面中提取梯度累积步数
            
            self.block_size = resume_page["block_size"]
            # 从恢复页面中提取块大小

            self.rl_beta = resume_page["rl_beta"]
            # 从恢复页面中提取强化学习 beta
            
            self.gamma = resume_page["gamma"]
            # 从恢复页面中提取 gamma
            
            self.resume_time = resume_page["time"]
            # 从恢复页面中提取保存时间
            
            self.all_epochs = resume_page["all_epochs"]
            # 从恢复页面中提取总 epoch 数
            
            self.now_epoch = resume_page["now_epoch"]
            # 从恢复页面中提取当前 epoch
            
            self.train_steps = resume_page["train_steps"]
            # 从恢复页面中提取训练步数
            
            self.tensorboard = resume_page["tensorboard"]
            # 从恢复页面中提取 tensorboard 标志
            
            self.tensorboard_dir = resume_page["tensorboard_dir"]
            # 从恢复页面中提取 tensorboard 目录
            
            self.writer_name = resume_page["writer_name"]
            # 从恢复页面中提取 writer 名称
            
            self.skip_steps = resume_page["skip_steps"]
            # 从恢复页面中提取已经加载的步数, 跳过多少步

            self.need_skip = True
            # 更改需要跳过批次的标志

            self.logger.info(f"恢复页面已加载: {self.ckpt_path}")   # logger

        except Exception as e:
            self.logger.error(f"加载恢复页面时出错: {e}")
            raise RuntimeError(f"加载恢复页面时出错: {e}")

    def _forward_calc(
        self,
        inputs: Tensor,
        target: Tensor,
        loss_mask: Tensor | None = None,
        accumulation_steps: int = 1,
    ) -> Tensor:
        """前向计算

        参数:
        - inputs (Tensor): 输入张量, 形状为 (batch_size, seq_len)
        - target (Tensor): 目标张量, 形状为 (batch_size, seq_len)
        - loss_mask (Tensor | None): 损失掩码张量, 形状为 (batch_size, seq_len), 默认为 None
        - accumulation_steps (int): 梯度累积步数, 默认为 1

        返回:
        - Tensor: 损失张量, 形状为 [loss]
        """
        inputs = inputs.to(self.device)
        target = target.to(self.device)
        # 移动到指定设备

        pred: Tensor = self.model(inputs)   # 前向计算
        pred = pred.view(-1, pred.size(-1))
        # 将 pred 重新形状为 [batch_size * sequence_length, vocab_size]

        target = target.view(-1)
        # 将 target 重新形状为 [batch_size * sequence_length]

        all_loss: Tensor = self.loss_fn(pred, target)
        # 计算所有位置的loss

        if loss_mask is not None:
            loss_mask = loss_mask.to(self.device)
            # 移动到指定设备

            loss_mask = loss_mask.view(-1)   # 展平掩码
            masked_loss = (all_loss * loss_mask) / accumulation_steps
            loss = masked_loss.sum() / (loss_mask.sum() + 1e-8)   # 避免除零
            # 只保留助手回复部分的loss

        else:
            loss = all_loss.mean() / accumulation_steps   # 计算平均损失

        return loss

    def _compute_dpo_loss(
        self,
        chosen_inputs: Tensor,
        chosen_targets: Tensor,
        chosen_masks: Tensor,
        rejected_inputs: Tensor,
        rejected_targets: Tensor,
        rejected_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """计算 DPO / IPO 损失"""
        batch_size = chosen_inputs.size(0)

        combined_inputs = torch.cat([chosen_inputs, rejected_inputs], dim=0)
        # 拼接 chosen 和 rejected

        combined_logits = self.model(combined_inputs)   # (2*batch, seq_len, vocab)
        # 策略模型前向

        with torch.no_grad():
            combined_ref_logits = self.ref_model(combined_inputs)
            # 参考模型前向 (不计算梯度)

        log_probs = torch.log_softmax(combined_logits, dim=-1)
        ref_log_probs = torch.log_softmax(combined_ref_logits, dim=-1)
        # 计算 log softmax

        chosen_log_probs = log_probs[:batch_size].gather(dim=-1, index=chosen_targets.unsqueeze(-1)).squeeze(-1)
        rejected_log_probs = log_probs[batch_size:].gather(dim=-1, index=rejected_targets.unsqueeze(-1)).squeeze(-1)
        chosen_ref_log_probs = ref_log_probs[:batch_size].gather(dim=-1, index=chosen_targets.unsqueeze(-1)).squeeze(-1)
        rejected_ref_log_probs = ref_log_probs[batch_size:].gather(dim=-1, index=rejected_targets.unsqueeze(-1)).squeeze(-1)
        # 收集目标 token 的对数概率

        chosen_log_probs = chosen_log_probs * chosen_masks
        rejected_log_probs = rejected_log_probs * rejected_masks
        chosen_ref_log_probs = chosen_ref_log_probs * chosen_masks
        rejected_ref_log_probs = rejected_ref_log_probs * rejected_masks
        # 应用损失掩码

        policy_chosen_logps = chosen_log_probs.sum(dim=-1)
        policy_rejected_logps = rejected_log_probs.sum(dim=-1)
        ref_chosen_logps = chosen_ref_log_probs.sum(dim=-1)
        ref_rejected_logps = rejected_ref_log_probs.sum(dim=-1)
        # 对序列求和, 得到每个样本的对数概率 [batch]

        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = ref_chosen_logps - ref_rejected_logps
        logits = pi_logratios - ref_logratios

        if self.train_method == "dpo":
            loss = -fc.logsigmoid(self.rl_beta * logits).mean()
            # 计算比率和 DPO / IPO 损失

        if self.train_method == "ipo":
            ipo_target = 1.0 / (2.0 * self.rl_beta)
            # IPO 目标值, 这里的 beta 是原论文的 tau

            loss = (logits - ipo_target) ** 2
            loss = loss.mean()   # 对所有样本求平均

        return loss, pi_logratios, ref_logratios

    def _compute_simpo_loss(
        self,
        chosen_inputs: Tensor,
        chosen_targets: Tensor,
        chosen_masks: Tensor,
        rejected_inputs: Tensor,
        rejected_targets: Tensor,
        rejected_masks: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """计算 SimPO 损失"""
        batch_size = chosen_inputs.size(0)

        combined_inputs = torch.cat([chosen_inputs, rejected_inputs], dim=0)
        # 拼接 chosen 和 rejected

        combined_logits = self.model(combined_inputs)   # (2*batch, seq_len, vocab)
        # 模型前向

        log_probs = torch.log_softmax(combined_logits, dim=-1)
        # 计算 log softmax

        chosen_log_probs = log_probs[:batch_size].gather(dim=-1, index=chosen_targets.unsqueeze(-1)).squeeze(-1)
        rejected_log_probs = log_probs[batch_size:].gather(dim=-1, index=rejected_targets.unsqueeze(-1)).squeeze(-1)
        # 收集目标 token 的对数概率

        chosen_log_probs = chosen_log_probs * chosen_masks
        rejected_log_probs = rejected_log_probs * rejected_masks
        # 应用损失掩码

        chosen_len = chosen_masks.sum(dim=-1).clamp(min=1e-8)
        rejected_len = rejected_masks.sum(dim=-1).clamp(min=1e-8)
        # 计算每个样本的有效长度

        policy_chosen_logps = chosen_log_probs.sum(dim=-1) / chosen_len
        policy_rejected_logps = rejected_log_probs.sum(dim=-1) / rejected_len
        # 对序列求和, 得到每个样本的对数概率

        logits = policy_chosen_logps - policy_rejected_logps
        loss = -fc.logsigmoid(self.rl_beta * logits - self.gamma).mean()
        # 计算 SimPO 损失

        return loss, policy_chosen_logps, policy_rejected_logps

    def _mixed_dtype(self) -> torch.dtype:
        """根据混合精度模式返回对应 dtype

        参数:
        - dtype (str): 混合精度模式, 可选值为 "full" "fp16" "bf16"

        返回:
        - torch.dtype: 对应 dtype
        """
        if self.mixed_precision == "full":
            return torch.float32
        elif self.mixed_precision == "fp16":
            return torch.float16
        elif self.mixed_precision == "bf16":
            return torch.bfloat16
        else:
            return torch.float16

    def _load_from_base_model(self):
        """从基础模型加载模型和优化器状态"""
        try:
            assert self.train_model_dir is not None and self.train_model_name is not None, "训练模型目录或名称为空, 无法加载检查点"
            model_path = os.path.join(self.train_model_dir, self.train_model_name+self.model_suffix)
            optimizer_path = os.path.join(self.train_model_dir, self.train_model_name+self.optimizer_suffix)

            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.logger.debug(f"成功加载基础模型模型: {model_path}")
            # 加载模型状态

            if self.load_optimizer:
                try:
                    self.optimizer.load_state_dict(torch.load(optimizer_path, map_location=self.device))
                    self.logger.debug(f"成功加载基础模型优化器: {optimizer_path}")
                except Exception as e:
                    self.logger.warning(f"加载基础模型优化器失败, 可能影响训练")
                # 加载优化器状态

                if self.scheduler_suffix and self.rate_scheduler is not None:
                    try:
                        scheduler_path = os.path.join(self.train_model_dir, self.train_model_name+self.scheduler_suffix)
                        self.rate_scheduler.load_state_dict(torch.load(scheduler_path, map_location=self.device))
                        self.logger.debug(f"成功加载基础模型调度器: {scheduler_path}")
                    except Exception as e:
                        self.logger.warning(f"加载基础模型调度器失败, 可能影响训练")
                    # 加载调度器状态

                self.logger.info(f"成功从基础模型加载模型和优化器状态: {model_path}")
                # logger

            else:
                self.logger.info(f"成功从基础模型加载模型状态: {model_path}")
                # logger      
        
        except Exception as e:
            self.logger.error(f"从检查点加载模型状态失败: {e}")
            raise RuntimeError(f"从检查点加载模型状态失败: {e}")

    def _load_from_resume(self):
        """从恢复页加载模型和优化器状态"""
        try:
            self.model.load_state_dict(torch.load(self.resume_model_path, map_location=self.device))
            self.logger.debug(f"成功加载恢复模型: {self.resume_model_path}")
            # 加载模型状态

            self.optimizer.load_state_dict(torch.load(self.resume_optimizer_path, map_location=self.device))
            self.logger.debug(f"成功加载恢复优化器: {self.resume_optimizer_path}")
            # 加载优化器状态

            if self.resume_scheduler_path is not None and self.rate_scheduler is not None:
                self.rate_scheduler.load_state_dict(torch.load(self.resume_scheduler_path, map_location=self.device))
                self.logger.debug(f"成功加载优化调度器: {self.resume_scheduler_path}")
            # 加载调度器状态

            if self.train_method in ["dpo", "ipo"]:
                self._init_ref_model()
                # 加载参考模型状态

            self.logger.info(f"成功从恢复页加载模型和优化器状态: {self.ckpt_path}")
            # logger

        except Exception as e:
            self.logger.error(f"从恢复页加载模型和优化器状态失败: {e}")
            raise RuntimeError(f"从恢复页加载模型和优化器状态失败: {e}")

    def _is_finetune(self):
        """
        微调判定, 冻结 embedding 层参数
        """
        if self.finetune:
            layers_freeze = [self.model.embedding]
            # 指定embedding层参数

            for layer in layers_freeze:
                for param in layer.parameters():
                    param.requires_grad = False
                # 冻结参数

            self.logger.info("微调模式加载成功")

        else:
            self.logger.info("非微调模式, 全部参数参与训练")

    def _grad_checkpoint(self):
        """梯度检查点"""
        pass   # TODO: 梯度检查点

    def _save_checkpoint(self, dir: str, suffix: str | int):
        """保存检查点
        
        参数:
        - dir: 检查点目录
        - suffix: 检查点后缀, 如 "epoch_1" "best"
        """
        os.makedirs(os.path.join(self.output_dir, dir), exist_ok=True)
        # 确保目录存在

        save_path = os.path.join(
            self.output_dir,
            dir,
            f"{self.output_model_name}_{suffix}{self.model_suffix}",
        )   # 构建保存路径

        save_optimizer_path = os.path.join(
            self.output_dir,
            dir,
            f"{self.output_model_name}_{suffix}{self.optimizer_suffix}",
        )   # 构建优化器保存路径

        if self.lr_scheduler is not None and self.scheduler_suffix is not None:
            save_scheduler_path = os.path.join(
                self.output_dir,
                dir,
                f"{self.output_model_name}_{suffix}{self.scheduler_suffix}",
            )   # 构建调度器保存路径

        os.makedirs(os.path.dirname(save_path), exist_ok=True)         # 确保目录存在
        torch.save(self.model.state_dict(), save_path)                 # 保存模型状态
        torch.save(self.optimizer.state_dict(), save_optimizer_path)   # 保存优化器状态

        if self.lr_scheduler is not None and self.rate_scheduler is not None:
            torch.save(
                self.rate_scheduler.state_dict(),
                save_scheduler_path,
            )   # 保存调度器状态

        self.logger.info(f"检查点已保存: {save_path}")   # logger

    def check_best_checkpoint(self, new_val_blue: float | None, new_val_loss: float, update_step: int):
        """检查并保存最佳检查点, BLUE 优先
        
        参数:
        - new_blue: 新的 BLUE 分数
        - new_ppl: 新的 PPL 分数
        - update_step: 当前训练更新步数
        """
        if new_val_blue is not None:
            if new_val_blue > self.best_blue:   # BLUE 优先
                self.best_blue = new_val_blue
                self.best_loss = new_val_loss
                self._save_checkpoint("best_checkpoint", "best_blue")
                self.logger.info(f"[BEST BLUE] 新最佳 BLUE 分数: {new_val_blue:.4f}, Loss: {new_val_loss:.4f}, 更新步数: {update_step}")

            if new_val_loss < self.best_loss:   # Loss 计算
                self.best_loss = new_val_loss
                self._save_checkpoint("best_checkpoint", "best_loss")
                self.logger.info(f"[BEST LOSS] 新最佳 BLUE 分数: {new_val_blue:.4f}, Loss: {new_val_loss:.4f}, 更新步数: {update_step}")

        else:
            if new_val_loss < self.best_loss:   # Loss 计算
                self.best_loss = new_val_loss
                self._save_checkpoint("best_checkpoint", "best_loss")
                self.logger.info(f"[BEST LOSS] 新最佳 Loss 分数: {new_val_loss:.4f}, 更新步数: {update_step}, 无 BLUE 分数")

    def delete_checkpoint(self, dir: str="time"):
        """删除旧的时间顺序检查点
        
        参数:
        - dir: 检查点目录, 默认 "time"
        """
        try:
            delete_dir = os.path.join(self.output_dir, dir)
            # 构建要删除的目录

            if not os.path.exists(delete_dir):   # 如果目录不存在, 直接返回
                self.logger.debug(f"目录不存在, 无需删除检查点: {delete_dir}")
                return

            checkpoints = []
            for item in os.listdir(delete_dir):   # 列出文件
                item_path = os.path.join(delete_dir, item)   # 构建子文件路径

                if os.path.isdir(item_path):   # 获取该子文件最后修改时间
                    mtime = os.path.getmtime(item_path)
                    checkpoints.append((mtime, item, item_path))   # 添加 (最后修改时间, 目录名, 完整路径) 到列表

            checkpoints.sort(key=lambda x: x[0])   # 按修改时间排序 (最旧的在前)
            if len(checkpoints) > self.max_checkpoints:
                for i in range(len(checkpoints) - self.max_checkpoints):
                    mtime, checkpoint_name, checkpoint_path = checkpoints[i]
                    shutil.rmtree(checkpoint_path)   # 删除目录
                    self.logger.debug(f"删除旧检查点: {checkpoint_path}")

        except Exception as e:
            self.logger.error(f"删除旧检查点失败: {e}")

    def train(self):
        """训练主进程"""
        self.model.train()   # 设置为训练模式
        self._init_progress()   # 初始化进度条

        tsp_show_txt = 'train_info_steps: {}/{}'.format(
                self.info_steps, self.all_tsp
        )   # 设置tsp更新信息

        epoch_show_txt = 'epoch: {}/{}'.format(
            self.now_epoch, self.all_epochs
        )   # 设置 epoch 更新信息

        self.train_progress.update(self.tsp_progress, show_info=tsp_show_txt, advance=0)
        self.train_progress.update(self.epoch_progress, show_info=epoch_show_txt, advance=1)
        # 初始进度条显示更新

        if self.need_skip:
            self._init_skip_progess()
            # 启动跳过批次进度条

            skip_show_txt = 'skipped_steps: {}/{}'.format(
                0, self.skip_steps
            )   # 设置跳过批次更新信息

            self.train_progress.update(self.main_skip_progress, show_info=skip_show_txt, advance=0)
            # 初始跳过批次进度条显示更新

            tsp_show_txt = 'train_info_steps: {}/{}'.format(
                    self.local_steps // self.info_update_interval, self.all_tsp
            )   # 设置tsp更新信息

            self.train_progress.update(self.tsp_progress, show_info=tsp_show_txt, advance=0)
            # 更新tsp进度条显示

        for epoch in range(self.all_epochs):
            if epoch < self.now_epoch:
                continue   # 跳过已经完成的轮次

            try:
                self.train_signal = True   # 训练信号

                if self.train_method in ["text", "chat"]:
                    self.train_one_epoch()    # 训练一个轮次
                    test_loss = self.test()   # 测试模型

                else:   # 强化学习训练
                    self.train_one_rl_epoch()    # 训练一个轮次
                    test_loss = self.rl_test()   # 测试模型

                self.now_epoch += 1
                # 轮次增加

                epoch_show_txt = 'epoch: {}/{}'.format(
                    self.now_epoch, self.all_epochs
                )   # 设置epoch更新信息

                self.train_progress.update(self.epoch_progress, show_info=epoch_show_txt, advance=1)
                # 更新epoch信息与进度条

                self._save_checkpoint(f"epoch/epoch{self.now_epoch}", f"epoch_{self.now_epoch}")
                # 保存检查点

            except Exception as e:
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = os.path.join("exit_save", now+"_exit_ckpt")
                os.makedirs(save_dir, exist_ok=True)
                # 构建错误备份目录

                self.train_progress.stop()
                # 停止训练进度条

                self._save_resume_page(
                    save_dir,
                    f"error_{now}",
                    now
                )   # 保存恢复页面

                self.logger.error(f"训练轮次 {epoch} 失败: {e}")
                self.train_signal = False   # 训练信号
                raise e

        self._save_checkpoint("epoch", f"final")
        self.train_signal = False   # 训练信号
        self.train_progress.stop()
        # 停止训练进度条

        self.logger.info("训练完成!")

    def train_one_epoch(self):
        """训练一个轮次"""
        total_loss = 0.0
        info_loss = 0.0
        for step, (x, y, loss_mask) in enumerate(self.train_dataloader):   # 生成步进索引
            if step < self.skip_steps and self.need_skip:
                self.train_progress.update(
                    self.main_skip_progress,
                    advance=1,
                    show_info=f'skipped_steps: {step+1}/{self.skip_steps}'
                )
                continue

            else:
                self.need_skip = False

            x: Tensor = x.to(self.device).long()
            y: Tensor = y.to(self.device).long() 
            loss_mask: Tensor | None = loss_mask.to(self.device).float() if loss_mask is not None else None
            # 将数据移动到指定设备上

            with autocast(device_type=str(self.device), dtype=self._mixed_dtype()):
                loss = self._forward_calc(x, y, loss_mask, self.accumulation_steps)
            # 前向计算

            if self.scaler is not None:   # 反向传播
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()
            info_loss += loss.item()
            self.local_steps = step
            # 累计损失和本地步数更新

            self.train_progress.update(self.tsp_progress, advance=1/self.info_update_interval)
            # 更新训练进度条

            if (step + 1) % self.accumulation_steps == 0:   # 梯度更新
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                    # 梯度裁剪

                if self.scaler is not None:   # 混合精度更新参数
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                else:   # 常规更新参数
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)
                self.train_steps += 1
                # 梯度清空

                if self.writer is not None and self.writer_name is not None:
                    self.writer.add_scalar(self.writer_name+'_train_loss', total_loss, self.train_steps)
                    # 记录训练损失

                if self.rate_scheduler is not None:
                    self.rate_scheduler.step()
                    # 更新学习率调度器

                total_loss = 0.0   # 重置批次损失

            if (step + 1) % self.info_update_interval == 0:
                self.info_steps += 1

                tsp_show_txt = 'train_info_steps: {}/{}'.format(
                        self.info_steps, self.all_tsp
                )   # 设置tsp更新信息

                self.train_progress.update(self.tsp_progress, show_info=tsp_show_txt)
                # 更新 tsp 信息

                avg_loss = (info_loss / self.info_update_interval) * self.accumulation_steps
                # 计算平均损失

                info_loss = 0.0   # 重置总损失

                if self.writer is not None and self.writer_name is not None:
                    self.writer.add_scalar(self.writer_name+'_avg_train_loss', avg_loss, self.info_steps)
                    # 记录训练损失

                    if self.ppl_eval:
                        ppl = math.exp(avg_loss)
                        self.writer.add_scalar(self.writer_name+'_avg_train_ppl', ppl, self.info_steps)
                        # 记录训练 PPL

                eval_loss = self.evaluate()
                # 评估模型

                self._save_checkpoint(f"time/step{self.info_steps}", f"epoch_{self.now_epoch}_step_{self.info_steps}")
                # 保存检查点

                self.delete_checkpoint("time")
                # 删除旧检查点

                if eval_loss is not None and self.save_best_checkpoint:
                    self.check_best_checkpoint(None, eval_loss, self.info_steps)
                    # 保存最佳检查点

        self.optimizer.zero_grad()
        # 清空梯度

    def train_one_rl_epoch(self):
        total_loss = 0.0
        info_loss = 0.0
        total_pi_logratios = 0.0
        total_ref_logratios = 0.0
        for step, batch in enumerate(self.train_dataloader):
            if step < self.skip_steps and self.need_skip:
                self.train_progress.update(self.main_skip_progress, advance=1)
                continue
            else:
                self.need_skip = False

            # ==================== 数据解析 ====================

            (chosen_inputs, chosen_targets, chosen_masks,
                rejected_inputs, rejected_targets, rejected_masks) = batch
            # 移到设备

            chosen_inputs: Tensor = chosen_inputs.to(self.device).long()
            chosen_targets: Tensor = chosen_targets.to(self.device).long()
            chosen_masks: Tensor = chosen_masks.to(self.device).float()
            rejected_inputs: Tensor = rejected_inputs.to(self.device).long()
            rejected_targets: Tensor = rejected_targets.to(self.device).long()
            rejected_masks: Tensor = rejected_masks.to(self.device).float()

            # ==================== Forward ====================

            with autocast(device_type=str(self.device), dtype=self._mixed_dtype()):
                if self.train_method in ["dpo", "ipo"]:
                    loss, pi_logratios, ref_logratios = self._compute_dpo_loss(
                        chosen_inputs, chosen_targets, chosen_masks,
                        rejected_inputs, rejected_targets, rejected_masks
                    )
                else:
                    loss, pi_logratios, ref_logratios = self._compute_simpo_loss(
                        chosen_inputs, chosen_targets, chosen_masks,
                        rejected_inputs, rejected_targets, rejected_masks
                    )

                loss = loss / self.accumulation_steps

            # ==================== Backward ====================

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item()
            total_pi_logratios += pi_logratios.mean().item() / self.accumulation_steps
            total_ref_logratios += ref_logratios.mean().item() / self.accumulation_steps
            info_loss += loss.item()
            self.local_steps = step

            self.train_progress.update(self.tsp_progress, advance=1/self.info_update_interval)

            # ==================== 优化更新 ====================

            if (step + 1) % self.accumulation_steps == 0:
                if self.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.train_steps += 1

                if self.writer is not None and self.writer_name is not None:
                    self.writer.add_scalar(self.writer_name+'_train_loss', total_loss, self.train_steps)
                    self.writer.add_scalar(self.writer_name+'_train_pi_logratios', total_pi_logratios, self.train_steps)
                    self.writer.add_scalar(self.writer_name+'_train_ref_logratios', total_ref_logratios, self.train_steps)
                    self.writer.add_scalar(self.writer_name+'_train_reward', total_pi_logratios - total_ref_logratios, self.train_steps)

                if self.rate_scheduler is not None:
                    self.rate_scheduler.step()

                total_loss = 0.0
                total_pi_logratios = 0.0
                total_ref_logratios = 0.0

            # ==================== 信息更新 ====================

            if (step + 1) % self.info_update_interval == 0:
                self.info_steps += 1
                tsp_show_txt = f'train_info_steps: {self.info_steps}/{self.all_tsp}'
                self.train_progress.update(self.tsp_progress, show_info=tsp_show_txt)

                avg_loss = (info_loss / self.info_update_interval) * self.accumulation_steps
                info_loss = 0.0

                if self.writer is not None and self.writer_name is not None:
                    self.writer.add_scalar(self.writer_name+'_avg_train_loss', avg_loss, self.info_steps)
                    if self.ppl_eval:
                        ppl = math.exp(avg_loss)
                        self.writer.add_scalar(self.writer_name+'_avg_train_ppl', ppl, self.info_steps)

                eval_loss = self.rl_evaluate()
                self._save_checkpoint(f"time/step{self.info_steps}", f"epoch_{self.now_epoch}_step_{self.info_steps}")
                self.delete_checkpoint("time")
                if eval_loss is not None and self.save_best_checkpoint:
                    self.check_best_checkpoint(None, eval_loss, self.info_steps)

        self.optimizer.zero_grad()
        # 清空梯度

    def evaluate(self):
        if self.valid_dataloader is not None:
            self.model.eval()   # 设置为评估模式
            epoch_test_loss = 0

            with autocast(device_type=str(self.device), dtype=self._mixed_dtype()):
            # 自动混合精度

                with torch.no_grad():  # 不需要计算梯度
                    for tx, ty, t_mask in self.valid_dataloader:
                        tx: Tensor = tx.to(self.device).long()
                        ty: Tensor = ty.to(self.device).long() 
                        t_mask: Tensor | None = t_mask.to(self.device).float() if t_mask is not None else None

                        test_loss = self._forward_calc(tx, ty, t_mask)
                        epoch_test_loss += test_loss.item()
                        # 同上
                
            self.avg_test_loss = (epoch_test_loss / len(self.valid_dataloader))   # type: ignore
            # 计算损失

            avg_val_ppl = math.exp(self.avg_test_loss)
            # 计算 PPL

            self.logger.info(f"更新步数: {self.info_steps}")
            self.logger.info(f"验证损失: {self.avg_test_loss:.4f}")

            if self.ppl_eval:
                self.logger.info(f"验证 PPL: {avg_val_ppl:.4f}")
            # 更新 logger 信息

            # TODO: BLUE 评估

            if self.writer is not None and self.writer_name is not None:   # 记录验证损失
                self.writer.add_scalar(self.writer_name+'_valid', self.avg_test_loss, self.train_steps)

                if self.ppl_eval:   # 记录验证 PPL
                    self.writer.add_scalar(self.writer_name+'_valid_ppl', avg_val_ppl, self.train_steps)

            self.model.train()
            return self.avg_test_loss

        else:   # 无验证集时跳过
            return None

    def rl_evaluate(self):
        if self.valid_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        total_correct = 0      # 统计 chosen 胜出的样本数
        total_samples = 0

        with autocast(device_type=str(self.device), dtype=self._mixed_dtype()):
            with torch.no_grad():
                for batch in self.valid_dataloader:
                    (chosen_inputs, chosen_targets, chosen_masks,
                    rejected_inputs, rejected_targets, rejected_masks) = batch
                    # 解析 batch

                    chosen_inputs: Tensor = chosen_inputs.to(self.device).long()
                    chosen_targets: Tensor = chosen_targets.to(self.device).long()
                    chosen_masks: Tensor = chosen_masks.to(self.device).float()
                    rejected_inputs: Tensor = rejected_inputs.to(self.device).long()
                    rejected_targets: Tensor = rejected_targets.to(self.device).long()
                    rejected_masks: Tensor = rejected_masks.to(self.device).float()
                    # 移到设备

                    if self.train_method in ["dpo", "ipo"]:
                        loss, pi_logratios, ref_logratios = self._compute_dpo_loss(
                            chosen_inputs, chosen_targets, chosen_masks,
                            rejected_inputs, rejected_targets, rejected_masks
                        )
                    else:  # simpo
                        loss, pi_logratios, ref_logratios = self._compute_simpo_loss(
                            chosen_inputs, chosen_targets, chosen_masks,
                            rejected_inputs, rejected_targets, rejected_masks
                        )
                    # 根据训练方法选择对应的损失计算函数

                    batch_size = chosen_inputs.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    correct = (pi_logratios > 0).sum().item()
                    total_correct += correct
                    # 计算偏好准确率: pi_logratios > 0 表示模型认为 chosen 优于 rejected
                    # pi_logratios = policy_chosen_logps - policy_rejected_logps

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        self.logger.info(f"验证步数: {self.info_steps}")
        self.logger.info(f"验证损失: {avg_loss:.4f}")
        self.logger.info(f"偏好准确率: {accuracy:.4f}")
        # 日志记录

        if self.writer is not None and self.writer_name is not None:
            self.writer.add_scalar(self.writer_name + '_valid_loss', avg_loss, self.train_steps)
            self.writer.add_scalar(self.writer_name + '_valid_accuracy', accuracy, self.train_steps)
            # TensorBoard 记录

        self.model.train()
        return avg_loss

    def test(self):
        """运用评估集测试模型"""
        if self.test_dataloader is not None:
            self.model.eval()   # 设置为评估模式
            epoch_test_loss = 0

            with autocast(device_type=str(self.device), dtype=self._mixed_dtype()):
            # 自动混合精度

                with torch.no_grad():  # 不需要计算梯度
                    for tx, ty, t_mask in self.test_dataloader:
                        tx: Tensor = tx.to(self.device).long()
                        ty: Tensor = ty.to(self.device).long() 
                        t_mask: Tensor | None = t_mask.to(self.device).float() if t_mask is not None else None

                        test_loss = self._forward_calc(tx, ty, t_mask)
                        epoch_test_loss += test_loss.item()
                        # 同上
                
            self.avg_test_loss = (epoch_test_loss / len(self.test_dataloader))   # type: ignore
            # 计算损失

            avg_test_ppl = math.exp(self.avg_test_loss)
            # 计算 PPL

            self.logger.info(f"更新轮次: {self.now_epoch}")
            self.logger.info(f"测试损失: {self.avg_test_loss:.4f}")

            if self.ppl_eval:
                self.logger.info(f"测试 PPL: {avg_test_ppl:.4f}")
            # 更新 logger 信息

            # TODO: BLUE 评估

            if self.writer is not None and self.writer_name is not None:   # 记录测试损失
                self.writer.add_scalar(self.writer_name+'_test', self.avg_test_loss, self.now_epoch)

                if self.ppl_eval:   # 记录测试 PPL
                    self.writer.add_scalar(self.writer_name+'_test_ppl', avg_test_ppl, self.now_epoch)

            self.model.train()
            return self.avg_test_loss

        else:   # 无测试集时跳过
            return None
        
    def rl_test(self):
        if self.test_dataloader is None:
            return None

        self.model.eval()
        total_loss = 0.0
        total_correct = 0      # 统计 chosen 胜出的样本数
        total_samples = 0

        with autocast(device_type=str(self.device), dtype=self._mixed_dtype()):
            with torch.no_grad():
                for batch in self.test_dataloader:
                    (chosen_inputs, chosen_targets, chosen_masks,
                    rejected_inputs, rejected_targets, rejected_masks) = batch
                    # 解析 batch

                    chosen_inputs: Tensor = chosen_inputs.to(self.device).long()
                    chosen_targets: Tensor = chosen_targets.to(self.device).long()
                    chosen_masks: Tensor = chosen_masks.to(self.device).float()
                    rejected_inputs: Tensor = rejected_inputs.to(self.device).long()
                    rejected_targets: Tensor = rejected_targets.to(self.device).long()
                    rejected_masks: Tensor = rejected_masks.to(self.device).float()
                    # 移到设备

                    if self.train_method in ["dpo", "ipo"]:
                        loss, pi_logratios, ref_logratios = self._compute_dpo_loss(
                            chosen_inputs, chosen_targets, chosen_masks,
                            rejected_inputs, rejected_targets, rejected_masks
                        )
                    else:  # simpo
                        loss, pi_logratios, ref_logratios = self._compute_simpo_loss(
                            chosen_inputs, chosen_targets, chosen_masks,
                            rejected_inputs, rejected_targets, rejected_masks
                        )
                    # 根据训练方法选择对应的损失计算函数

                    batch_size = chosen_inputs.size(0)
                    total_loss += loss.item() * batch_size
                    total_samples += batch_size
                    correct = (pi_logratios > 0).sum().item()
                    total_correct += correct
                    # 计算偏好准确率: pi_logratios > 0 表示模型认为 chosen 优于 rejected
                    # pi_logratios = policy_chosen_logps - policy_rejected_logps

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples if total_samples > 0 else 0.0

        self.logger.info(f"验证步数: {self.info_steps}")
        self.logger.info(f"验证损失: {avg_loss:.4f}")
        self.logger.info(f"偏好准确率: {accuracy:.4f}")
        # 日志记录

        if self.writer is not None and self.writer_name is not None:
            self.writer.add_scalar(self.writer_name + '_test_loss', avg_loss, self.train_steps)
            self.writer.add_scalar(self.writer_name + '_test_accuracy', accuracy, self.train_steps)
            # TensorBoard 记录

        self.model.train()
        return avg_loss

    def print_info(self):
        """打印训练配置信息"""
        print("=" * 60)
        print("训练配置信息")
        print("=" * 60)

        # 模型配置
        print("📊 模型配置:")
        print(f"  ├── 解码器层数: {self.decoder_num}")
        print(f"  ├── 注意力头数: {self.head_num}")
        print(f"  ├── 隐藏层维度: {self.d}")
        print(f"  ├── KV 维度: {self.dk}")
        print(f"  ├── 前馈网络维度: {self.dff}")
        print(f"  └── 词表大小: {self.vocab_size}")

        # 训练器配置
        print("🚀 训练器配置:")
        print(f"  ├── 训练方式: {self.train_method}")
        print(f"  ├── 续训练: {self.keep_train}")
        print(f"  ├── 微调模式: {self.finetune}")
        print(f"  ├── 编译优化: {self.compile}")
        print(f"  └── 加载优化器: {self.load_optimizer}")

        # 设备配置
        print("💻 设备配置:")
        print(f"  ├── 训练设备: {self.device}")
        print(f"  └── 混合精度: {self.mixed_precision}")

        # 路径配置
        print("📁 路径配置:")
        print(f"  ├── 预训练模型目录: {self.train_model_dir}")
        print(f"  ├── 预训练模型名: {self.train_model_name}")
        print(f"  ├── 输出目录: {self.output_dir}")
        print(f"  ├── 输出模型名: {self.output_model_name}")
        print(f"  ├── 模型文件后缀: {self.model_suffix}")
        print(f"  └── 优化器文件后缀: {self.optimizer_suffix}")

        # 检查点配置
        print("💾 检查点配置:")
        print(f"  ├── 最大检查点数量: {self.max_checkpoints}")
        print(f"  └── 保存最佳检查点: {self.save_best_checkpoint}")

        # 数据集配置
        print("📚 数据集配置:")
        print(f"  ├── 训练数据路径: {self.train_data_path}")
        print(f"  ├── 验证数据路径: {self.valid_data_path}")
        print(f"  ├── 测试数据路径: {self.test_data_path}")
        print(f"  ├── 分词器路径: {self.tokenizer_path}")
        print(f"  ├── 数据加载线程数: {self.num_workers}")
        print(f"  ├── 固定内存: {self.pin_memory}")
        print(f"  └── 流式加载: {self.yield_load}")

        # 训练参数
        print("⚙️ 训练参数:")
        print(f"  ├── 总训练轮数: {self.all_epochs}")
        print(f"  ├── 批次大小: {self.batch_size}")
        print(f"  ├── 序列长度: {self.block_size}")
        print(f"  ├── 梯度累计步数: {self.accumulation_steps}")
        print(f"  └── 信息更新间隔: {self.info_update_interval}")

        # 优化器配置
        print("📈 优化器配置:")
        print(f"  ├── 优化器: {self.optimizer_name}")
        print(f"  ├── 学习率: {self.learning_rate}")
        print(f"  ├── Betas: {self.betas}")
        print(f"  ├── Epsilon: {self.eps}")
        print(f"  ├── 权重衰减: {self.weight_decay}")
        print(f"  ├── 学习率调度器: {self.lr_scheduler}")
        print(f"  ├── 预热比例: {self.pct_start}")
        print(f"  ├── 最大学习率倍率: {self.max_lr_rate}")
        print(f"  ├── 初始学习率倍率: {self.div_factor}")
        print(f"  └── 退火策略: {self.anneal_strategy}")

        # 强化学习配置
        if self.train_method in ['dpo', 'ipo', 'simpo']:
            print("🚩 RL 配置:")
            print(f"  ├── RL 奖励系数: {self.rl_beta}")
            print(f"  └── SimPO Gamma 因子: {self.gamma}")

        # 模型技术参数配置
        print("🎯 模型技术参数配置:")
        print(f"  ├── 梯度裁剪: {self.grad_clip}")
        print(f"  ├── 梯度检查点: {self.grad_checkpoint}")
        print(f"  └── Dropout: {self.dropout}")

        # 评估配置
        print("📊 评估配置:")
        print(f"  ├── 困惑度评估: {self.ppl_eval}")
        print(f"  └── BLEU-4 评估: {self.bleu_eval}")

        # 可视化配置
        print("📈 可视化配置:")
        print(f"  ├── TensorBoard: {self.tensorboard}")
        print(f"  ├── TensorBoard 目录: {self.tensorboard_dir}")
        print(f"  └── TensorBoard 日志名: {self.writer_name}")

        # 日志配置
        print("📝 日志配置:")
        print(f"  ├── 日志名称: {self.logger_name}")
        print(f"  ├── 控制台日志级别: {self.std_level}")
        print(f"  ├── 文件日志级别: {self.file_level}")
        print(f"  ├── 控制台输出: {self.std_out}")
        print(f"  ├── 保存到文件: {self.save_info}")
        print(f"  └── 日志文件名: {self.file_name}")

        print("=" * 60)

    def update_step(self):
        pass

    def _init_progress(self):
        """进度条可视化训练进度"""
        progress = Progress(
            TextColumn("[progress.description]{task.description}"),   # 显示任务的描述信息
            BarColumn(),   # 显示进度条
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),   # 设置样式,保留三位数的整数百分比,右对齐
            TimeRemainingColumn(),   # 显示基于当前进度推测估计的剩余时间
            TimeElapsedColumn(),   # 显示运行时间
            TextColumn("[bold blue]{task.fields[show_info]}"),   # 额外信息
            refresh_per_second=1,  # 每1秒钟更新一次
        )

        self.epoch_progress = progress.add_task(description='epoch: ', show_info='', total=self.all_epochs)
        # epoch进度条

        self.all_tsp = self.data_length * self.all_epochs // (self.batch_size * self.info_update_interval)
        self.tsp_progress = progress.add_task(description='steps: ', show_info='', total=self.all_tsp)
        # tsp进度条

        self.train_progress = progress   # 对象化进度条
        self.train_progress.start()   # 启动进度条

    def _init_skip_progess(self):
        """跳过批次进度条"""
        self.main_skip_progress = self.train_progress.add_task(description='skipping: ', show_info='', total=self.skip_steps)
        # 跳过批次进度条

    def exit(self, signum, frame):
        """进程退出时调用

        参数:
        - signum (int): 信号编号
        - frame (frame): 当前栈帧
        """
        if self.train_signal:
            choice = input("你正在使用 `Ctrl+C` 中断训练, 是否保存检查点? (y/n): ")
            self.train_progress.stop()

            if choice.lower() == 'y':
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_dir = os.path.join("exit_save", now+"_exit_ckpt")
                os.makedirs(save_dir, exist_ok=True)
                self._save_checkpoint(
                    dir=save_dir,
                    suffix=f"exit_{now}",
                )   # 保存检查点

                self._save_resume_page(
                    dir=save_dir,
                    suffix=f"exit_{now}",
                    now_time=now,
                )    # 保存恢复页面

                self.logger.info(f"退出训练, 检查点已保存至 {save_dir}")
                sys.exit(0)

            else:
                self.logger.info("退出训练, 不保存检查点")
                sys.exit(0)

        else:
            self.logger.info("未在训练阶段, 退出进程")
            sys.exit(0)
