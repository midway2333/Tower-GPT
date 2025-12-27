from train import Trainer
from train_config import TrainerConfig

cfg = TrainerConfig(
    # 模型架构配置
    decoder_num=6,
    head_num=6,
    d=384,
    dk=64,
    dff=1536,
    vocab_size=32768,
    
    # 训练器配置
    train_method="chat",  # 蒸馏数据通常是对话格式
    keep_train=False,
    ckpt_path=None,
    finetune=False,
    compile=False,  # 如果PyTorch >= 2.0
    load_optimizer=False,
    
    # 设备配置
    device="auto",
    mixed_precision="bf16",  # 会根据设备自动降级
    
    # 模型路径配置
    train_model_dir=None,  # 从头开始训练
    train_model_name=None,
    output_dir="./output",
    output_model_name="tower_gpt_23m_distilled",
    model_suffix=".bin",
    optimizer_suffix=".pt",
    scheduler_suffix=".sdl",
    max_checkpoints=5,  # 多保存几个方便选择
    save_best_checkpoint=True,
    
    # 数据集配置
    train_data_path="data\\sft.jsonl",  # 你的1GB蒸馏数据
    valid_data_path="data\\val.jsonl",  # 建议保留10%作为验证集
    test_data_path=None,
    tokenizer_path="tokenizer\\tower_dict_v2.4_32768.model",  # 分词器路径
    num_workers=0,  # 小数据量不需要太多worker
    pin_memory=True,
    yield_load=True,

    # 训练参数配置
    all_epochs=4,  # 蒸馏数据可以多训几轮
    batch_size=8,
    block_size=512,  # 适合对话长度
    accumulation_steps=16,  # 有效batch = 32×4 = 128
    info_update_interval=2048,  # 更频繁地监控
    
    # 优化器配置
    optimizer="adamw",
    learning_rate=3e-4,  # 比标准预训练稍低
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,  # 较强的权重衰减防过拟合
    
    # 学习率调度器
    lr_scheduler=True,
    pct_start=0.1,  # 10% warmup
    max_lr_rate=15.0,  # 保守的峰值
    div_factor=75.0,
    anneal_strategy="cos",
    
    # 模型技术参数配置
    grad_clip=1.0,  # 梯度裁剪稳定训练
    grad_checkpoint=False,  # 23M模型不需要
    dropout=0.2,  # 较高的dropout防过拟合
    
    # 评估配置
    ppl_eval=True,
    bleu_eval=False,  # 对话模型评估BLEU
    
    # 可视化配置
    tensorboard=True,
    tensorboard_dir="./tensorboard",  # 会自动使用./runs/{output_model_name}
    writer_name="distill_train",
    
    # 日志配置
    logger_name="distill_trainer",
    std_level="info",
    file_level="debug",
    std_out=True,
    save_info=True,  # 保存日志便于分析
    file_name=None,  # 自动使用日期
)

if __name__ == "__main__":
    trainer = Trainer(cfg)
    trainer.train()
