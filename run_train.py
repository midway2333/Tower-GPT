from train import Trainer
from train_config import TrainerConfig

cfg = TrainerConfig(
    # 模型架构配置
    decoder_num=8,
    head_num=8,
    d=768,
    dk=64,
    dff=4096,
    vocab_size=32768,
    
    # 训练器配置
    train_method="simpo",
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
    train_model_name=None,  # 从头开始训练
    output_dir="./output",
    output_model_name="",
    model_suffix=".bin",
    optimizer_suffix=".pt",
    scheduler_suffix=".sdl",
    max_checkpoints=3,
    save_best_checkpoint=True,
    
    # 数据集配置
    train_data_path="data\\dpo.jsonl",
    valid_data_path=None,
    test_data_path=None,
    tokenizer_path="tokenizer\\tower_dict_v2.4_32768.model",
    num_workers=0,
    pin_memory=True,
    yield_load=True,

    # 训练参数配置
    all_epochs=2,
    batch_size=4,
    block_size=512,
    accumulation_steps=1,
    info_update_interval=128,
    
    # 优化器配置
    optimizer="adamw",
    learning_rate=2e-8,  # 比标准预训练稍低
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,  # 较强的权重衰减防过拟合
    
    # 学习率调度器
    lr_scheduler=True,
    pct_start=0.1,  # 10% warmup
    max_lr_rate=4.0,  # 保守的峰值
    div_factor=4.0,
    anneal_strategy="cos",
    
    # RL 配置
    rl_beta=2.5,
    gamma=1.2,

    # 模型技术参数配置
    grad_clip=1.0,  # 梯度裁剪稳定训练
    grad_checkpoint=False,
    dropout=0.1,
    
    # 评估配置
    ppl_eval=True,
    bleu_eval=False,
    
    # 可视化配置
    tensorboard=True,
    tensorboard_dir="./tensorboard",
    writer_name="trainer",
    
    # 日志配置
    logger_name="rl_trainer",
    std_level="info",
    file_level="debug",
    std_out=True,
    save_info=True,  # 保存日志便于分析
    file_name=None,  # 自动使用日期
)

if __name__ == "__main__":
    trainer = Trainer(cfg)
    trainer.train()
