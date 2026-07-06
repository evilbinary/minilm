"""
Mini GPT 配置文件 — 修改此文件即可调整模型和训练参数。

用法：
    from config import get_config, GPTConfig
    cfg = get_config(vocab_size=16000)
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """模型架构参数"""
    vocab_size: int = 16000     # BPE 词表大小
    max_seq_len: int = 512      # 最大序列长度（对话需要更长）
    d_model: int = 768          # 模型维度
    n_layers: int = 12          # Transformer 层数
    n_heads: int = 12           # 注意力头数（需能被 d_model 整除）
    d_ff: int = 3072            # FFN 隐藏层维度
    dropout: float = 0.1        # Dropout 概率
    bias: bool = True           # Linear 层是否使用 bias


@dataclass
class TrainConfig:
    """训练参数"""
    max_iters: int = 20000       # 总训练步数
    lr: float = 3e-4            # 峰值学习率
    batch_size: int = 32         # 批次大小
    log_interval: int = 10       # 每 N 步打印 loss
    eval_interval: int = 200     # 每 N 步验证 + 保存
    warmup_ratio: float = 0.05   # warmup 占总步数比例
    hold_ratio: float = 0.25     # 恒定 LR 占总步数比例


# ── 常用模型规格 ──────────────────────────────────
PRESETS = {
    "4.5M":  dict(vocab_size=4726,  max_seq_len=128, d_model=192,  n_layers=6,  n_heads=6,  d_ff=768,   dropout=0.1),
    "16M":   dict(vocab_size=4726,  max_seq_len=128, d_model=320,  n_layers=10, n_heads=8,  d_ff=1408,  dropout=0.2),
    "40M":   dict(vocab_size=4726,  max_seq_len=128, d_model=512,  n_layers=12, n_heads=8,  d_ff=2048,  dropout=0.2),
    # ── BPE 对话模型 ─────────────────────────────
    "100M":  dict(vocab_size=16000, max_seq_len=512, d_model=768,  n_layers=12, n_heads=12, d_ff=3072,  dropout=0.1),
    "200M":  dict(vocab_size=16000, max_seq_len=512, d_model=1024, n_layers=16, n_heads=16, d_ff=4096,  dropout=0.1),
    "400M":  dict(vocab_size=16000, max_seq_len=512, d_model=1280, n_layers=20, n_heads=16, d_ff=5120,  dropout=0.1),
}


def get_config(vocab_size: int = 16000, preset: str = None,
               **overrides) -> GPTConfig:
    """
    获取模型配置。

    参数:
        vocab_size: 词表大小（BPE tokenizer 决定）
        preset:     使用预置规格
        **overrides: 直接覆盖具体参数

    示例:
        cfg = get_config(preset="100M")
        cfg = get_config(preset="200M", d_model=896)
    """
    params = dict(vocab_size=vocab_size, max_seq_len=512, d_model=768,
                  n_layers=12, n_heads=12, d_ff=3072, dropout=0.1)
    if preset and preset in PRESETS:
        params.update(PRESETS[preset])
    params.update(overrides)
    return GPTConfig(**params)
