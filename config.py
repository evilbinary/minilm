"""
Mini GPT 配置文件 — 修改此文件即可调整模型和训练参数。

用法：
    from config import get_config, GPTConfig
    cfg = get_config(vocab_size=4726)
"""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    """模型架构参数"""
    vocab_size: int = 100       # 词表大小（自动计算，无需修改）
    max_seq_len: int = 128      # 最大序列长度
    d_model: int = 320          # 模型维度 ↑ 越大模型越宽
    n_layers: int = 10          # Transformer 层数 ↑ 越大模型越深
    n_heads: int = 8            # 注意力头数（需能被 d_model 整除）
    d_ff: int = 1408            # FFN 隐藏层维度 ↑ 越大每层越宽
    dropout: float = 0.2        # Dropout 概率（0.1~0.3，过拟合时调大）
    bias: bool = True           # Linear 层是否使用 bias


@dataclass
class TrainConfig:
    """训练参数"""
    max_iters: int = 20000       # 总训练步数
    lr: float = 5e-4            # 峰值学习率
    batch_size: int = 64        # 批次大小（越大梯度越稳）
    log_interval: int = 10      # 每 N 步打印 loss
    eval_interval: int = 200    # 每 N 步验证 + 保存
    warmup_ratio: float = 0.05  # warmup 占总步数比例
    hold_ratio: float = 0.25    # 恒定 LR 占总步数比例


# ── 常用模型规格 ──────────────────────────────────
PRESETS = {
    "4.5M":  dict(d_model=192,  n_layers=6,  n_heads=6, d_ff=768,  dropout=0.1),
    "8.8M":  dict(d_model=256,  n_layers=8,  n_heads=8, d_ff=1024, dropout=0.15),
    "16M":   dict(d_model=320,  n_layers=10, n_heads=8, d_ff=1408, dropout=0.2),   # ← 默认
    "25M":   dict(d_model=384,  n_layers=12, n_heads=8, d_ff=1792, dropout=0.2),
    "40M":   dict(d_model=512,  n_layers=12, n_heads=8, d_ff=2048, dropout=0.2),
}


def get_config(vocab_size: int = 100, preset: str = None,
               **overrides) -> GPTConfig:
    """
    获取模型配置。

    参数:
        vocab_size: 词表大小（由数据决定）
        preset:     使用预置规格 ("4.5M" / "16M" / "25M" 等)
        **overrides: 直接覆盖具体参数，优先级最高

    示例:
        cfg = get_config(vocab_size=5000)
        cfg = get_config(vocab_size=5000, preset="25M")
        cfg = get_config(vocab_size=5000, d_model=384, n_layers=8)
    """
    params = {}
    if preset and preset in PRESETS:
        params.update(PRESETS[preset])
    params.update(overrides)
    return GPTConfig(vocab_size=vocab_size, **params)
