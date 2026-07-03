# Mini GPT

一个极简的 decoder-only Transformer 语言模型，用于学习 LLM 核心原理。

## 架构

```
Embedding (Token + 位置编码)
    ↓
┌─────────────────────────┐  × N 层
│ LayerNorm               │
│ Causal Self-Attention   │  ← 多头缩放点积注意力 + causal mask
│    ↓                    │
│ LayerNorm               │
│ FeedForward             │  ← Linear → ReLU → Linear
│    + 残差连接            │
└─────────────────────────┘
    ↓
LayerNorm → Linear → softmax → 输出
```

### 组件说明

| 组件 | 说明 |
|------|------|
| **Token Embedding** | 将离散 token 映射为稠密向量 |
| **位置编码** | 可学习的位置编码，注入序列位置信息 |
| **因果自注意力** | 每个位置只能看自己和之前的位置，保证自回归 |
| **FeedForward** | 两层全连接 + ReLU，提供非线性变换 |
| **LayerNorm** | 标准 LayerNorm（非 RMSNorm），稳定训练 |
| **残差连接** | 缓解深层网络梯度消失问题 |

## 依赖

- Python 3.10+
- PyTorch 2.0+

```bash
pip install torch
```

## 使用

### 1. 下载数据集

```bash
make data
# 或: python minigpt.py --download
```

自动下载 [Tiny Shakespeare](https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt) 数据集。

### 2. 训练

```bash
# 默认训练（1000 步）
make train

# 自定义步数
make train ARGS="--max-iters 5000"

# 指定超参数
make train ARGS="--max-iters 2000 --lr 5e-4 --batch-size 128"
```

训练过程会打印：

```
  Iter | Train Loss |  Val PPL |     Time |         LR
=================================================================
     0 |     4.2891 |          |     0.2s |   0.000300
    10 |     3.4668 |          |     0.3s |   0.000300
    20 |     3.1132 |          |     0.4s |   0.000300
  ───> |            |    18.10 |     0.8s |  ← 验证集 PPL
```

### 3. 续训

```bash
# 从 checkpoint 继续训练
make resume ARGS="--max-iters 3000"
```

Checkpoint 每 `--eval-interval` 步自动保存，包含模型权重、优化器状态和训练步数。

### 4. 生成文本

```bash
# 默认 prompt
make generate

# 自定义 prompt
make generate ARGS="--prompt \"To be or not to be\""

# 调整生成参数
make generate ARGS="--prompt \"O Romeo\" --temperature 0.6 --top-k 20"
```

### 5. 全部做完

```bash
make all        # 下载数据 → 训练 → 生成示例
```

## 命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--download` | — | 下载数据集 |
| `--train` | — | 训练模式 |
| `--generate` | — | 生成模式 |
| `--resume` | — | 从 checkpoint 续训 |
| `--max-iters` | 1000 | 训练迭代步数 |
| `--lr` | 3e-4 | 学习率 |
| `--batch-size` | 64 | 批次大小 |
| `--log-interval` | 10 | 每 N 步打印训练 loss |
| `--eval-interval` | 200 | 每 N 步计算验证 PPL |
| `--prompt` | "O Romeo" | 生成提示词 |
| `--max-new-tokens` | 200 | 最大生成长度 |
| `--temperature` | 0.8 | 采样温度（0=greedy） |
| `--top-k` | 40 | Top-K 采样 |
| `--device` | auto | 训练设备（cuda/cpu） |

## 超参数

默认配置 (`d_model=192, n_layers=6, n_heads=6, d_ff=768`)：

| 参数 | 值 | 说明 |
|------|-----|------|
| vocab_size | 67 | 字符级词表 |
| d_model | 192 | 模型维度 |
| n_layers | 6 | Transformer 层数 |
| n_heads | 6 | 注意力头数 |
| d_ff | 768 | FFN 隐藏层维度 |
| max_seq_len | 128 | 最大序列长度 |
| dropout | 0.1 | Dropout |
| 参数量 | **2.7M** | 约 270 万 |

## 文件结构

```
├── minigpt.py              # 模型代码 + 训练 + 生成
├── README.md               # 本文件
├── Makefile                # 常用命令
├── data/
│   └── tinyshakespeare.txt # 数据集
├── minigpt.pt              # 训练好的模型权重
└── minigpt_checkpoint.pt   # 训练 checkpoint（含优化器状态）
```

## 参考资料

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer 论文
- [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) — OpenAI GPT-2
- [Andrej Karpathy: Let's build GPT](https://www.youtube.com/watch?v=kCc8FmEb1nY) — 视频教程
