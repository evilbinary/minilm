"""
Mini GPT — 极简版 GPT，用于学习 LLM 核心原理。

只保留最核心的组件：
  1. Token Embedding + 位置编码
  2. 多头因果自注意力
  3. 两层 FFN (Linear + ReLU + Linear)
  4. Layer Norm + 残差连接

Usage:
    python minigpt.py              # 训练 + 生成演示
    python minigpt.py --train      # 仅训练
    python minigpt.py --generate   # 仅生成（需已有模型）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
from dataclasses import dataclass
from typing import Optional


# ═══════════════════════════════════════════════════════
#  1. 配置
# ═══════════════════════════════════════════════════════

@dataclass
class GPTConfig:
    vocab_size: int = 100      # 词表大小
    max_seq_len: int = 128     # 最大序列长度
    d_model: int = 128         # 模型维度
    n_layers: int = 4          # Transformer 层数
    n_heads: int = 4           # 注意力头数
    d_ff: int = 512            # FFN 隐藏层维度
    dropout: float = 0.1       # Dropout


# ═══════════════════════════════════════════════════════
#  2. 组件
# ═══════════════════════════════════════════════════════

class PositionalEmbedding(nn.Module):
    """学习式位置编码"""

    def __init__(self, d_model: int, max_seq_len: int):
        super().__init__()
        self.pe = nn.Embedding(max_seq_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, S, D)
        S = x.size(1)
        positions = torch.arange(S, device=x.device)
        return x + self.pe(positions)  # (B, S, D) + (S, D) broadcast


class CausalSelfAttention(nn.Module):
    """多头因果自注意力"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # Q, K, V 投影合并到一个矩阵（效率考虑，但原理上等价于三个独立 Linear）
        self.qkv = nn.Linear(config.d_model, 3 * config.d_model)
        self.out = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

        # Causal mask: 上三角为 0（表示看不到未来）
        mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
        self.register_buffer("causal_mask", mask.view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape

        # 生成 QKV
        qkv = self.qkv(x)                                               # (B, S, 3D)
        qkv = qkv.reshape(B, S, 3, self.n_heads, self.head_dim)        # (B, S, 3, nh, hd)
        qkv = qkv.permute(2, 0, 3, 1, 4)                                # (3, B, nh, S, hd)
        q, k, v = qkv[0], qkv[1], qkv[2]                                # each (B, nh, S, hd)

        # Scaled Dot-Product Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale                        # (B, nh, S, S)
        attn = attn.masked_fill(self.causal_mask[:, :, :S, :S] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # 加权聚合
        y = attn @ v                                                     # (B, nh, S, hd)
        y = y.transpose(1, 2).reshape(B, S, D)                          # (B, S, D)
        y = self.out(y)
        return y


class FeedForward(nn.Module):
    """经典 FFN: Linear -> ReLU -> Linear"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.w1 = nn.Linear(config.d_model, config.d_ff)
        self.w2 = nn.Linear(config.d_ff, config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.relu(self.w1(x))))


class TransformerBlock(nn.Module):
    """一个 Decoder Block: Norm -> Attn -> + -> Norm -> FFN -> +"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.d_model)  # 标准 LayerNorm
        self.attn = CausalSelfAttention(config)
        self.ln2 = nn.LayerNorm(config.d_model)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm: 先 norm 再进子层，然后残差相加
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ═══════════════════════════════════════════════════════
#  3. GPT 模型
# ═══════════════════════════════════════════════════════

class MiniGPT(nn.Module):
    """Decoder-only 语言模型"""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embedding = PositionalEmbedding(config.d_model, config.max_seq_len)
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_final = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)

        # 初始化
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """简单初始化"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        idx: (B, S)          — 输入 token ids
        targets: (B, S)      — 目标 token ids（训练时）
        返回: {logits, loss}
        """
        _, S = idx.shape
        assert S <= self.config.max_seq_len

        # Embedding
        x = self.token_embedding(idx)        # (B, S, D)
        x = self.pos_embedding(x)            # + 位置编码

        # Transformer 层
        for block in self.blocks:
            x = block(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)             # (B, S, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: Optional[int] = None,
                 eos_id: Optional[int] = None):
        """
        自回归生成。
        idx: (B, S) — 初始 prompt
        返回: (B, S+新token)
        """
        self.eval()
        for _ in range(max_new_tokens):
            # 只保留最后 max_seq_len 个 token（窗口）
            idx_cond = idx[:, -self.config.max_seq_len:]

            # 预测下一个 token
            logits = self(idx_cond)["logits"][:, -1, :]    # (B, V)

            # 采样
            if temperature > 0:
                logits = logits / temperature
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    vals, _ = torch.topk(logits, top_k)
                    logits[logits < vals[:, -1:]] = float("-inf")
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (B, 1)
            else:
                next_token = logits.argmax(dim=-1, keepdim=True)

            idx = torch.cat([idx, next_token], dim=-1)

            if eos_id is not None and (next_token == eos_id).any():
                break

        self.train()
        return idx

    @torch.no_grad()
    def evaluate_perplexity(self, data_loader, max_batches: int = 50) -> float:
        """在验证集上计算 perplexity（最多取 max_batches 个 batch，避免死循环）"""
        self.eval()
        total_loss = 0.0
        total_tokens = 0
        for _ in range(max_batches):
            x, y = next(data_loader)
            x, y = x.to(self.lm_head.weight.device), y.to(self.lm_head.weight.device)
            out = self(x, targets=y)
            n_tokens = y.numel()
            total_loss += out["loss"].item() * n_tokens
            total_tokens += n_tokens
        ppl = math.exp(total_loss / total_tokens) if total_tokens > 0 else float("inf")
        self.train()
        return ppl


# ═══════════════════════════════════════════════════════
#  4. 数据（字符级）
# ═══════════════════════════════════════════════════════

class CharTokenizer:
    """字符级 tokenizer"""

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars) + 2       # + PAD, EOS
        self.stoi = {c: i + 2 for i, c in enumerate(chars)}
        self.itos = {i + 2: c for i, c in enumerate(chars)}
        self.itos[0] = "<PAD>"
        self.itos[1] = "<EOS>"
        self.pad_id = 0
        self.eos_id = 1

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(c, self.pad_id) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos.get(i, "") for i in ids if i not in (0, 1))


DATA_PATH = "data/tinyshakespeare.txt"
CKPT_PATH = "minigpt_checkpoint.pt"


def save_checkpoint(path: str, model: MiniGPT, optimizer, step: int, best_loss: float):
    """保存 checkpoint（模型 + 优化器 + 训练状态）"""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_loss": best_loss,
    }, path)
    print(f"  [Checkpoint] 已保存到 {path} (step={step})")


def load_checkpoint(path: str, model: MiniGPT, optimizer=None, device="cpu"):
    """加载 checkpoint，返回 (step, best_loss)"""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    print(f"  [Checkpoint] 已加载 {path} (step={ckpt['step']})")
    return ckpt["step"], ckpt["best_loss"]


def get_data() -> str:
    """从本地文件读取 Tiny Shakespeare 数据集"""
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"数据文件 {DATA_PATH} 不存在！")
        print(f"请先运行 python minigpt.py --download 下载数据集")
        raise


def download_data():
    """下载 Tiny Shakespeare 数据集到本地"""
    import urllib.request
    import os
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    os.makedirs("data", exist_ok=True)
    print(f"正在下载 Tiny Shakespeare 数据集...")
    urllib.request.urlretrieve(url, DATA_PATH)
    print(f"已保存到 {DATA_PATH}")


def make_dataloaders(text: str, tokenizer: CharTokenizer,
                     config: GPTConfig, batch_size: int = 32):
    """切分训练/验证集，返回迭代器"""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data, val_data = data[:n], data[n:]

    def get_batch(src):
        ix = torch.randint(len(src) - config.max_seq_len, (batch_size,))
        x = torch.stack([src[i:i + config.max_seq_len] for i in ix])
        y = torch.stack([src[i + 1:i + config.max_seq_len + 1] for i in ix])
        return x, y

    class DataLoader:
        def __init__(self, src): self.src = src
        def __iter__(self): return self
        def __next__(self): return get_batch(self.src)

    return DataLoader(train_data), DataLoader(val_data)


# ═══════════════════════════════════════════════════════
#  5. 训练
# ═══════════════════════════════════════════════════════

def train(model: MiniGPT, train_loader, val_loader,
          max_iters: int = 2000, lr: float = 1e-3,
          eval_interval: int = 200, log_interval: int = 10,
          resume_from: Optional[str] = None,
          device: str = "cpu"):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # 加载 checkpoint
    start_step = 0
    best_loss = float("inf")
    if resume_from is not None:
        start_step, best_loss = load_checkpoint(resume_from, model, optimizer, device)
        # 把优化器的 lr 重新设一下（因为加载后会覆盖）
        for g in optimizer.param_groups:
            g["lr"] = lr

    print(f"\n设备: {device}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"起始步: {start_step}, 目标步数: {max_iters}")
    print(f"{'='*65}")
    print(f"{'Iter':>6} | {'Train Loss':>10} | {'Val PPL':>8} | {'Time':>8} | {'LR':>10}")
    print(f"{'='*65}")

    start = time.time()
    for step in range(start_step, max_iters):
        x, y = next(train_loader)
        x, y = x.to(device), y.to(device)

        loss = model(x, targets=y)["loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 每 log_interval 步打印训练 loss
        if step % log_interval == 0 or step == max_iters - 1:
            elapsed = time.time() - start
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{step:>6} | {loss.item():>10.4f} | {'':>8} | {elapsed:>7.1f}s | {lr_now:>10.6f}")

        # 每 eval_interval 步计算验证集 PPL + 保存 checkpoint
        if (step > 0 and step % eval_interval == 0) or step == max_iters - 1:
            ppl = model.evaluate_perplexity(val_loader)
            elapsed = time.time() - start
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"{'───>':>6} | {'':>10} | {ppl:>8.2f} | {elapsed:>7.1f}s | {'':>10}")

            if loss.item() < best_loss:
                best_loss = loss.item()
            # 保存 checkpoint
            save_checkpoint(CKPT_PATH, model, optimizer, step, best_loss)

    print(f"{'='*65}")
    print("训练完成!")


# ═══════════════════════════════════════════════════════
#  6. 入口
# ═══════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true", help="下载数据集")
    parser.add_argument("--train", action="store_true", help="训练")
    parser.add_argument("--generate", action="store_true", help="生成")
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-interval", type=int, default=10, help="每 N 步打印训练 loss")
    parser.add_argument("--eval-interval", type=int, default=200, help="每 N 步计算验证集 PPL")
    parser.add_argument("--prompt", type=str, default="O Romeo")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 继续训练")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    # 仅下载
    if args.download:
        download_data()
        return

    do_train = args.train or not args.generate
    do_generate = args.generate or not args.train

    # 数据（从本地文件读取）
    text = get_data()
    tokenizer = CharTokenizer(text)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        d_model=192,
        n_layers=6,
        n_heads=6,
        d_ff=768,
    )

    model = MiniGPT(config).to(args.device)

    if do_train:
        print(f"数据: {len(text)} 字符, 词表: {tokenizer.vocab_size}")
        train_loader, val_loader = make_dataloaders(text, tokenizer, config, args.batch_size)
        train(model, train_loader, val_loader, args.max_iters, args.lr,
              eval_interval=args.eval_interval, log_interval=args.log_interval,
              resume_from=CKPT_PATH if args.resume else None,
              device=args.device)
        # 保存最终模型（仅权重，供生成用）
        torch.save(model.state_dict(), "minigpt.pt")
        print(f"最终模型已保存到 minigpt.pt")

    if do_generate:
        if not do_train:
            try:
                model.load_state_dict(torch.load("minigpt.pt", map_location=args.device))
            except FileNotFoundError:
                print("未找到 minigpt.pt，使用随机参数生成")

        prompt_ids = torch.tensor(tokenizer.encode(args.prompt), dtype=torch.long).unsqueeze(0).to(args.device)
        out_ids = model.generate(prompt_ids, args.max_new_tokens, args.temperature, args.top_k, tokenizer.eos_id)
        generated = tokenizer.decode(out_ids[0].tolist())

        print(f"\nPrompt: {args.prompt}")
        print("─" * 50)
        print(generated)
        print("─" * 50)


if __name__ == "__main__":
    main()
