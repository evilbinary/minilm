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
import os
from typing import Optional
import glob

from config import GPTConfig, TrainConfig, get_config


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

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None,
                loss_mask: Optional[torch.Tensor] = None):
        """
        idx: (B, S)          — 输入 token ids
        targets: (B, S)      — 目标 token ids（训练时）
        loss_mask: (B, S)    — 1=计算loss, 0=忽略（对话时只算 assistant 部分）
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
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                   reduction='none')  # 逐位置 loss
            if loss_mask is not None:
                loss = (loss * loss_mask.view(-1)).sum() / loss_mask.view(-1).sum().clamp(min=1)
            else:
                loss = loss.mean()

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
            batch = next(data_loader)
            x, y = batch[0].to(self.lm_head.weight.device), batch[1].to(self.lm_head.weight.device)
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


DATA_DIR = "data"
# 数据文件列表（每个语言可指定多个文件，全部用于训练）
DATA_FILES = {
    "en": ["data/tinyshakespeare.txt"],
    "zh": ["data/xyj.txt", "data/hlm.txt"],  # 西游记 + 红楼梦
}
DATA_URLS = {
    "en": ["https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"],
    "zh": ["", ""],  # 空表示手动下载
}
CKPT_PATH = "checkpoint/minigpt_checkpoint.pt"


def save_checkpoint(path: str, model: MiniGPT, optimizer, step: int, best_loss: float,
                    data_files: Optional[list] = None, vocab: Optional[dict] = None):
    """保存 checkpoint（模型 + 优化器 + 训练状态 + 词表）"""
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "best_loss": best_loss,
        "config": model.config,
        "data_files": data_files,
        "vocab": vocab,  # stoi 映射，保证续训时数据一致
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


def get_data_paths(lang: str = "en") -> list[str]:
    """获取数据文件路径列表（展平多文件配置）"""
    if lang == "both":
        return DATA_FILES["en"] + DATA_FILES["zh"]
    return DATA_FILES.get(lang, DATA_FILES["en"])


def get_data(lang: str = "en") -> str:
    """从本地文件读取数据集（支持多语言合并）"""
    paths = get_data_paths(lang)
    texts = []
    for path in paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                texts.append(f.read())
        except FileNotFoundError:
            print(f"数据文件 {path} 不存在！")
            print(f"请先下载数据集后再试")
            raise
    return "\n".join(texts)


def download_data(lang: str = "en"):
    """下载数据集到本地（已有则跳过）"""
    import urllib.request
    import os
    os.makedirs(DATA_DIR, exist_ok=True)
    for key in ([lang] if lang != "both" else ["en", "zh"]):
        urls = DATA_URLS.get(key, [])
        paths = DATA_FILES.get(key, [])
        for i, path in enumerate(paths):
            fname = os.path.basename(path)
            if os.path.exists(path):
                print(f"  [{key}] {fname} 已存在")
                continue
            url = urls[i] if i < len(urls) else ""
            if not url:
                print(f"  [{key}] {fname} 未找到，请手动放入 {path}")
                continue
            print(f"  正在下载 {fname}...")
            urllib.request.urlretrieve(url, path)
            print(f"  已保存 {path}")


def make_dataloaders(text: str, tokenizer,
                     config: GPTConfig, batch_size: int = 32):
    """创建训练/验证 DataLoader"""
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    # 切分成不重叠的 segment，打散后按 9:1 切分
    seg_len = config.max_seq_len + 1  # 多 1 留给 target
    n_seg = len(data) // seg_len
    data = data[:n_seg * seg_len].view(n_seg, seg_len)
    indices = torch.randperm(n_seg)
    n_train = int(n_seg * 0.9)
    train_seg, val_seg = data[indices[:n_train]], data[indices[n_train:]]

    def get_batch(src):
        ix = torch.randint(len(src), (batch_size,))
        x = torch.stack([src[i, :-1] for i in ix])
        y = torch.stack([src[i, 1:] for i in ix])
        return x, y

    class DataLoader:
        def __init__(self, src): self.src = src
        def __iter__(self): return self
        def __next__(self): return get_batch(self.src)

    return DataLoader(train_seg), DataLoader(val_seg)


def make_dialogue_dataloaders(text: str, tokenizer,
                               config: GPTConfig, batch_size: int = 32,
                               assistant_id: int = 4, end_id: int = 2):
    """
    对话格式 DataLoader — 生成 (input, target, loss_mask)。

    loss_mask: 只在 assistant 回复部分计算 loss，忽略 user 输入。
    """
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    seg_len = config.max_seq_len + 1
    n_seg = max(1, len(data) // seg_len)
    # 取整并 reshape
    n_tokens = n_seg * seg_len
    data = data[:n_tokens].view(n_seg, seg_len)
    indices = torch.randperm(n_seg)
    # 小数据集保证至少 1 条训练、1 条验证
    n_train = max(1, int(n_seg * 0.9)) if n_seg > 1 else 1
    n_train = min(n_train, n_seg - 1)   # 至少留 1 条给验证
    train_seg, val_seg = data[indices[:n_train]], data[indices[n_train:]]

    def make_mask(seg: torch.Tensor) -> torch.Tensor:
        """构建 loss_mask: 1=assistant 回答部分, 0=user 输入"""
        mask = torch.zeros(seg_len, dtype=torch.float)
        in_assistant = False
        for i in range(seg_len):
            tok = seg[i].item()
            if tok == assistant_id:
                in_assistant = True
                mask[i] = 0.0  # <|assistant|> 本身不算
            elif tok == end_id:
                in_assistant = False
                mask[i] = 0.0  # <|end|> 本身不算
            elif in_assistant:
                mask[i] = 1.0
        return mask

    def get_batch(src):
        ix = torch.randint(len(src), (batch_size,))
        x = torch.stack([src[i, :-1] for i in ix])
        y = torch.stack([src[i, 1:] for i in ix])
        m = torch.stack([make_mask(src[i])[:-1] for i in ix])
        return x, y, m

    class DataLoader:
        def __init__(self, src): self.src = src
        def __iter__(self): return self
        def __next__(self):
            x, y, m = get_batch(self.src)
            return x, y, m

    return DataLoader(train_seg), DataLoader(val_seg)


# ═══════════════════════════════════════════════════════
#  5. 训练
# ═══════════════════════════════════════════════════════

def train(model: MiniGPT, train_loader, val_loader,
          max_iters: int = 2000, lr: float = 1e-3,
          eval_interval: int = 200, log_interval: int = 10,
          resume_from: Optional[str] = None,
          ckpt_path: str = CKPT_PATH, data_files: Optional[list] = None,
          vocab: Optional[dict] = None, device: str = "cpu"):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))

    # 加载 checkpoint
    start_step = 0
    best_loss = float("inf")
    if resume_from is not None:
        start_step, best_loss = load_checkpoint(resume_from, model, optimizer, device)
        # 把优化器的 lr 重新设一下
        for g in optimizer.param_groups:
            g["lr"] = lr

    # 三段式学习率: warmup → 恒定 → 余弦衰减
    warmup_iters = max_iters // 20           # 5% warmup
    hold_iters = max_iters // 4              # 25% 恒定峰值 LR
    import math as _math

    def get_lr(step: int) -> float:
        if step < warmup_iters:
            return lr * (step + 1) / warmup_iters
        if step < warmup_iters + hold_iters:
            return lr
        # 余弦衰减到 0
        decay_steps = max_iters - warmup_iters - hold_iters
        progress = (step - warmup_iters - hold_iters) / max(1, decay_steps)
        return lr * 0.1 + 0.9 * lr * (1 + _math.cos(_math.pi * progress)) / 2

    print(f"设备: {device}")
    print(f"起始步: {start_step}, 目标步数: {max_iters}")
    print(f"{'='*65}")
    print(f"{'Iter':>6} | {'Train Loss':>10} | {'Val PPL':>8} | {'Time':>8} | {'LR':>10}")
    print(f"{'='*65}")

    start = time.time()
    for step in range(start_step, max_iters):
        batch = next(train_loader)
        x, y = batch[0].to(device), batch[1].to(device)
        m = batch[2].to(device) if len(batch) > 2 else None

        loss = model(x, targets=y, loss_mask=m)["loss"]

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪
        optimizer.step()

        # 更新学习率
        for g in optimizer.param_groups:
            g["lr"] = get_lr(step)

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
            save_checkpoint(ckpt_path, model, optimizer, step, best_loss,
                            data_files=data_files, vocab=vocab)

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
    parser.add_argument("--max-iters", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--log-interval", type=int, default=10, help="每 N 步打印训练 loss")
    parser.add_argument("--eval-interval", type=int, default=200, help="每 N 步计算验证集 PPL")
    parser.add_argument("--prompt", type=str, default="O Romeo")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--resume", action="store_true", help="从 checkpoint 继续训练")
    parser.add_argument("--lang", choices=["en", "zh", "both"], default="en", help="语言: en=英文, zh=中文, both=中英混合")
    parser.add_argument("--mode", choices=["completion", "dialogue"], default="completion",
                        help="训练模式: completion=续写, dialogue=对话")
    parser.add_argument("--dialogue-data", type=str, nargs="+", default=None,
                        help="对话数据文件（多个 JSONL 用空格隔开）")
    parser.add_argument("--preset", type=str, default=None, help="模型规格: 4.5M/16M/40M/100M/200M")
    parser.add_argument("--d-model", type=int, default=None, help="模型维度（覆盖 preset）")
    parser.add_argument("--n-layers", type=int, default=None, help="Transformer 层数")
    parser.add_argument("--n-heads", type=int, default=None, help="注意力头数")
    parser.add_argument("--d-ff", type=int, default=None, help="FFN 隐藏层维度")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    # 语言相关配置
    lang = args.lang
    os.makedirs("checkpoint", exist_ok=True)
    ckpt_path = f"checkpoint/minigpt_{'dialogue' if args.mode == 'dialogue' else lang}_checkpoint.pt"
    model_path = f"checkpoint/minigpt_{'dialogue' if args.mode == 'dialogue' else lang}.pt"
    lang_prompts = {"en": "O Romeo", "zh": "话说唐僧", "both": "Hello 你好"}
    default_prompt = lang_prompts.get(lang, "O Romeo")

    # 仅下载
    if args.download:
        download_data(lang)
        return

    do_train = args.train or not args.generate
    do_generate = args.generate or not args.train

    # 续训时从 checkpoint 读取数据文件列表
    data_files = None
    config = None
    if args.resume and os.path.exists(ckpt_path):
        ckpt_data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        config = ckpt_data.get("config")
        data_files = ckpt_data.get("data_files")
        if config is None:
            print("  ⚠ checkpoint 中无配置，从权重反推")

    # ── 模式: completion(续写) vs dialogue(对话) ──
    is_dialogue = args.mode == "dialogue"

    # 数据
    if data_files:
        text = "\n".join(open(f, encoding="utf-8").read() for f in data_files)
    elif is_dialogue:
        dia_paths = args.dialogue_data or ["data/dialogue_zh.txt"]
        # 展开目录：扫描所有 .jsonl 文件
        expanded = []
        for p in dia_paths:
            if os.path.isdir(p):
                expanded.extend(sorted(glob.glob(os.path.join(p, "*.jsonl"))))
            else:
                expanded.append(p)
        dia_paths = expanded
        # 检查数据
        all_exist = all(os.path.exists(p) for p in dia_paths)
        if not all_exist and dia_paths == ["data/dialogue_zh.txt"]:
            from prepare_data import generate_simple_zh
            generate_simple_zh(dia_paths[0], repeat=50)
        # JSONL 统一转换
        jsonl_files = [p for p in dia_paths if p.endswith(".jsonl")]
        if jsonl_files:
            from prepare_data import convert_jsonl
            convert_jsonl(jsonl_files, "data/dialogue_train.txt")
            dia_paths = ["data/dialogue_train.txt"]
        text = "\n".join(open(p, encoding="utf-8").read() for p in dia_paths)
    else:
        text = get_data(lang)
        if config:
            tok_c = CharTokenizer(text)
            if tok_c.vocab_size > config.vocab_size:
                keep = [f for f in get_data_paths(lang) if "hlm" not in f]
                text = "\n".join(open(f, encoding="utf-8").read() for f in keep)

    # Tokenizer（对话模式用 BPE）
    if is_dialogue:
        from tokenizer import load_tokenizer, train_tokenizer
        try:
            tokenizer = load_tokenizer("checkpoint/tokenizer.json")
        except FileNotFoundError:
            tokenizer = train_tokenizer(get_data_paths(lang), save_path="checkpoint/tokenizer.json")
    else:
        tokenizer = CharTokenizer(text)

    if config is None:
        overrides = {}
        for k in ["d_model", "n_layers", "n_heads", "d_ff"]:
            v = getattr(args, k.replace("-", "_"))
            if v is not None:
                overrides[k] = v
        vs = tokenizer.vocab_size
        config = get_config(vocab_size=vs, preset=args.preset, **overrides)

    model = MiniGPT(config).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*54}")
    print(f"  模式: {'对话' if is_dialogue else '续写'}")
    print(f"  模型: {config.d_model}x{config.n_layers}  |  {n_params/1e6:.1f}M 参数")
    print(f"  结构: d_model={config.d_model}  n_layers={config.n_layers}")
    print(f"         n_heads={config.n_heads}  d_ff={config.d_ff}  dropout={config.dropout}")
    print(f"{'='*54}\n")

    if do_train:
        tcfg = TrainConfig()
        max_iters = args.max_iters or tcfg.max_iters
        lr = args.lr or tcfg.lr
        batch_size = args.batch_size or tcfg.batch_size

        print(f"数据: {len(text)} 字符, 词表: {config.vocab_size}")
        if is_dialogue:
            train_loader, val_loader = make_dialogue_dataloaders(
                text, tokenizer, config, batch_size)
        else:
            train_loader, val_loader = make_dataloaders(text, tokenizer, config, batch_size)
        train(model, train_loader, val_loader, max_iters, lr,
              eval_interval=args.eval_interval, log_interval=args.log_interval,
              resume_from=ckpt_path if args.resume else None,
              ckpt_path=ckpt_path, data_files=data_files,
              vocab=tokenizer.stoi if hasattr(tokenizer, 'stoi') else None,
              device=args.device)
        torch.save(model.state_dict(), model_path)
        print(f"最终模型已保存到 {model_path}")

    if do_generate:
        if not do_train:
            try:
                model.load_state_dict(torch.load(model_path, map_location=args.device))
            except FileNotFoundError:
                print(f"未找到 {model_path}，使用随机参数生成")

        prompt = args.prompt or default_prompt
        prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(args.device)
        out_ids = model.generate(prompt_ids, args.max_new_tokens, args.temperature, args.top_k, tokenizer.eos_id)
        generated = tokenizer.decode(out_ids[0].tolist())

        print(f"\nPrompt: {prompt}")
        print("─" * 50)
        print(generated)
        print("─" * 50)


if __name__ == "__main__":
    main()
