"""
Chat — 交互式文本生成。

从 checkpoint 自动读取模型配置，无需硬编码架构参数。

Usage:
    python chat.py                     # 英文
    python chat.py --lang zh           # 中文
    python chat.py --lang both         # 中英混合
"""

from minigpt import MiniGPT, CharTokenizer, get_data
from config import GPTConfig, get_config
import torch
import os


def _infer_config(state_dict: dict) -> GPTConfig:
    """从 state_dict 推断模型架构（兼容旧版 checkpoint）"""
    # 通过 lm_head 输入维度反推 d_model
    lm_w = state_dict["lm_head.weight"]            # (vocab, d_model)
    vocab, d_model = lm_w.shape

    # 通过 qkv 权重反推 n_heads
    qkv_w = state_dict["blocks.0.attn.qkv.weight"]  # (3*d_model, d_model)
    assert qkv_w.shape[1] == d_model
    head_dim = qkv_w.shape[0] // 3 // 8              # 假设 8 头，算 head_dim
    # 找最接近的整数头数
    for nh in [4, 6, 8, 12, 16]:
        if (3 * d_model) % (3 * nh) == 0:
            break

    # 通过 FFN 权重反推 d_ff
    ffn_w = state_dict["blocks.0.ffn.w1.weight"]     # (d_ff, d_model)
    d_ff = ffn_w.shape[0]

    # 层数
    n_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".ln1.weight"))

    return GPTConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=8 if d_model % 8 == 0 else 4,
        d_ff=d_ff,
    )


def load_model(lang: str = "en", device: str = "cpu"):
    """加载训练好的模型（配置从 checkpoint 读取）"""
    text = get_data(lang)
    tokenizer = CharTokenizer(text)

    ckpt_path = f"checkpoint/minigpt_{lang}_checkpoint.pt"
    model_path = f"checkpoint/minigpt_{lang}.pt"
    vocab_size = tokenizer.vocab_size

    for path in [model_path, ckpt_path]:
        if not os.path.exists(path):
            continue
        data = torch.load(path, map_location=device, weights_only=False)
        state_dict = data.get("model_state_dict", data)

        config = data.get("config", None)
        if config is None:
            config = _infer_config(state_dict)
            print(f"  ⚠ 从权重反推配置: d={config.d_model} l={config.n_layers} f={config.d_ff}")

        model = MiniGPT(config).to(device)
        model.load_state_dict(state_dict)
        print(f"  已加载: {path}")
        return model, tokenizer

    model = MiniGPT(get_config(vocab_size=vocab_size)).to(device)
    print("  ⚠ 未找到训练好的模型，使用随机参数")
    return model, tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mini GPT 交互式聊天")
    parser.add_argument("--lang", choices=["en", "zh", "both"], default="en",
                        help="语言: en=英文, zh=中文, both=中英混合")
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K 采样")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="最大生成长度")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    lang = args.lang
    device = args.device
    lang_names = {"en": "English", "zh": "中文", "both": "中英双语"}
    lang_name = lang_names.get(lang, "English")

    print(f"加载 Mini GPT ({lang_name})...")
    model, tokenizer = load_model(lang, device)

    print(f"\n{'='*60}")
    print(f"  Mini GPT Chat ({lang_name} · 交互式生成)")
    print(f"  温度={args.temperature}  top-k={args.top_k}")
    print(f"  输入 /temp 0.6 调温度，/topk 20 调 top-k")
    print(f"  输入 /quit 或 Ctrl+C 退出")
    print(f"{'='*60}\n")

    temp = args.temperature
    topk = args.top_k

    while True:
        try:
            prompt = input("你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见!")
            break

        if not prompt:
            continue

        if prompt.startswith("/"):
            parts = prompt.split()
            cmd = parts[0]
            if cmd in ("/quit", "/exit"):
                break
            elif cmd == "/temp" and len(parts) > 1:
                temp = float(parts[1])
                print(f"  → 温度设为 {temp}")
                continue
            elif cmd == "/topk" and len(parts) > 1:
                topk = int(parts[1])
                print(f"  → top-k 设为 {topk}")
                continue
            else:
                print(f"  命令: /temp N, /topk N, /quit")
                continue

        prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=topk,
                eos_id=tokenizer.eos_id,
            )

        generated = tokenizer.decode(out_ids[0].tolist())
        new_text = generated[len(prompt):]
        print(f"GPT > {new_text}")
        print()


if __name__ == "__main__":
    main()
