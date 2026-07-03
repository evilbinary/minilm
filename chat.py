"""
Chat — 交互式文本生成。

和训练好的 Mini GPT 对话（字符级续写风格）。

Usage:
    python chat.py                          # 英文（默认）
    python chat.py --lang zh                # 中文
    python chat.py --lang zh --temperature 0.7 --top-k 30
"""

from minigpt import MiniGPT, GPTConfig, CharTokenizer, get_data
import torch


def load_model(lang: str = "en", device: str = "cpu"):
    """加载训练好的模型和 tokenizer"""
    text = get_data(lang)
    tokenizer = CharTokenizer(text)

    config = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=128,
        d_model=192,
        n_layers=6,
        n_heads=6,
        d_ff=768,
    )

    model = MiniGPT(config).to(device)

    # 尝试加载模型权重
    model_path = f"minigpt_{lang}.pt"
    ckpt_path = f"minigpt_{lang}_checkpoint.pt"
    loaded = False
    for path in [model_path, ckpt_path]:
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
            if "model_state_dict" in ckpt:
                model.load_state_dict(ckpt["model_state_dict"])
            else:
                model.load_state_dict(ckpt)
            print(f"  已加载: {path}")
            loaded = True
            break
        except FileNotFoundError:
            continue

    if not loaded:
        print("  ⚠ 未找到训练好的模型，使用随机参数（效果较差）")

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

        # 命令
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

        # 编码 + 生成
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
