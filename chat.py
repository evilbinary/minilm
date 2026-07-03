"""
Chat — 交互式文本生成。

和训练好的 Mini GPT 对话（字符级续写风格）。

Usage:
    python chat.py
    python chat.py --temperature 0.7 --top-k 30
"""

from minigpt import MiniGPT, GPTConfig, CharTokenizer, get_data, CKPT_PATH
import torch
import sys


def load_model(device: str):
    """加载训练好的模型和 tokenizer"""
    text = get_data()
    tokenizer = CharTokenizer(text)

    # 先从 checkpoint 加载配置（如果有），否则用默认
    try:
        ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=ckpt.get("max_seq_len", 128),
            d_model=ckpt.get("d_model", 192),
            n_layers=ckpt.get("n_layers", 6),
            n_heads=ckpt.get("n_heads", 6),
            d_ff=ckpt.get("d_ff", 768),
        )
    except FileNotFoundError:
        # 回退到默认配置
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
    loaded = False
    for path in ["minigpt.pt", CKPT_PATH]:
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
    parser.add_argument("--temperature", type=float, default=0.8, help="采样温度")
    parser.add_argument("--top-k", type=int, default=40, help="Top-K 采样")
    parser.add_argument("--max-new-tokens", type=int, default=150, help="最大生成长度")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    device = args.device

    print("加载 Mini GPT...")
    model, tokenizer = load_model(device)

    print(f"\n{'='*60}")
    print("  Mini GPT Chat (交互式生成)")
    print(f"  温度={args.temperature}  top-k={args.top_k}")
    print(f"  输入 /temp 0.6 调整温度，/topk 20 调整 top-k")
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
            if cmd == "/quit" or cmd == "/exit":
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

        # 去掉 prompt 部分，只显示新生成的内容
        new_text = generated[len(prompt):]

        print(f"GPT > {new_text}")
        print()


if __name__ == "__main__":
    main()
