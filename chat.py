"""
Chat — 交互式文本生成。

支持续写和对话两种模式。

续写模式:
    python chat.py                          # 英文续写
    python chat.py --lang zh                # 中文续写

对话模式:
    python chat.py --mode dialogue          # 对话
"""

from minigpt import MiniGPT, CharTokenizer, get_data
from config import GPTConfig, get_config
import torch
import os


def _infer_config(state_dict: dict) -> GPTConfig:
    """从 state_dict 推断模型架构"""
    lm_w = state_dict["lm_head.weight"]
    vocab, d_model = lm_w.shape
    ffn_w = state_dict["blocks.0.ffn.w1.weight"]
    d_ff = ffn_w.shape[0]
    n_layers = sum(1 for k in state_dict if k.startswith("blocks.") and k.endswith(".ln1.weight"))
    return GPTConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                     n_heads=8 if d_model % 8 == 0 else 4, d_ff=d_ff)


def load_model(mode: str = "completion", lang: str = "en", device: str = "cpu"):
    """加载模型"""
    if mode in ("dialogue", "combined"):
        from tokenizer import load_tokenizer
        tokenizer = load_tokenizer("checkpoint/tokenizer.json")
        ckpt_name = f"minigpt_{mode}"
    else:
        text = get_data(lang)
        tokenizer = CharTokenizer(text)
        ckpt_name = f"minigpt_{lang}"

    ckpt_path = f"checkpoint/{ckpt_name}_checkpoint.pt"
    model_path = f"checkpoint/{ckpt_name}.pt"

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

    # 未找到模型
    vs = tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else len(tokenizer.stoi)
    model = MiniGPT(get_config(vocab_size=vs)).to(device)
    print("  ⚠ 未找到模型，使用随机参数")
    return model, tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mini GPT Chat")
    parser.add_argument("--mode", choices=["completion", "dialogue", "combined"],
                        default="completion", help="completion=续写 dialogue=对话 combined=混合")
    parser.add_argument("--lang", choices=["en", "zh", "both"], default="en",
                        help="语言（续写模式）")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()

    device = args.device
    mode = args.mode

    mode_name = {"completion": "续写", "dialogue": "对话", "combined": "混合"}
    print(f"加载 Mini GPT ({mode_name.get(mode, mode)})...")
    model, tokenizer = load_model(mode, args.lang, device)

    eos_id = tokenizer.eos_id
    user_tag = "<|user|>"
    end_tag = "<|end|>"

    print(f"\n{'='*60}")
    mode_title = {"completion": "续写模式", "dialogue": "对话模式", "combined": "混合模式（对话+续写）"}
    print(f"  Mini GPT ({mode_title.get(mode, mode)})")
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

        # 对话模式：自动包装 <|user|> 标签
        if mode == "dialogue":
            input_text = f"{user_tag}{prompt}{end_tag}\n<|assistant|>"
        else:
            input_text = prompt

        prompt_ids = torch.tensor(tokenizer.encode(input_text), dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                prompt_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=temp,
                top_k=topk,
                eos_id=eos_id,
            )

        generated = tokenizer.decode(out_ids[0].tolist())

        # 提取新生成的内容
        if mode == "dialogue":
            # 去掉输入部分，只保留 assistant 回复
            reply = generated[len(input_text):]
            # 去掉尾部的 <|end|> 等特殊 token
            for tag in [end_tag, "<|user|>", "<|assistant|>"]:
                reply = reply.split(tag)[0]
        else:
            reply = generated[len(input_text):]

        print(f"GPT > {reply.strip()}")
        print()


if __name__ == "__main__":
    main()
