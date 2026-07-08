import warnings
warnings.filterwarnings("ignore", message=".*quantize_dynamic.*deprecated.*")
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
    # 从 QKV 权重推断头数: qkv shape = (3*d_model, d_model), d_model = head_dim * n_heads
    qkv_w = state_dict["blocks.0.attn.qkv.weight"]
    head_dim_candidates = [64, 96, 128]
    n_heads = n_heads = 8
    for hd in head_dim_candidates:
        if d_model % hd == 0 and qkv_w.shape[0] % 3 == 0:
            nh = d_model // hd
            if nh * hd == d_model:
                n_heads = nh
                break
    return GPTConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                     n_heads=n_heads, d_ff=d_ff)


def load_model(mode: str = "completion", lang: str = "en", device: str = "cpu",
               ckpt: str = None):
    """加载模型（ckpt 指定路径时覆盖默认）"""
    if ckpt:
        # 直接指定路径
        paths = [ckpt]
        is_bpe = any(tag in ckpt for tag in ["pretrain", "dialogue", "combined", "sft"])
        from tokenizer import load_tokenizer
        tokenizer = load_tokenizer("checkpoint/tokenizer.json") if is_bpe else CharTokenizer(get_data(lang))
    elif mode in ("dialogue", "combined"):
        from tokenizer import load_tokenizer
        tokenizer = load_tokenizer("checkpoint/tokenizer.json")
        ckpt_name = f"minigpt_{mode}"
        paths = [f"checkpoint/{ckpt_name}.pt", f"checkpoint/{ckpt_name}_checkpoint.pt"]
    else:
        text = get_data(lang)
        tokenizer = CharTokenizer(text)
        ckpt_name = f"minigpt_{lang}"
        paths = [f"checkpoint/{ckpt_name}.pt", f"checkpoint/{ckpt_name}_checkpoint.pt"]

    for path in paths:
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
        if device == "cpu":
            print("  ⚡ 应用 INT8 量化加速...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8
            )
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
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="直接指定 checkpoint 路径（覆盖 mode 的默认路径）")
    parser.add_argument("--lang", choices=["en", "zh", "both"], default="en",
                        help="语言（续写模式）")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--threads", type=int, default=4, help="CPU 线程数")
    parser.add_argument("--device", default=None,
                        help="cpu / cuda (默认自动检测，显存不足自动切 cpu)")
    args = parser.parse_args()

    # 自动选择设备（GPU 显存不足时降级到 CPU）
    device = args.device
    if device is None:
        if torch.cuda.is_available():
            try:
                free_mem = torch.cuda.mem_get_info()[0] / 1024**3
                if free_mem > 2:
                    device = "cuda"
                else:
                    device = "cpu"
                    print(f"  ⚠ GPU 显存不足 ({free_mem:.1f}G 空闲)，切到 CPU")
            except:
                device = "cpu"
                print("  ⚠ GPU 不可用，切到 CPU")
        else:
            device = "cpu"
    mode = args.mode

    mode_name = {"completion": "续写", "dialogue": "对话", "combined": "混合"}
    print(f"加载 Mini GPT ({mode_name.get(mode, mode)})...")
    model, tokenizer = load_model(mode, args.lang, device, ckpt=args.checkpoint)

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

        # 对话/混合模式：自动包装 <|user|> 标签
        if args.checkpoint and "pretrain" in args.checkpoint:
            input_text = prompt  # 预训练模型：续写，不加标签
        elif mode in ("dialogue", "combined"):
            input_text = f"{user_tag}{prompt}{end_tag}\n<|assistant|>"
        else:
            input_text = prompt

        prompt_ids = torch.tensor(tokenizer.encode(input_text), dtype=torch.long).unsqueeze(0).to(device)

        try:
            with torch.no_grad():
                out_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temp,
                    top_k=topk,
                    eos_id=eos_id,
                )
        except torch.OutOfMemoryError:
            print("  ⚠ GPU 显存不足，自动切换到 CPU")
            device = "cpu"
            model.to(device)
            prompt_ids = prompt_ids.to(device)
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
        if args.checkpoint and "pretrain" in args.checkpoint:
            reply = generated[len(input_text):]  # 预训练：不加标签，直接续写结果
        elif mode in ("dialogue", "combined"):
            reply = generated[len(input_text):]
            for tag in [end_tag, "<|user|>", "<|assistant|>"]:
                reply = reply.split(tag)[0]
        else:
            reply = generated[len(input_text):]

        print(f"GPT > {reply.strip()}")
        print()


if __name__ == "__main__":
    main()
