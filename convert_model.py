"""
模型转换脚本 — 导出 GGUF (llama.cpp) 和 ONNX 格式。

用法:
    python convert_model.py gguf --checkpoint checkpoint/minigpt_sft.pt --output model.gguf
    python convert_model.py onnx --checkpoint checkpoint/minigpt_sft.pt --output model.onnx
"""

import torch
import os
import sys

# ── 加载模型 ──

def load_model(checkpoint_path: str):
    """加载 MiniGPT 模型"""
    data = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = data.get("model_state_dict", data)
    d_model = sd["lm_head.weight"].shape[1]
    vocab = sd["lm_head.weight"].shape[0]
    n_layers = sum(1 for k in sd if k.startswith("blocks.") and k.endswith(".ln1.weight"))
    d_ff = sd["blocks.0.ffn.w1.weight"].shape[0]
    n_heads = 16 if d_model % 16 == 0 else 12
    max_seq_len = sd["pos_embedding.pe.weight"].shape[0]

    from minigpt import MiniGPT, GPTConfig
    config = GPTConfig(vocab_size=vocab, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, d_ff=d_ff, max_seq_len=max_seq_len)
    model = MiniGPT(config)
    model.load_state_dict(sd)
    model.eval()
    print(f"  模型: {d_model}x{n_layers} | {vocab} 词表 | {max_seq_len} 长度")
    return model, sd, config


# ── GGUF 导出 ──

def export_gguf(sd, config, output: str = "model.gguf"):
    """导出 GGUF 格式（用于 llama.cpp）"""
    try:
        import gguf
    except ImportError:
        print("请先安装 gguf: pip install gguf")
        return

    d_model = config.d_model
    n_layers = config.n_layers
    n_heads = config.n_heads

    print(f"\n📦 导出 GGUF: {output}")
    writer = gguf.GGUFWriter(output, "gpt2")

    # ── 超参数 ──
    writer.add_block_count(n_layers)
    writer.add_context_length(config.max_seq_len)
    writer.add_embedding_length(d_model)
    writer.add_feed_forward_length(config.d_ff)
    writer.add_head_count(n_heads)
    writer.add_layer_norm_eps(1e-5)  # gpt2.attention.layer_norm_epsilon
    writer.add_head_count_kv(n_heads)
    writer.add_rope_freq_base(10000.0)
    writer.add_rope_scaling_type(gguf.RopeScalingType.NONE)

    # ── 词嵌入 ──
    writer.add_tensor("token_embd.weight", sd["token_embedding.weight"].numpy())

    # ── 位置编码 ──
    writer.add_tensor("position_embd.weight", sd["pos_embedding.pe.weight"].numpy())

    # ── 各层 ──
    for i in range(n_layers):
        prefix = f"blocks.{i}."
        print(f"  层 {i}/{n_layers}...")

        # LayerNorm 1 (pre-attention)
        writer.add_tensor(f"blk.{i}.attn_norm.weight", sd[f"{prefix}ln1.weight"].numpy())
        writer.add_tensor(f"blk.{i}.attn_norm.bias",   sd[f"{prefix}ln1.bias"].numpy())

        # QKV → 拆分为 Q, K, V
        qkv_w = sd[f"{prefix}attn.qkv.weight"].numpy()
        qkv_b = sd[f"{prefix}attn.qkv.bias"].numpy()
        writer.add_tensor(f"blk.{i}.attn_q.weight", qkv_w[:d_model])
        writer.add_tensor(f"blk.{i}.attn_q.bias",   qkv_b[:d_model])
        writer.add_tensor(f"blk.{i}.attn_k.weight", qkv_w[d_model:2*d_model])
        writer.add_tensor(f"blk.{i}.attn_k.bias",   qkv_b[d_model:2*d_model])
        writer.add_tensor(f"blk.{i}.attn_v.weight", qkv_w[2*d_model:])
        writer.add_tensor(f"blk.{i}.attn_v.bias",   qkv_b[2*d_model:])

        # Attention 输出
        writer.add_tensor(f"blk.{i}.attn_output.weight", sd[f"{prefix}attn.out.weight"].numpy())
        writer.add_tensor(f"blk.{i}.attn_output.bias",   sd[f"{prefix}attn.out.bias"].numpy())

        # LayerNorm 2 (pre-FFN)
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", sd[f"{prefix}ln2.weight"].numpy())
        writer.add_tensor(f"blk.{i}.ffn_norm.bias",   sd[f"{prefix}ln2.bias"].numpy())

        # FFN
        writer.add_tensor(f"blk.{i}.ffn_up.weight",   sd[f"{prefix}ffn.w1.weight"].numpy())
        writer.add_tensor(f"blk.{i}.ffn_up.bias",     sd[f"{prefix}ffn.w1.bias"].numpy())
        writer.add_tensor(f"blk.{i}.ffn_down.weight", sd[f"{prefix}ffn.w2.weight"].numpy())
        writer.add_tensor(f"blk.{i}.ffn_down.bias",   sd[f"{prefix}ffn.w2.bias"].numpy())

    # ── 最终 LayerNorm ──
    writer.add_tensor("output_norm.weight", sd["ln_final.weight"].numpy())
    writer.add_tensor("output_norm.bias", sd["ln_final.bias"].numpy())

    # ── LM Head ──
    writer.add_tensor("output.weight", sd["lm_head.weight"].numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    size = os.path.getsize(output) / 1024 / 1024
    print(f"\n✅ GGUF 已导出: {output} ({size:.0f}MB)")


# ── ONNX 导出 ──

def export_onnx(model, config, output: str = "model.onnx"):
    """导出 ONNX 格式"""
    print(f"\n📦 导出 ONNX: {output}")

    # 构造 dummy 输入
    batch, seq = 1, config.max_seq_len
    dummy = torch.randint(0, config.vocab_size, (batch, seq))

    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits":    {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        verbose=False,
    )
    size = os.path.getsize(output) / 1024 / 1024
    print(f"✅ ONNX 已导出: {output} ({size:.0f}MB)")
    print("运行: onnxruntime 推理示例:")
    print("  import onnxruntime as ort")
    print("  sess = ort.InferenceSession('model.onnx')")
    print("  out = sess.run(None, {'input_ids': tokens})")


# ── CLI ──

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="模型格式转换")
    parser.add_argument("format", choices=["gguf", "onnx"], help="输出格式")
    parser.add_argument("--checkpoint", default="checkpoint/minigpt_sft.pt", help="模型路径")
    parser.add_argument("--output", default=None, help="输出路径")
    args = parser.parse_args()

    model, sd, config = load_model(args.checkpoint)
    output = args.output or f"model.{args.format}"

    if args.format == "gguf":
        export_gguf(sd, config, output)
    elif args.format == "onnx":
        export_onnx(model, config, output)
