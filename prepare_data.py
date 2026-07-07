"""
数据准备脚本 — 统一处理预训练和 SFT 数据。

用法:
    python prepare_data.py pretrain                    # 准备预训练数据
    python prepare_data.py sft                         # 准备 SFT 数据
    python prepare_data.py --mode pretrain --output data/pretrain_text.txt
    python prepare_data.py --mode sft --files data/moss_sft.jsonl
"""

import json
import os
import re
import glob


# ── 清洗函数 ─────────────────────────────

def clean_content(text: str) -> str:
    """去掉 MOSS 标签、多余空格，保留正文"""
    text = re.sub(r'<\|[A-Za-z]+\|>:\s*', '', text)  # <|Human|>: 等
    for tag in ['<eoh>', '<eoa>', '<eom>', '<eos>']:
        text = text.replace(tag, '')
    return text.strip()


def has_json_pattern(text: str) -> bool:
    """检查文本是否含有 JSON 结构（用于过滤）"""
    patterns = ['"role":', '"content":', '"conversations":', '"messages":',
                '"text":', '"gt":', '"tools":', '"reasoning_content"',
                '"Inner Thoughts"', '"Commands"']
    return any(p in text for p in patterns)


# ── 格式转换 ─────────────────────────────

def format_dialogue(messages: list[dict]) -> str:
    """对话列表 → <|user|>...<|end|><|assistant|>...<|end|> 格式"""
    parts = []
    for msg in messages:
        role = msg.get("role", "")
        tag = "<|assistant|>" if role == "assistant" else "<|user|>"
        content = msg.get("content", "").replace("\n", " ").replace("\r", " ")
        content = clean_content(content)
        if content:
            parts.append(f"{tag}{content}<|end|>")
    return "".join(parts)


# ── 预处理对话格式（统一各种 JSONL 格式）──

def extract_conversations(data: dict) -> list[dict]:
    """从 JSON 行中提取对话列表，支持多种格式"""
    # 标准格式: {"conversations": [{"role":"user","content":"..."}, ...]}
    # MOSS 格式: {"chat": {"turn_1": {"Human":"...", "MOSS":"..."}, ...}}
    # 消息格式: {"messages": [{"role":"user","content":"..."}, ...]}

    conv = data.get("conversations") or data.get("messages") or data.get("chat")
    if conv is None:
        return []

    if isinstance(conv, dict):
        # MOSS 格式: {"turn_1": {...}, "turn_2": {...}}
        result = []
        for key in sorted(conv.keys(), key=lambda k: int(k.split("_")[1])):
            turn = conv[key]
            if turn.get("Human"):
                result.append({"role": "user", "content": clean_content(turn["Human"])})
            if turn.get("MOSS"):
                result.append({"role": "assistant", "content": clean_content(turn["MOSS"])})
        return result

    if isinstance(conv, list):
        return conv

    return []


# ── 文件处理 ─────────────────────────────

def convert_jsonl(jsonl_paths: list[str], output: str = None,
                  max_lines: int = None, skip_json: bool = True) -> str:
    """将 JSONL 文件转换为训练文本"""
    if isinstance(jsonl_paths, str):
        jsonl_paths = [jsonl_paths]

    all_lines = []
    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if max_lines and i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 尝试多种格式
                conv = extract_conversations(data)
                if conv:
                    text = format_dialogue(conv)
                elif "text" in data and data["text"].strip():
                    text = clean_content(data["text"])
                else:
                    continue

                # 过滤 JSON 残留
                if skip_json and has_json_pattern(text):
                    continue
                if text:
                    all_lines.append(text)

    result = "\n".join(all_lines)
    if output:
        os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"已生成 {len(all_lines)} 条 → {output} ({len(result)/1024/1024:.0f}MB)")

    return result


# ── 主入口 ─────────────────────────────

def prepare_pretrain(output: str = None):
    if output is None:
        output = "data/pretrain_text.txt"
    """准备预训练数据：过滤所有 JSON 特征"""
    paths = sorted(glob.glob("data/pretrain/*.jsonl") + glob.glob("data/pretrain/*.txt"))

    print("预训练数据文件:")
    all_lines = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
        # JSONL 文件：提取 text 字段后再过滤
        if p.endswith(".jsonl"):
            texts = []
            for line in raw.split("\n"):
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    t = d.get("text", "")
                    if t and not has_json_pattern(t):
                        texts.append(clean_content(t))
                except json.JSONDecodeError:
                    continue
            all_lines.extend(texts)
            print(f"  {os.path.basename(p)}: {len(texts)} 条文本 (JSONL)")
        else:
            lines = [l for l in raw.split("\n") if not has_json_pattern(l)]
            all_lines.extend(lines)
            print(f"  {os.path.basename(p)}: {len(lines)} 行")

    result = "\n".join(all_lines)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(result)

    print(f"\n预训练数据已生成: {output} ({len(result)/1024/1024:.0f}MB)")
    print(f"含 JSON 残留: {has_json_pattern(result)}")


def prepare_sft(files: list[str] = None, output: str = None, max_lines: int = None):
    """准备 SFT 数据：转换多种格式为统一对话格式"""
    if output is None:
        output = "data/sft_train.txt"
    if files is None:
        files = ["data/sft/sft_t2t_mini.jsonl", "data/sft/moss_sft.jsonl",
                 "data/sft/yuki_ruozhiba_1.5k.jsonl"]
    convert_jsonl(files, output, max_lines=max_lines, skip_json=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数据准备")
    parser.add_argument("mode", choices=["pretrain", "sft"], help="pretrain / sft")
    parser.add_argument("--files", nargs="+", help="SFT 数据文件列表")
    parser.add_argument("--output", type=str, help="输出文件")
    parser.add_argument("--max-lines", type=int, help="最多处理行数")
    args = parser.parse_args()

    if args.mode == "pretrain":
        prepare_pretrain(args.output)
    else:
        prepare_sft(args.files, args.output, args.max_lines)
