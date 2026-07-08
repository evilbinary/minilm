"""
MOSS 数据集提取与清洗脚本。

从 ModelScope 下载的 moss-003-sft-data 格式特殊：
  - chat 字段是 dict {"turn_1": {...}, "turn_2": {...}}
  - 每轮含 Human/MOSS/Inner Thoughts/Commands 等字段
  - 内容含有 <|Human|>: 和 <eoh> 等标签

本脚本将其转换为标准 {"conversations": [{"role":"user","content":"..."}, ...]} 格式。

用法:
    python prepare_moss.py --input moss-003-sft-no-tools.jsonl.zip --output data/moss_sft.jsonl --max-lines 10000
    python prepare_moss.py --input moss-003-sft-no-tools.jsonl.zip --output data/moss_sft.jsonl  # 全部转换
"""

import zipfile
import json
import os
import re


def clean(text: str) -> str:
    """去掉 <|Human|>:、<|MOSS|>:、<eoh>、<eom> 等标签"""
    text = re.sub(r'<\|[A-Za-z]+\|>:\s*', '', text)
    for tag in ['<eoh>', '<eoa>', '<eom>', '<eos>']:
        text = text.replace(tag, '')
    return text.strip()


def find_moss_zip() -> str:
    """自动查找 MOSS zip 文件"""
    candidates = [
        "~/.cache/modelscope/datasets/openmoss--moss-003-sft-data/snapshots/*/moss-003-sft-no-tools.jsonl.zip",
        "/mnt/workspace/.cache/modelscope/datasets/openmoss--moss-003-sft-data/snapshots/*/moss-003-sft-no-tools.jsonl.zip",
    ]
    import glob
    for pattern in candidates:
        matches = glob.glob(os.path.expanduser(pattern))
        if matches:
            return matches[0]
    return ""


def convert_moss(input_path: str = None, output_path: str = "data/moss_sft.jsonl", max_lines: int = None):
    """
    将 MOSS 原始 zip 文件转换为标准对话格式。

    参数:
        input_path: moss-003-sft-no-tools.jsonl.zip 路径（None 自动查找）
        output_path: 输出 JSONL 文件路径
        max_lines: 最多提取多少条（None=全部）
    """
    if not input_path:
        input_path = find_moss_zip()
    if not input_path or not os.path.exists(input_path):
        print(f"❌ 未找到 MOSS zip 文件")
        return 0

    print(f"📦 MOSS zip: {input_path}")
    count = 0
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with zipfile.ZipFile(input_path) as zf:
        name = zf.namelist()[0]  # zip 内的文件名
        with open(output_path, 'w', encoding='utf-8') as out:
            for line in zf.open(name):
                if max_lines and count >= max_lines:
                    break
                try:
                    d = json.loads(line)
                except json.JSONDecodeError:
                    continue

                chat = d.get('chat', {})
                conversations = []

                # turn_1, turn_2, ... 按数字排序
                turn_keys = sorted(chat.keys(), key=lambda k: int(k.split('_')[1]))
                for turn_key in turn_keys:
                    turn = chat[turn_key]
                    human = clean(turn.get('Human', ''))
                    moss = clean(turn.get('MOSS', ''))
                    if human:
                        conversations.append({'role': 'user', 'content': human})
                    if moss:
                        conversations.append({'role': 'assistant', 'content': moss})

                if conversations:
                    out.write(json.dumps({'conversations': conversations},
                                         ensure_ascii=False) + '\n')
                    count += 1

    print(f"已提取 {count} 条 → {output_path}")
    return count


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MOSS 数据集清洗")
    parser.add_argument("--input", type=str, default=None,
                        help="MOSS zip 文件路径")
    parser.add_argument("--output", type=str, default="data/moss_sft.jsonl",
                        help="输出文件路径")
    parser.add_argument("--max-lines", type=int, default=10000,
                        help="最多提取条数（默认 10000）")
    args = parser.parse_args()

    convert_moss(args.input, args.output, args.max_lines)
