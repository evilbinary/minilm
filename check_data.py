"""
数据质量检查脚本 — 检查训练数据里是否混入了 JSON 结构。

用法:
    python check_data.py                    # 检查全部
    python check_data.py --pretrain         # 仅检查预训练数据
    python check_data.py --sft              # 仅检查 SFT 数据
"""

import re
import os


# 要检测的 JSON 特征
JSON_PATTERNS = [
    '"role":',
    '"content":',
    '"conversations":',
    '"messages":',
    '"text":',
    '"gt":',
    '"tools":',
    '"reasoning_content"',
    '"Inner Thoughts"',
    '"Commands"',
]


def check_file(path: str, name: str) -> int:
    """检查单个文件中含 JSON 特征的行数"""
    if not os.path.exists(path):
        print(f"  ❌ {name}: 文件不存在")
        return 0

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    bad_lines = []
    for i, line in enumerate(lines):
        for pat in JSON_PATTERNS:
            if pat in line:
                bad_lines.append((i, line.strip()[:120]))
                break

    total = len(lines)
    bad = len(bad_lines)
    pct = bad * 100 / total if total > 0 else 0

    if bad == 0:
        print(f"  ✅ {name}: {total} 行, 无 JSON 残留")
    else:
        print(f"  ⚠️  {name}: {total} 行, 含 JSON {bad} 行 ({pct:.1f}%)")
        for i, line in bad_lines[:3]:
            print(f"    第{i}行: {line}")
        if bad > 3:
            print(f"    ... 还有 {bad-3} 行")

    return bad


def check_pretrain():
    """检查预训练数据（只检查生成后的文件）"""
    print("\n📚 预训练数据检查:")
    return check_file("data/pretrain_text.txt", "pretrain_text.txt")


def check_sft():
    """检查 SFT 数据（只检查生成后的文件）"""
    print("\n🗣️  SFT 数据检查:")
    return check_file("data/sft_train.txt", "sft_train.txt")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="数据质量检查")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--sft", action="store_true")
    args = parser.parse_args()

    bad = 0
    if args.pretrain:
        bad += check_pretrain()
    elif args.sft:
        bad += check_sft()
    else:
        bad += check_pretrain()
        bad += check_sft()

    if bad == 0:
        print("\n✅ 全部数据干净，可以放心训练")
    else:
        print(f"\n⚠️  共 {bad} 行含 JSON，建议清理后重新生成数据")
