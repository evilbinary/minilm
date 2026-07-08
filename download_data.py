"""
数据下载脚本 — 一键下载全部默认数据集。

用法:
    python download_data.py                    # 下载全部
    python download_data.py --pretrain          # 仅预训练数据
    python download_data.py --sft              # 仅 SFT 数据
"""

import os
import sys
import urllib.request
import zipfile
import json

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


class ProgressBar:
    """下载进度条"""
    def __init__(self, desc_text):
        self.desc = desc_text
    def __call__(self, block, size, total):
        if total > 0:
            pct = min(100, block * size * 100 / total)
            mb = block * size / 1024 / 1024
            total_mb = total / 1024 / 1024
            bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
            sys.stdout.write(f"\r  ⏳ {self.desc} [{bar}] {pct:.0f}% ({mb:.0f}MB/{total_mb:.0f}MB)")
            sys.stdout.flush()


def download(url: str, path: str, desc: str = ""):
    """下载文件，已有则跳过"""
    if os.path.exists(path):
        print(f"  ✅ {desc}: 已存在 ({os.path.getsize(path)/1024/1024:.0f}MB)")
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    urllib.request.urlretrieve(url, path, ProgressBar(desc))
    print(f"\n  ✅ 已保存 ({os.path.getsize(path)/1024/1024:.0f}MB)")


# ── 预训练数据 ──

PRETRAIN_FILES = {
    "data/pretrain/tinyshakespeare.txt": {
        "url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
        "desc": "Tiny Shakespeare",
    },
    "data/pretrain/xyj.txt": {
        "url": "https://github.com/Honlan/fullstack-data-engineer/raw/master/xyj.txt",
        "desc": "西游记",
    },
    "data/pretrain/hlm.txt": {
        "url": "https://raw.githubusercontent.com/llmtest/xiyouji/refs/heads/main/hlm.txt",
        "desc": "红楼梦",
    },
    "data/pretrain/pretrain_t2t_mini.jsonl": {
        "url": "https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/pretrain_t2t_mini.jsonl",
        "desc": "预训练问答数据",
    },
}


def download_pretrain():
    print("\n📚 预训练数据:")
    for path, info in PRETRAIN_FILES.items():
        download(info["url"], os.path.join(DATA_DIR, path), info["desc"])
    print("  注: pretrain_t2t_mini 较大，下载可能较慢")


# ── SFT 数据 ──

# MOSS 数据通过 ModelScope 下载
MOSS_ZIP = os.path.expanduser(
    "~/.cache/modelscope/datasets/openmoss--moss-003-sft-data/"
    "snapshots/master/moss-003-sft-no-tools.jsonl.zip"
)


def download_sft():
    print("\n🗣️  SFT 数据:")
    # MOSS 数据
    if os.path.exists(MOSS_ZIP):
        print(f"  ✅ MOSS 数据集: 已存在 ({os.path.getsize(MOSS_ZIP)/1024/1024:.0f}MB)")
    else:
        print("  ⏳ MOSS 未下载，执行:")
        print("     modelscope download --dataset openmoss/moss-003-sft-data")

    SFT_URL = "https://www.modelscope.cn/datasets/gongjy/minimind_dataset/resolve/master/"
    download(f"{SFT_URL}sft_t2t_mini.jsonl", DATA_DIR + "/data/sft/sft_t2t_mini.jsonl", "SFT 问答数据")
    # yuki 在代码仓库中，检查是否存在
    yuki = DATA_DIR + "/data/sft/yuki_ruozhiba_1.5k.jsonl"
    if os.path.exists(yuki):
        print(f"  ✅ yuki_ruozhiba_1.5k.jsonl: 已存在")



# ── 全部下载 ──

def download_all():
    print("=" * 50)
    print("Mini GPT 数据下载")
    print("=" * 50)
    os.makedirs(f"{DATA_DIR}/pretrain", exist_ok=True)
    os.makedirs(f"{DATA_DIR}/sft", exist_ok=True)
    download_pretrain()
    download_sft()
    expected = [
        "data/pretrain/tinyshakespeare.txt",
        "data/pretrain/xyj.txt",
        "data/pretrain/hlm.txt",
        "data/pretrain/pretrain_t2t_mini.jsonl",
        "data/sft/sft_t2t_mini.jsonl",
        "data/sft/yuki_ruozhiba_1.5k.jsonl",
    ]
    missing = [p for p in expected if not os.path.exists(os.path.join(DATA_DIR, p))]
    if missing:
        print(f"\n❌ 缺失 {len(missing)} 个文件，请手动下载:")
        for p in missing:
            print(f"  {p}")
    else:
        print("\n✅ 全部文件已就绪！")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="下载数据集")
    parser.add_argument("--pretrain", action="store_true", help="仅下载预训练数据")
    parser.add_argument("--sft", action="store_true", help="仅下载 SFT 数据")
    args = parser.parse_args()

    if args.pretrain:
        download_pretrain()
    elif args.sft:
        download_sft()
    else:
        download_all()
