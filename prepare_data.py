"""
对话数据准备脚本 — 下载并格式化对话数据。

支持的数据集:
    - simple_zh: 简单中文问答（内置，无需下载）
    - openassistant: OpenAssistant 对话数据集

用法:
    python prepare_data.py --dataset simple_zh
    python prepare_data.py --dataset simple_zh --format dialogue
"""

import os
import json  # noqa
from tokenizer import load_tokenizer


SIMPLE_ZH_DIALOGUES = [
    # ── 日常对话 ──
    [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
    ],
    [
        {"role": "user", "content": "今天天气怎么样？"},
        {"role": "assistant", "content": "抱歉，我无法查询实时天气，建议查看天气预报 App。"},
    ],
    [
        {"role": "user", "content": "你是谁？"},
        {"role": "assistant", "content": "我是 Mini GPT，一个基于 Transformer 的语言模型。"},
    ],
    [
        {"role": "user", "content": "你会做什么？"},
        {"role": "assistant", "content": "我可以帮你写文章、回答问题、翻译文字、编写代码等。"},
    ],
    # ── 知识问答 ──
    [
        {"role": "user", "content": "什么是机器学习？"},
        {"role": "assistant", "content": "机器学习是人工智能的一个分支，让计算机从数据中学习规律和模式，而不需要显式编程。常见方法有监督学习、无监督学习和强化学习。"},
    ],
    [
        {"role": "user", "content": "Python 是什么？"},
        {"role": "assistant", "content": "Python 是一种高级编程语言，以其简洁易读的语法著称。广泛应用于数据分析、Web 开发、人工智能等领域。"},
    ],
    [
        {"role": "user", "content": "什么是 Transformer？"},
        {"role": "assistant", "content": "Transformer 是一种基于自注意力机制的神经网络架构，由 Google 在 2017 年提出。它是 GPT、BERT 等大语言模型的基础，核心创新是自注意力机制和位置编码。"},
    ],
    # ── 英文 ──
    [
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "Hello! I'm doing well, thank you for asking. How can I help you today?"},
    ],
    [
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris, known for its art, culture, and landmarks such as the Eiffel Tower."},
    ],
    [
        {"role": "user", "content": "Tell me about deep learning."},
        {"role": "assistant", "content": "Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence 'deep') to learn hierarchical representations of data. It excels at tasks like image recognition, natural language processing, and speech recognition."},
    ],
    # ── 编程 ──
    [
        {"role": "user", "content": "用 Python 写一个快速排序"},
        {"role": "assistant", "content": "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n```"},
    ],
    [
        {"role": "user", "content": "解释一下 Git 的用法"},
        {"role": "assistant", "content": "Git 是分布式版本控制系统。基本用法：\n1. git init — 初始化仓库\n2. git add — 暂存文件\n3. git commit — 提交更改\n4. git push — 推送到远程\n5. git pull — 拉取最新代码"},
    ],
    # ── 多轮对话 ──
    [
        {"role": "user", "content": "给我讲个笑话"},
        {"role": "assistant", "content": "程序员问妻子：我去超市买点东西，马上回来。\n妻子说：顺便买一盒牛奶。\n程序员：好的。\n二十年后...\n妻子：你买的牛奶呢？\n程序员：啊！我忘了买牛奶！"},
    ],
    [
        {"role": "user", "content": "再讲一个"},
        {"role": "assistant", "content": "为什么程序员总是在万圣节和圣诞节搞混？\n因为 Oct 31 == Dec 25！"},
    ],
]


def format_dialogue(messages: list[dict]) -> str:
    """将对话列表格式化为训练文本"""
    parts = []
    for msg in messages:
        role = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
        parts.append(f"{role}{msg['content']}<|end|>")
    return "".join(parts)


def generate_simple_zh(output: str = "data/dialogue_zh.txt", repeat: int = 50):
    """生成内置的中文对话数据（repeat 次重复以增加数据量）"""
    dialogues = SIMPLE_ZH_DIALOGUES
    lines = []
    for _ in range(repeat):
        for d in dialogues:
            lines.append(format_dialogue(d))
    text = "\n".join(lines)
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"已生成 {len(dialogues)} 条对话 → {output} ({len(text)} 字符)")
    return output


def convert_jsonl(jsonl_path: str, output: str = None):
    """将 JSONL 对话数据转换为训练文本格式"""
    import json
    dialogues = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                conv = data.get("conversations", data.get("messages", []))
                if conv:
                    dialogues.append(format_dialogue(conv))
            except json.JSONDecodeError:
                continue

    text = "\n".join(dialogues)
    if output:
        with open(output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已转换 {len(dialogues)} 条对话 → {output} ({len(text)} 字符)")
    else:
        print(f"共 {len(dialogues)} 条对话, {len(text)} 字符")
    return text


def main():
    import argparse
    parser = argparse.ArgumentParser(description="对话数据准备")
    parser.add_argument("--dataset", default="simple_zh",
                        help="数据集: simple_zh")
    parser.add_argument("--jsonl", type=str, default=None,
                        help="JSONL 对话数据文件路径")
    parser.add_argument("--output", default="data/dialogue_zh.txt",
                        help="输出文件路径")
    parser.add_argument("--repeat", type=int, default=1,
                        help="重复次数（simple_zh 用）")
    args = parser.parse_args()

    if args.jsonl:
        convert_jsonl(args.jsonl, args.output)
    elif args.dataset == "simple_zh":
        generate_simple_zh(args.output, repeat=args.repeat)
    else:
        print(f"未知数据集: {args.dataset}")

    # 验证 tokenizer 编码
    try:
        tok = load_tokenizer()
        with open(args.output) as f:
            sample = f.read()[:200]
        ids = tok.encode(sample)
        decoded = tok.decode(ids)
        print(f"Tokenizer 验证: {len(sample)} 字符 → {len(ids)} tokens")
    except FileNotFoundError:
        print("Tokenizer 未训练，请先运行 python tokenizer.py")


if __name__ == "__main__":
    main()
