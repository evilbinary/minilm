"""
BPE Tokenizer — 支持中英文、对话格式。

用法:
    from tokenizer import BPETokenizer
    tok = BPETokenizer()
    tok.train(["data/xyj.txt", "data/hlm.txt", "data/tinyshakespeare.txt"])
    tok.save("checkpoint/tokenizer.json")
    ids = tok.encode("你好 <|assistant|>")
    text = tok.decode(ids)
"""

import os
from tokenizers import Tokenizer as HFTokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder
from tokenizers.normalizers import NFC


# 特殊 token（对话标记）
SPECIAL_TOKENS = {
    "<|pad|>": 0,
    "<|unk|>": 1,
    "<|end|>": 2,         # 句子结束 / EOS
    "<|user|>": 3,        # 用户消息起始
    "<|assistant|>": 4,   # 助手消息起始
}


class BPETokenizer:
    """BPE Tokenizer（字节级，通用中英文）"""

    def __init__(self, vocab_size: int = 16000):
        self.vocab_size = vocab_size
        # 用 ByteLevel BPE 模型（天然支持中英文）
        self.tokenizer = HFTokenizer(BPE(unk_token="<|unk|>"))
        self.tokenizer.normalizer = NFC()
        self.tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = ByteLevelDecoder()
        self._special_tokens = SPECIAL_TOKENS

    @property
    def pad_id(self) -> int: return self._special_tokens["<|pad|>"]
    @property
    def unk_id(self) -> int: return self._special_tokens["<|unk|>"]
    @property
    def eos_id(self) -> int: return self._special_tokens["<|end|>"]
    @property
    def user_id(self) -> int: return self._special_tokens["<|user|>"]
    @property
    def assistant_id(self) -> int: return self._special_tokens["<|assistant|>"]
    @property
    def vocab_size(self) -> int: return self._vocab_size
    @vocab_size.setter
    def vocab_size(self, v): self._vocab_size = v

    def train(self, files: list[str]):
        """在文本文件上训练 BPE 词表"""
        added = 0
        tokens = list(self._special_tokens.keys())
        trainer = BpeTrainer(
            vocab_size=self._vocab_size,
            special_tokens=tokens,
            show_progress=True,
            initial_alphabet=ByteLevel.alphabet(),
        )
        self.tokenizer.train(files, trainer)
        print(f"BPE 词表训练完成: {self.tokenizer.get_vocab_size()} tokens")

    def encode(self, text: str) -> list[int]:
        """文本 → token ids"""
        return self.tokenizer.encode(text).ids

    def decode(self, ids: list[int]) -> str:
        """token ids → 文本"""
        return self.tokenizer.decode(ids)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.tokenizer.save(path)
        print(f"Tokenizer 已保存到 {path}")

    def load(self, path: str):
        self.tokenizer = HFTokenizer.from_file(path)
        print(f"Tokenizer 已加载: {self.tokenizer.get_vocab_size()} tokens")

    @staticmethod
    def format_dialogue(messages: list[dict]) -> str:
        """
        将对话列表格式化为训练文本。

        messages = [
            {"role": "user", "content": "你好"},
            {"role": "assistant", "content": "你好！有什么可以帮你？"},
        ]
        → "<|user|>你好<|end|><|assistant|>你好！有什么可以帮你？<|end|>"
        """
        parts = []
        for msg in messages:
            role = "<|user|>" if msg["role"] == "user" else "<|assistant|>"
            parts.append(f"{role}{msg['content']}<|end|>")
        return "\n".join(parts)


# ── 快捷函数 ─────────────────────────────

def train_tokenizer(files: list[str], vocab_size: int = 16000,
                    save_path: str = "checkpoint/tokenizer.json") -> BPETokenizer:
    """训练并保存 BPE tokenizer"""
    tok = BPETokenizer(vocab_size)
    tok.train(files)
    tok.save(save_path)
    return tok


def load_tokenizer(path: str = "checkpoint/tokenizer.json") -> BPETokenizer:
    """加载已训练的 BPE tokenizer"""
    tok = BPETokenizer()
    tok.load(path)
    return tok


if __name__ == "__main__":
    # 命令行训练 tokenizer
    import argparse
    import json, tempfile
    parser = argparse.ArgumentParser()
    parser.add_argument("--files", nargs="+", default=[
        "data/tinyshakespeare.txt", "data/xyj.txt", "data/hlm.txt"
    ])
    parser.add_argument("--vocab-size", type=int, default=16000)
    parser.add_argument("--save", default="checkpoint/tokenizer.json")
    args = parser.parse_args()

    # JSONL 文件 → 临时纯文本
    txt_files = []
    for f in args.files:
        if f.endswith(".jsonl"):
            tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8")
            with open(f, encoding="utf-8") as jf:
                for line in jf:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        conv = data.get("conversations", data.get("messages", []))
                        for msg in conv:
                            tmp.write(msg["content"] + "\n")
                    except json.JSONDecodeError:
                        continue
            tmp.close()
            txt_files.append(tmp.name)
            print(f"  展开 JSONL: {f} → {tmp.name}")
        else:
            txt_files.append(f)

    tok = train_tokenizer(txt_files, args.vocab_size, args.save)
    print(f"测试: {tok.decode(tok.encode('你好 <|assistant|>Hello world!'))}")
