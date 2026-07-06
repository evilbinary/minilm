.PHONY: data train resume generate chat tokenizer dialogue dialogue-resume chat-dialogue all clean help

# ── 续写模型参数 ──
TRAIN_ARGS ?= --preset 40M --max-iters 1000
RESUME_ARGS ?= --preset 40M --max-iters 20000
CHAT_ARGS ?= --temperature 0.8
MODEL_LANG ?= both

# ── 对话模型参数 ──
DIA_DATA ?= data/dialogue_zh.txt
DIA_ARGS ?= --preset 100M --max-iters 5000 --batch-size 4
DIA_RESUME_ARGS ?= --preset 100M --max-iters 10000 --batch-size 4

help:
	@echo "Mini GPT — Makefile"
	@echo ""
	@echo "── 续写模型 ──"
	@echo "  make data             下载数据集"
	@echo "  make train            训练续写模型"
	@echo "  make resume           续训续写模型"
	@echo "  make generate         生成文本"
	@echo "  make chat             交互式生成"
	@echo ""
	@echo "── 对话模型 ──"
	@echo "  make tokenizer        训练 BPE tokenizer"
	@echo "  make dialogue         训练对话模型"
	@echo "  make dialogue-resume  续训对话模型"
	@echo "  make chat-dialogue    对话式聊天"
	@echo ""
	@echo "── 通用 ──"
	@echo "  make all              续写: 数据->训练->生成"
	@echo "  make clean            删除训练产物"
	@echo ""
	@echo "示例:"
	@echo "  make dialogue DIA_DATA=data/yuki_ruozhiba_1.5k.jsonl"
	@echo "  make dialogue DIA_DATA='data/a.jsonl data/b.jsonl'"
	@echo "  make dialogue DIA_ARGS='--preset 200M --max-iters 10000'"

# ── 续写 ──

data:
	python minigpt.py --download --lang $(MODEL_LANG)

train: data
	python minigpt.py --train --lang $(MODEL_LANG) $(TRAIN_ARGS)

resume:
	python minigpt.py --train --resume --lang $(MODEL_LANG) $(RESUME_ARGS)

generate:
	python minigpt.py --generate --lang $(MODEL_LANG) $(TRAIN_ARGS)

chat:
	python chat.py --lang $(MODEL_LANG) $(CHAT_ARGS)

all: data train generate

# ── 对话 ──

tokenizer:
	python tokenizer.py --files data/tinyshakespeare.txt data/xyj.txt --save checkpoint/tokenizer.json

dialogue: tokenizer
	python minigpt.py --train --mode dialogue --dialogue-data $(DIA_DATA) $(DIA_ARGS)

dialogue-resume:
	python minigpt.py --train --mode dialogue --resume $(DIA_RESUME_ARGS)

chat-dialogue:
	python chat.py --mode dialogue $(CHAT_ARGS)

# ── 清理 ──

clean:
	rm -rf checkpoint
	rm -rf __pycache__
	@echo "已清理训练产物"
