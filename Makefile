.PHONY: help download prepare tokenizer pretrain sft pretrain-resume sft-resume chat-pretrain chat-sft check clean

# ── 通用 ──
CHAT_ARGS ?= --temperature 0.8
MODEL_LANG ?= both

BATCH_SIZE ?= 4

# ── 预训练参数 ──
PRETRAIN_ARGS ?= --preset 200M --max-iters 50000 --batch-size $(BATCH_SIZE)
PRETRAIN_RESUME_ARGS ?= --preset 200M --max-iters 100000 --batch-size $(BATCH_SIZE)

# ── SFT 参数 ──
SFT_DATA ?= data/sft/sft_t2t_mini.jsonl data/sft/moss_sft.jsonl data/sft/yuki_ruozhiba_1.5k.jsonl
SFT_ARGS ?= --preset 200M --batch-size $(BATCH_SIZE) --max-iters 50000 --lr 1e-4

download:
	python download_data.py

prepare: download
	python prepare_data.py pretrain
	python prepare_data.py sft

tokenizer: prepare
	python tokenizer.py --files data/pretrain/tinyshakespeare.txt data/pretrain/xyj.txt data/pretrain/hlm.txt data/sft/yuki_ruozhiba_1.5k.jsonl --save checkpoint/tokenizer.json

check:
	python3 check_data.py

help:
	@echo "Mini GPT — Makefile"
	@echo ""
	@echo "  make download       下载原始数据"
	@echo "  make prepare        下载 + 生成训练数据"
	@echo "  make tokenizer      训练 BPE (自动先下载+准备)"
	@echo "  make pretrain       预训练 (自动全流程)"
	@echo "  make sft            SFT 微调 (自动全流程)"
	@echo "  make chat-pretrain  预训练模型聊天"
	@echo "  make chat-sft       SFT 模型聊天"
	@echo "  make check          数据质量检查"
	@echo "  make clean          清理"

# ── 预训练 ──

pretrain: tokenizer
	python minigpt.py --train --mode pretrain $(PRETRAIN_ARGS)

pretrain-resume:
	python minigpt.py --train --mode pretrain --resume $(PRETRAIN_RESUME_ARGS)

chat-pretrain:
	python chat.py --checkpoint checkpoint/minigpt_pretrain.pt --temperature 0.8

# ── SFT ──

sft: tokenizer
	python minigpt.py --train --mode sft --resume-from checkpoint/minigpt_pretrain.pt \
	  --dialogue-data $(SFT_DATA) $(SFT_ARGS)

sft-resume:
	python minigpt.py --train --mode sft --resume --dialogue-data $(SFT_DATA) $(SFT_ARGS)

chat-sft:
	python chat.py --mode combined --checkpoint checkpoint/minigpt_sft.pt $(CHAT_ARGS)

# ── 清理 ──

clean:
	rm -rf checkpoint
	rm -rf __pycache__
	rm -f data/sft_train.txt data/pretrain_text.txt
	@echo "已清理"
