.PHONY: help tokenizer pretrain sft chat-pretrain chat-sft clean

# ── 通用 ──
CHAT_ARGS ?= --temperature 0.8
MODEL_LANG ?= both

# ── 预训练参数 ──
PRETRAIN_ARGS ?= --preset 200M --max-iters 50000 --batch-size 4
PRETRAIN_RESUME_ARGS ?= --preset 200M --max-iters 100000 --batch-size 4

# ── SFT 参数 ──
SFT_DATA ?= data/sft/sft_t2t_mini.jsonl data/sft/moss_sft.jsonl data/sft/yuki_ruozhiba_1.5k.jsonl
SFT_ARGS ?= --preset 200M --batch-size 4 --max-iters 50000 --lr 1e-4

help:
	@echo "Mini GPT — 两阶段训练"
	@echo ""
	@echo "── 第一阶段：预训练 ──"
	@echo "  make tokenizer        训练 BPE tokenizer"
	@echo "  make pretrain         在文本数据上预训练"
	@echo "  make pretrain-resume  续训预训练"
	@echo "  make chat-pretrain    预训练模型聊天"
	@echo ""
	@echo "── 第二阶段：SFT 微调 ──"
	@echo "  make sft              从预训练模型微调对话"
	@echo "  make sft-resume       续训微调"
	@echo "  make chat-sft         SFT 模型聊天"
	@echo ""
	@echo "── 通用 ──"
	@echo "  make clean            删除训练产物"
	@echo ""
	@echo "示例:"
	@echo "  make pretrain PRETRAIN_ARGS='--max-iters 100000'"
	@echo "  make sft     SFT_DATA=data/yuki_ruozhiba_1.5k.jsonl"

# ── Tokenizer ──

tokenizer:
	python tokenizer.py --files data/pretrain/tinyshakespeare.txt data/pretrain/xyj.txt data/pretrain/hlm.txt data/sft/yuki_ruozhiba_1.5k.jsonl --save checkpoint/tokenizer.json

# ── 预训练 ──

pretrain: tokenizer
	python prepare_data.py pretrain
	python minigpt.py --train --mode pretrain $(PRETRAIN_ARGS)

pretrain-resume:
	python minigpt.py --train --mode pretrain --resume $(PRETRAIN_RESUME_ARGS)

chat-pretrain:
	python chat.py --checkpoint checkpoint/minigpt_pretrain.pt --temperature 0.8

# ── SFT（从预训练模型微调）──

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
