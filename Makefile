.PHONY: help download prepare tokenizer pretrain sft pretrain-resume sft-resume chat-pretrain chat-sft check clean

# ── 通用 ──
CHAT_ARGS ?= --temperature 0.8 --device cpu
MODEL_LANG ?= both

BATCH_SIZE ?= 4
PROMPT ?= "你好"
N_TOKENS ?= 256
THREADS ?= 8
TEMP ?= 0.8

# ── 预训练参数 ──
PRETRAIN_ARGS ?= --preset 200M --max-iters 50000 --batch-size $(BATCH_SIZE)
PRETRAIN_RESUME_ARGS ?= --preset 200M --max-iters 100000 --batch-size $(BATCH_SIZE)

# ── SFT 参数 ──
SFT_DATA ?= data/sft/sft_t2t_mini.jsonl data/sft/moss_sft.jsonl data/sft/yuki_ruozhiba_1.5k.jsonl
SFT_ARGS ?= --dropout 0.1 --preset 200M --batch-size $(BATCH_SIZE) --max-iters 50000 --lr 1e-4

download:
	python download_data.py

prepare: download
	python prepare_data.py pretrain
	python prepare_data.py sft

tokenizer: prepare
	python tokenizer.py --files data/pretrain/tinyshakespeare.txt data/pretrain/xyj.txt data/pretrain/hlm.txt data/sft/yuki_ruozhiba_1.5k.jsonl --save checkpoint/tokenizer.json

clean-data:
	rm -f data/pretrain_text.txt data/sft_train.txt

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
	python chat.py --checkpoint checkpoint/minigpt_pretrain.pt $(CHAT_ARGS)

# ── SFT ──

sft: tokenizer
	python minigpt.py --train --mode sft --resume-from checkpoint/minigpt_pretrain.pt \
	  --dialogue-data $(SFT_DATA) $(SFT_ARGS)

sft-resume:
	python minigpt.py --train --mode sft --resume --dialogue-data $(SFT_DATA) $(SFT_ARGS)

chat-sft:
	python chat.py --mode combined --checkpoint checkpoint/minigpt_sft.pt $(CHAT_ARGS)


# ── 导出 ──

LLAMA_DIR ?= llama.cpp

setup-llama:
	@if [ ! -d "$(LLAMA_DIR)" ]; then \
		echo "下载 llama.cpp..."; \
		git clone --depth 1 https://github.com/ggerganov/llama.cpp.git $(LLAMA_DIR); \
	fi
	@if [ ! -f "$(LLAMA_DIR)/build/bin/llama-cli" ]; then \
		echo "编译 llama.cpp..."; \
		cd $(LLAMA_DIR) && mkdir -p build && cd build && cmake .. -DLLAMA_CUDA=OFF && make -j4 llama-cli 2>/dev/null; \
	fi
	@echo "✅ llama.cpp 就绪"

chat-llama: setup-llama gguf
	./$(LLAMA_DIR)/build/bin/llama-cli -m model.gguf -p "$(PROMPT)" -n $(N_TOKENS) -t $(THREADS) --temp $(TEMP)

gguf:
	python convert_model.py gguf --checkpoint checkpoint/minigpt_sft.pt --output model.gguf

onnx:
	python convert_model.py onnx --checkpoint checkpoint/minigpt_sft.pt --output model.onnx

# ── 清理 ──
clean:
	rm -rf checkpoint
	rm -rf __pycache__
	rm -f data/sft_train.txt data/pretrain_text.txt
	@echo "已清理"
