.PHONY: data train resume generate chat all clean help

# 默认参数
ARGS ?=

# 数据集
DATA_FILE = data/tinyshakespeare.txt

help:
	@echo "Mini GPT — Makefile"
	@echo ""
	@echo "用法:"
	@echo "  make data        下载数据集"
	@echo "  make train       训练模型 (默认 1000 步)"
	@echo "  make resume      从 checkpoint 续训"
	@echo "  make generate    生成文本"
	@echo "  make chat        交互式生成"
	@echo "  make all         下载 → 训练 → 生成"
	@echo "  make clean       删除训练产物"
	@echo ""
	@echo "传参示例:"
	@echo "  make train   ARGS=\"--max-iters 5000 --lr 5e-4\""
	@echo "  make resume  ARGS=\"--max-iters 3000\""
	@echo "  make generate ARGS=\"--prompt \\\"To be\\\" --temperature 0.6\""
	@echo "  make chat    ARGS=\"--temperature 0.7 --top-k 30\""

data: $(DATA_FILE)

$(DATA_FILE):
	python minigpt.py --download

train: $(DATA_FILE)
	python minigpt.py --train $(ARGS)

resume:
	python minigpt.py --train --resume $(ARGS)

generate:
	python minigpt.py --generate $(ARGS)

chat:
	python chat.py $(ARGS)

all: data train generate

clean:
	rm -f minigpt.pt minigpt_checkpoint.pt
	rm -rf __pycache__
	@echo "已清理训练产物"

clean-all: clean
	rm -f $(DATA_FILE)
	@echo "已清理所有文件（含数据集）"
