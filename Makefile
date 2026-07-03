.PHONY: data train resume generate chat all clean help

# 默认参数
ARGS ?=
# 语言: en=英文, zh=中文, both=中英混合（不要用 LANG，和系统环境变量冲突）
MODEL_LANG ?= both

help:
	@echo "Mini GPT — Makefile"
	@echo ""
	@echo "用法:"
	@echo "  make data        下载数据集"
	@echo "  make train       训练模型"
	@echo "  make resume      从 checkpoint 续训"
	@echo "  make generate    生成文本"
	@echo "  make chat        交互式生成"
	@echo "  make all         下载 → 训练 → 生成"
	@echo "  make clean       删除训练产物"
	@echo ""
	@echo "语言选择:"
	@echo "  make MODEL_LANG=en <cmd>    英文"
	@echo "  make MODEL_LANG=zh <cmd>    中文（西游记）"
	@echo "  make MODEL_LANG=both <cmd>  中英混合（默认）"
	@echo ""
	@echo "传参示例:"
	@echo "  make MODEL_LANG=en train  ARGS=\"--max-iters 5000\""
	@echo "  make MODEL_LANG=both chat ARGS=\"--temperature 0.7\""

data:
	python minigpt.py --download --lang $(MODEL_LANG)

train: data
	python minigpt.py --train --lang $(MODEL_LANG) $(ARGS)

resume:
	python minigpt.py --train --resume --lang $(MODEL_LANG) $(ARGS)

generate:
	python minigpt.py --generate --lang $(MODEL_LANG) $(ARGS)

chat:
	python chat.py --lang $(MODEL_LANG) $(ARGS)

all: data train generate

clean:
	rm -f minigpt_*.pt
	rm -rf __pycache__
	@echo "已清理训练产物"
