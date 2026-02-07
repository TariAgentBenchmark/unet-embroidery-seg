#!/bin/bash
# 下载 SAM3 模型脚本（使用 ModelScope）
# 适合在国内网络环境下使用

set -e

echo "=========================================="
echo "SAM3 Model Download Script (ModelScope)"
echo "=========================================="

# 检查 modelscope
if ! command -v modelscope &> /dev/null; then
    echo "Error: modelscope not found"
    echo "Please install it: pip install modelscope"
    exit 1
fi

# 创建目录
echo ""
echo "Creating directory..."
mkdir -p weights/sam3

# 下载模型文件
echo ""
echo "Downloading SAM3 model files from ModelScope..."
echo "This may take a while (model size ~2-3GB)..."

modelscope download --model facebook/sam3 --local_dir weights/sam3

echo ""
echo "=========================================="
echo "Model downloaded successfully!"
echo "Location: weights/sam3/"
echo "=========================================="
echo ""
echo "You can now run the relabel script:"
echo "  python sam3_relabel.py relabel --checkpoint weights/sam3/sam3.pt"
