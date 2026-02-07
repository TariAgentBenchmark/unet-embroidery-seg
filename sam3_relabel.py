#!/usr/bin/env python3
"""
使用 SAM3 重新标注 VOCdevkit 数据集

Usage:
    python sam3_relabel.py relabel --checkpoint weights/sam3/sam3.pt
    python sam3_relabel.py check
    python sam3_relabel.py download-script
"""

import os
import sys
import base64
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import numpy as np
import torch

import click

from openai import OpenAI


def get_vlm_client() -> OpenAI:
    
    api_key = os.environ.get("VLM_API_KEY")
    base_url = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        raise ValueError("VLM_API_KEY environment variable not set")
    
    return OpenAI(api_key=api_key, base_url=base_url)


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_prompt_with_vlm(
    image_path: str,
    category: str,
    client: OpenAI,
    model: str = None,
) -> str:
    """使用多模态大模型生成 prompt
    
    Args:
        image_path: 图片路径
        category: 类别名称（动物类/植物类/复合类）
        client: OpenAI 客户端
        model: 模型名称，必须从环境变量 VLM_MODEL 获取
    
    Returns:
        生成的 prompt 字符串
    """
    if model is None:
        model = os.environ.get("VLM_MODEL")
        if not model:
            raise ValueError("VLM_MODEL environment variable not set")
    
    base64_image = encode_image_to_base64(image_path)
    
    # 根据类别构建不同的提示词
    category_hints = {
        "动物类": "animal or creature patterns",
        "植物类": "floral, botanical or plant patterns",
        "复合类": "decorative, ornamental or composite patterns",
    }
    hint = category_hints.get(category, "pattern")
    
    system_prompt = f"""You are an expert in analyzing traditional embroidery and textile patterns.
Your task is to describe the main pattern/motif in the image for segmentation purposes.

The image contains: {hint}

Provide a concise description (10-20 words) that would help an AI segmentation model identify and segment the main pattern. Focus on:
- The type of pattern/motif
- Key visual characteristics
- Colors and shapes
- Style (traditional, modern, etc.)

Output only the description, nothing else."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            },
        ],
        max_tokens=100,
        temperature=0.3,
    )
    
    prompt = response.choices[0].message.content.strip()
    return prompt

# 类别到 prompt 的映射
CATEGORY_PROMPTS = {
    "动物类": [
        "Traditional Chinese Ruyi cloud motif, quadrilobed symmetrical scroll pattern, four interlocking S-shaped volutes, auspicious cloud embroidery design",
    ],
    "植物类": [
        "Traditional Chinese Ruyi cloud motif, quadrilobed symmetrical scroll pattern, four interlocking S-shaped volutes, auspicious cloud embroidery design",
    ],
    "复合类": [
        "Traditional Chinese Ruyi cloud motif, quadrilobed symmetrical scroll pattern, four interlocking S-shaped volutes, auspicious cloud embroidery design",
    ],
}


def check_file(path: str, description: str) -> bool:
    """检查文件是否存在"""
    if os.path.exists(path):
        size = os.path.getsize(path) / 1024 / 1024  # MB
        click.echo(f"  ✓ {description}: {path} ({size:.1f} MB)")
        return True
    else:
        click.echo(f"  ✗ {description}: {path} (NOT FOUND)")
        return False


@click.group()
def cli():
    """SAM3 VOCdevkit 重新标注工具"""
    pass


@cli.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default="raw_datasets/VOCdevkit/VOC2012/JPEGImages",
    help="输入图像目录",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default="raw_datasets/VOCdevkit_SAM3/VOC2012/JPEGImages",
    help="输出标注目录",
)
@click.option(
    "--checkpoint",
    type=click.Path(exists=False, dir_okay=False),
    default="weights/sam3/sam3.pt",
    help="SAM3 模型检查点路径",
)
@click.option(
    "--categories",
    multiple=True,
    default=["动物类", "植物类", "复合类"],
    help="要处理的类别（可多次指定）",
)
@click.option(
    "--confidence",
    type=float,
    default=0.3,
    help="置信度阈值",
)
@click.option(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="计算设备 (cuda/cpu)",
)
@click.option(
    "--max-images",
    type=int,
    default=None,
    help="每个类别最大处理图像数（用于测试）",
)
@click.option(
    "--use-vlm",
    is_flag=True,
    help="使用多模态大模型生成 prompt（需要配置环境变量 VLM_API_KEY, VLM_BASE_URL, VLM_MODEL）",
)
@click.option(
    "--vlm-cache",
    type=click.Path(file_okay=False, path_type=Path),
    default=None,
    help="VLM prompt 缓存目录（避免重复调用 API）",
)
def relabel(
    input_dir: Path,
    output_dir: Path,
    checkpoint: str,
    categories: Tuple[str],
    confidence: float,
    device: str,
    max_images: Optional[int],
    use_vlm: bool,
    vlm_cache: Optional[Path],
):
    """使用 SAM3 重新标注数据集"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    click.echo(f"Device: {device}")
    
    # 如果使用 VLM，检查环境变量
    vlm_client = None
    vlm_model = None
    if use_vlm:
        vlm_model = os.environ.get("VLM_MODEL")
        if not vlm_model:
            raise ValueError("VLM_MODEL environment variable not set (required when using --use-vlm)")
        vlm_client = get_vlm_client()
        click.echo(f"VLM Model: {vlm_model}")
        click.echo(f"VLM Base URL: {os.environ.get('VLM_BASE_URL', 'https://api.openai.com/v1')}")
        
        if vlm_cache:
            vlm_cache.mkdir(parents=True, exist_ok=True)
            click.echo(f"VLM cache: {vlm_cache}")
    
    # 加载 SAM3 模型
    model, processor = load_sam3_model(checkpoint, device)
    
    # 处理每个类别
    for category in categories:
        process_category(
            model,
            processor,
            input_dir,
            output_dir,
            category,
            confidence,
            max_images,
            use_vlm,
            vlm_client,
            vlm_model,
            vlm_cache,
        )
    
    click.echo("\nDone!")


@cli.command()
def check():
    """检查 SAM3 环境是否配置正确"""
    
    click.echo("=" * 60)
    click.echo("SAM3 Environment Check")
    click.echo("=" * 60)
    
    all_ok = True
    
    # 检查 Python 版本
    click.echo("\n[Python Version]")
    version = sys.version_info
    click.echo(f"  Python {version.major}.{version.minor}.{version.micro}")
    if version.major == 3 and version.minor >= 12:
        click.echo("  ✓ Python version is compatible (>= 3.12)")
    else:
        click.echo("  ✗ Python version must be >= 3.12")
        all_ok = False
    
    # 检查依赖包
    click.echo("\n[Dependencies]")
    packages = [
        ("sam3", "sam3"),
        ("torch", "torch"),
        ("numpy", "numpy"),
        ("PIL", "PIL"),
        ("skimage", "skimage"),
        ("psutil", "psutil"),
        ("click", "click"),
    ]
    for name, import_name in packages:
        __import__(import_name)
        click.echo(f"  ✓ {name}")
    
    # 检查 BPE Tokenizer
    click.echo("\n[BPE Tokenizer]")
    venv_path = Path(sys.executable).parent.parent
    py_version = f"python{version.major}.{version.minor}"
    bpe_path = venv_path / "lib" / py_version / "site-packages" / "assets" / "bpe_simple_vocab_16e6.txt.gz"
    if not check_file(str(bpe_path), "BPE Tokenizer"):
        click.echo("\n  Run the following to fix:")
        click.echo(f"  mkdir -p {bpe_path.parent}")
        click.echo(f"  curl -L -o {bpe_path} \\")
        click.echo(f'    "https://raw.githubusercontent.com/openai/CLIP/main/clip/bpe_simple_vocab_16e6.txt.gz"')
        all_ok = False
    
    # 检查模型文件
    click.echo("\n[Model Checkpoint]")
    has_model = check_file("weights/sam3/sam3.pt", "sam3.pt")
    has_config = check_file("weights/sam3/config.json", "config.json")
    
    if not has_model or not has_config:
        all_ok = False
        click.echo("\n  To download the model, run:")
        click.echo("  python sam3_relabel.py download-script")
        click.echo("  bash download_sam3_model.sh")
        click.echo("  (Uses ModelScope for faster download in China)")
    
    # 检查数据集
    click.echo("\n[Dataset]")
    dataset_path = Path("raw_datasets/VOCdevkit/VOC2012/JPEGImages")
    if dataset_path.exists():
        num_images = len(list(dataset_path.glob("*.png")))
        click.echo(f"  ✓ {dataset_path} ({num_images} images)")
    else:
        click.echo(f"  ✗ {dataset_path} (NOT FOUND)")
    
    # 检查 CUDA
    click.echo("\n[CUDA]")
    if torch.cuda.is_available():
        click.echo(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        click.echo(f"    Memory: {mem:.1f} GB")
    else:
        click.echo("  ! CUDA not available (will use CPU)")
    
    # 检查 VLM 环境变量
    click.echo("\n[VLM Configuration]")
    vlm_api_key = os.environ.get("VLM_API_KEY")
    vlm_base_url = os.environ.get("VLM_BASE_URL", "https://api.openai.com/v1")
    vlm_model = os.environ.get("VLM_MODEL")
    
    if vlm_api_key:
        masked_key = vlm_api_key[:8] + "..." + vlm_api_key[-4:] if len(vlm_api_key) > 12 else "***"
        click.echo(f"  ✓ VLM_API_KEY: {masked_key}")
    else:
        click.echo("  ! VLM_API_KEY: not set (required for --use-vlm)")
    
    if vlm_model:
        click.echo(f"  ✓ VLM_MODEL: {vlm_model}")
    else:
        click.echo("  ! VLM_MODEL: not set (required for --use-vlm)")
    
    click.echo(f"  • VLM_BASE_URL: {vlm_base_url} (optional, has default)")
    
    # 总结
    click.echo("\n" + "=" * 60)
    if all_ok:
        click.echo("✓ All checks passed! You can run SAM3 now.")
        click.echo("\nBasic usage:")
        click.echo("  python sam3_relabel.py relabel --checkpoint weights/sam3/sam3.pt")
        click.echo("\nWith VLM-generated prompts:")
        click.echo("  export VLM_API_KEY=your_key")
        click.echo("  export VLM_MODEL=gpt-4o")
        click.echo("  python sam3_relabel.py relabel --checkpoint weights/sam3/sam3.pt --use-vlm")
    else:
        click.echo("✗ Some checks failed. Please fix the issues above.")
    click.echo("=" * 60)


@cli.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(file_okay=True, dir_okay=False),
    default="download_sam3_model.sh",
    help="输出脚本文件名",
)
def download_script(output: str):
    """生成模型下载脚本（使用 ModelScope）"""
    
    script = '''#!/bin/bash
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
'''
    
    with open(output, "w") as f:
        f.write(script)
    
    # 添加执行权限
    os.chmod(output, 0o755)
    
    click.echo(f"Generated {output}")
    click.echo("Run this script to download the model from ModelScope:")
    click.echo(f"  bash {output}")
    click.echo("")
    click.echo("Then run: python sam3_relabel.py relabel --checkpoint weights/sam3/sam3.pt")


# ============ 功能函数 ============

def load_sam3_model(checkpoint_path: Optional[str] = None, device: str = "cuda"):
    """加载 SAM3 模型"""
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    
    click.echo("Loading SAM3 model...")
    click.echo(f"Device: {device}")
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        click.echo(f"Using local checkpoint: {checkpoint_path}")
        model = build_sam3_image_model(
            checkpoint_path=checkpoint_path,
            load_from_HF=False,
        )
    else:
        click.echo("Attempting to download from HuggingFace...")
        if checkpoint_path:
            click.echo(f"Warning: Checkpoint not found at {checkpoint_path}")
        model = build_sam3_image_model()
    
    model = model.to(device)
    model.eval()
    
    processor = Sam3Processor(model)
    
    click.echo("Model loaded successfully!")
    return model, processor


def load_image(image_path: str) -> Image.Image:
    """加载图像"""
    return Image.open(image_path).convert("RGB")


def segment_with_sam3(
    model,
    processor,
    image: Image.Image,
    prompts: List[str],
    confidence_threshold: float = 0.3,
) -> Tuple[List[np.ndarray], List[str], List[float]]:
    """使用 SAM3 对图像进行分割"""
    inference_state = processor.set_image(image)
    
    all_masks = []
    all_labels = []
    all_scores = []
    
    for prompt in prompts:
        output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        masks = output["masks"]
        scores = output["scores"]
        
        for i, score in enumerate(scores):
            if score >= confidence_threshold:
                mask = masks[i].cpu().numpy() if torch.is_tensor(masks[i]) else masks[i]
                all_masks.append(mask)
                all_labels.append(prompt)
                all_scores.append(float(score))
    
    return all_masks, all_labels, all_scores


def merge_masks(masks: List[np.ndarray], image_size: Tuple[int, int]) -> np.ndarray:
    """合并多个 mask 为一张图片
    
    Args:
        masks: mask 列表
        image_size: (height, width)
    
    Returns:
        合并后的 mask (0-255)
    """
    height, width = image_size
    merged = np.zeros((height, width), dtype=np.uint8)
    
    for mask in masks:
        # 确保 mask 是 2D
        if mask.ndim > 2:
            mask = mask.squeeze()
        
        # 归一化到 0-255
        if mask.dtype == np.bool_ or mask.max() <= 1.0:
            mask = (mask * 255).astype(np.uint8)
        else:
            mask = mask.astype(np.uint8)
        
        # 调整大小以匹配图像
        if mask.shape[0] != height or mask.shape[1] != width:
            from PIL import Image as PILImage
            mask_pil = PILImage.fromarray(mask)
            mask_pil = mask_pil.resize((width, height), PILImage.NEAREST)
            mask = np.array(mask_pil)
        
        # 合并（取并集）
        merged = np.maximum(merged, mask)
    
    return merged


def process_category(
    model,
    processor,
    input_dir: Path,
    output_dir: Path,
    category: str,
    confidence_threshold: float = 0.3,
    max_images: Optional[int] = None,
    use_vlm: bool = False,
    vlm_client = None,
    vlm_model: str = None,
    vlm_cache: Optional[Path] = None,
):
    """处理一个类别的所有图像"""
    # 默认 prompts
    default_prompts = CATEGORY_PROMPTS.get(category, [category])
    
    image_files = sorted(input_dir.glob(f"{category}*.png"))
    
    if max_images:
        image_files = image_files[:max_images]
    
    click.echo(f"\nProcessing category: {category}")
    click.echo(f"Found {len(image_files)} images")
    if use_vlm:
        click.echo(f"Using VLM to generate prompts (model: {vlm_model})")
    else:
        click.echo(f"Using default prompts: {default_prompts}")
    
    for i, image_path in enumerate(image_files):
        click.echo(f"  [{i+1}/{len(image_files)}] {image_path.name}")
        
        output_path = output_dir / f"{image_path.stem}.png"
        if output_path.exists():
            click.echo("    Already processed, skipping")
            continue
        
        image = load_image(str(image_path))
        
        # 确定 prompts
        prompts = default_prompts
        if use_vlm:
            cache_file = None
            if vlm_cache:
                cache_file = vlm_cache / f"{image_path.stem}_prompt.txt"
            
            prompts = get_or_generate_prompt(
                str(image_path),
                category,
                vlm_client,
                vlm_model,
                cache_file,
            )
            click.echo(f"    VLM prompt: {prompts[0][:80]}...")
        
        masks, labels, scores = segment_with_sam3(
            model, processor, image, prompts, confidence_threshold
        )
        
        if not masks:
            click.echo("    No masks found")
            # 创建空白 mask
            mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            click.echo(f"    Found {len(masks)} masks")
            # 合并所有 masks
            mask = merge_masks(masks, (image.height, image.width))
        
        # 保存为 PNG
        mask_img = Image.fromarray(mask)
        mask_img.save(output_path)


def get_or_generate_prompt(
    image_path: str,
    category: str,
    client,
    model: str,
    cache_file: Optional[Path] = None,
) -> List[str]:
    """获取或生成 prompt
    
    如果缓存文件存在，直接读取；否则调用 VLM 生成并缓存
    
    Returns:
        prompt 列表（单元素列表，兼容原有接口）
    """
    # 检查缓存
    if cache_file and cache_file.exists():
        prompt = cache_file.read_text().strip()
        return [prompt]
    
    # 调用 VLM 生成
    prompt = generate_prompt_with_vlm(image_path, category, client, model)
    
    # 保存缓存
    if cache_file:
        cache_file.write_text(prompt)
    
    return [prompt]


if __name__ == "__main__":
    cli()
