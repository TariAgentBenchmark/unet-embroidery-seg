import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from transformers import Sam3Processor, Sam3Model


DEFAULT_CLASS_MAP = {
    "动物类": {"class_id": 1, "prompt": "animal pattern"},
    "复合类": {"class_id": 2, "prompt": "composite pattern"},
    "生活类": {"class_id": 3, "prompt": "life pattern"},
    "植物类": {"class_id": 4, "prompt": "plant pattern"},
}


def load_class_map(path: str | None):
    if not path:
        return DEFAULT_CLASS_MAP
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def infer_class_key(stem: str, class_map: dict):
    for key in class_map.keys():
        if stem.startswith(key):
            return key
    return None


def list_image_ids(dataset_root: Path, split: str):
    images_dir = dataset_root / "JPEGImages"
    if split == "all":
        return sorted([p.stem for p in images_dir.glob("*.jpg")])

    split_file = dataset_root / "ImageSets" / "Segmentation" / f"{split}.txt"
    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    with open(split_file, "r", encoding="utf-8") as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def parse_dtype(name: str):
    name = name.lower()
    if name in ("fp16", "float16"):
        return torch.float16
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float32


def build_mask_from_results(results):
    masks = results.get("masks")
    if masks is None or len(masks) == 0:
        return None
    if isinstance(masks, torch.Tensor):
        masks = masks.cpu().numpy()
    masks = masks.astype(np.uint8)
    union = np.any(masks > 0, axis=0).astype(np.uint8)
    return union


def main():
    parser = argparse.ArgumentParser(description="Re-annotate masks with SAM3 using text prompts.")
    parser.add_argument("--dataset-root", default="raw_datasets/VOCdevkit/VOC2012", help="VOC2012 root dir")
    parser.add_argument("--output-dir", default="", help="Output mask dir (default: <dataset_root>/SegmentationClass_sam3)")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"], help="Which split to process")
    parser.add_argument("--class-map", default="", help="JSON mapping for class prompts/ids")
    parser.add_argument("--prompt", default="", help="Use a fixed prompt for all images (overrides class-map)")
    parser.add_argument("--class-id", type=int, default=1, help="Class id to use with fixed prompt (non-binary)")
    parser.add_argument("--binary", action=argparse.BooleanOptionalAction, default=False, help="Output binary masks (0/1)")
    parser.add_argument("--model-id", default="facebook/sam3", help="SAM3 model id or local path")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--dtype", default="bf16", choices=["fp32", "fp16", "bf16"], help="Model dtype on GPU")
    parser.add_argument("--threshold", type=float, default=0.5, help="Instance threshold")
    parser.add_argument("--mask-threshold", type=float, default=0.5, help="Mask threshold")
    parser.add_argument("--max-images", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--start-index", type=int, default=0, help="Start index for slicing")
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True,
                        help="Skip if output mask already exists")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    images_dir = dataset_root / "JPEGImages"
    output_dir = Path(args.output_dir) if args.output_dir else (dataset_root / "SegmentationClass_sam3")
    output_dir.mkdir(parents=True, exist_ok=True)

    class_map = load_class_map(args.class_map)
    use_fixed_prompt = bool(args.prompt.strip())

    image_ids = list_image_ids(dataset_root, args.split)
    if args.start_index > 0:
        image_ids = image_ids[args.start_index:]
    if args.max_images and args.max_images > 0:
        image_ids = image_ids[: args.max_images]

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dtype = parse_dtype(args.dtype) if device.type == "cuda" else torch.float32

    model = Sam3Model.from_pretrained(args.model_id, torch_dtype=dtype).to(device)
    processor = Sam3Processor.from_pretrained(args.model_id)
    model.eval()

    processed = 0
    skipped = 0
    missing = 0

    for image_id in tqdm(image_ids, desc="SAM3 re-annotate"):
        img_path = images_dir / f"{image_id}.jpg"
        if not img_path.exists():
            missing += 1
            continue

        if use_fixed_prompt:
            class_id = int(args.class_id)
            prompt = args.prompt.strip()
        else:
            class_key = infer_class_key(image_id, class_map)
            if class_key is None:
                missing += 1
                continue
            class_id = int(class_map[class_key]["class_id"])
            prompt = class_map[class_key]["prompt"]

        out_path = output_dir / f"{image_id}.png"
        if args.skip_existing and out_path.exists():
            skipped += 1
            continue

        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_instance_segmentation(
            outputs,
            threshold=args.threshold,
            mask_threshold=args.mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist(),
        )[0]

        union_mask = build_mask_from_results(results)
        if union_mask is None:
            out_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        else:
            if args.binary:
                out_mask = union_mask.astype(np.uint8)
            else:
                out_mask = (union_mask * class_id).astype(np.uint8)

        Image.fromarray(out_mask, mode="L").save(out_path)
        processed += 1

    print(f"Done. processed={processed}, skipped={skipped}, missing={missing}, output={output_dir}")


if __name__ == "__main__":
    main()
