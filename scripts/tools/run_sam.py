#!/usr/bin/env python3
"""Runner Segment Anything (facebook/sam-vit-base via transformers).

Segmente l'objet situé au point (x, y) et renvoie un PNG RGBA : l'objet sur fond
transparent. Lancé en sous-process pour ne pas verrouiller les DLL torch.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--x", type=int, required=True)
    ap.add_argument("--y", type=int, required=True)
    args = ap.parse_args()

    import torch
    from PIL import Image
    try:
        from transformers import SamModel, SamProcessor
    except ImportError:
        sys.exit("transformers manquant. Réinstallez l'outil depuis le Toolkit.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sam] chargement du modèle sur {device}…", flush=True)
    model = SamModel.from_pretrained(args.model_dir).to(device).eval()
    processor = SamProcessor.from_pretrained(args.model_dir)

    img = Image.open(args.input).convert("RGB")
    points = [[[int(args.x), int(args.y)]]]   # [image][point][x, y]
    print(f"[sam] segmentation au point ({args.x}, {args.y})…", flush=True)
    inputs = processor(img, input_points=points, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu())
    scores = outputs.iou_scores.cpu()[0][0]      # (3,) : 3 masques candidats
    cand = masks[0][0]                            # (3, H, W)
    best = int(scores.argmax())
    mask = (cand[best].numpy().astype("uint8") * 255)

    cutout = img.convert("RGBA")
    cutout.putalpha(Image.fromarray(mask, "L"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (Path(args.input).stem + "_sam.png")
    cutout.save(dest)
    print(f"[sam] objet extrait (score {float(scores[best]):.2f}) : {dest}",
          flush=True)


if __name__ == "__main__":
    main()
