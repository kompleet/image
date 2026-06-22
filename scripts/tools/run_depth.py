#!/usr/bin/env python3
"""Runner d'estimation de profondeur (Depth Anything V2 via transformers).

Charge le modèle local fourni par --model-dir, calcule une carte de profondeur
en niveaux de gris (clair = proche, sombre = loin) et l'enregistre. Lancé en
sous-process pour ne pas verrouiller les DLL de torch dans le process Gradio.
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
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    try:
        from transformers import pipeline
    except ImportError:
        sys.exit("transformers manquant. Réinstallez l'outil depuis le Toolkit.")

    device = 0 if torch.cuda.is_available() else -1
    print(f"[depth] chargement du modèle sur "
          f"{'cuda' if device == 0 else 'cpu'}…", flush=True)
    pipe = pipeline("depth-estimation", model=args.model_dir, device=device)

    img = Image.open(args.input).convert("RGB")
    print(f"[depth] estimation ({img.width}x{img.height})…", flush=True)
    res = pipe(img)

    depth = res.get("depth") if isinstance(res, dict) else None
    if depth is None:
        # Repli : normaliser le tenseur brut en niveaux de gris.
        pd = res["predicted_depth"].squeeze().detach().cpu().numpy().astype("float32")
        pd = (pd - pd.min()) / (pd.max() - pd.min() + 1e-8)
        depth = Image.fromarray((pd * 255).round().astype("uint8"))

    depth = depth.convert("L")
    if depth.size != img.size:
        depth = depth.resize(img.size, Image.BICUBIC)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (Path(args.input).stem + "_depth.png")
    depth.save(dest)
    print(f"[depth] carte écrite : {dest}", flush=True)


if __name__ == "__main__":
    main()
