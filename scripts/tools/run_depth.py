#!/usr/bin/env python3
"""Runner d'estimation de profondeur (Depth Anything V2 via transformers).

Deux modes :
  depth  -> carte de profondeur en niveaux de gris (clair = proche).
  normal -> carte de normales (RGB) dérivée des gradients de la profondeur.

Lancé en sous-process pour ne pas verrouiller les DLL de torch dans Gradio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def _normal_from_depth(depth_l, strength: float):
    """Carte de normales tangentes depuis une profondeur (PIL « L »)."""
    import numpy as np
    z = np.asarray(depth_l, dtype="float32") / 255.0
    zy, zx = np.gradient(z)
    nx = -zx * strength
    ny = -zy * strength
    nz = np.ones_like(z)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz) + 1e-8
    nx, ny, nz = nx / norm, ny / norm, nz / norm
    rgb = np.stack([nx * 0.5 + 0.5, ny * 0.5 + 0.5, nz * 0.5 + 0.5], axis=-1)
    from PIL import Image
    return Image.fromarray((rgb * 255).round().astype("uint8"), "RGB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--mode", choices=["depth", "normal"], default="depth")
    ap.add_argument("--strength", type=float, default=2.0,
                    help="intensité du relief (mode normal)")
    args = ap.parse_args()

    import torch
    from PIL import Image
    try:
        from transformers import pipeline
    except ImportError:
        sys.exit("transformers manquant. Réinstallez l'outil depuis le Toolkit.")

    device = 0 if torch.cuda.is_available() else -1
    print(f"[{args.mode}] chargement du modèle sur "
          f"{'cuda' if device == 0 else 'cpu'}…", flush=True)
    pipe = pipeline("depth-estimation", model=args.model_dir, device=device)

    img = Image.open(args.input).convert("RGB")
    print(f"[{args.mode}] estimation ({img.width}x{img.height})…", flush=True)
    res = pipe(img)

    depth = res.get("depth") if isinstance(res, dict) else None
    if depth is None:
        pd = res["predicted_depth"].squeeze().detach().cpu().numpy().astype("float32")
        pd = (pd - pd.min()) / (pd.max() - pd.min() + 1e-8)
        depth = Image.fromarray((pd * 255).round().astype("uint8"))
    depth = depth.convert("L")
    if depth.size != img.size:
        depth = depth.resize(img.size, Image.BICUBIC)

    if args.mode == "normal":
        out_img = _normal_from_depth(depth, args.strength)
        suffix = "_normal"
    else:
        out_img = depth
        suffix = "_depth"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (Path(args.input).stem + suffix + ".png")
    out_img.save(dest)
    print(f"[{args.mode}] carte écrite : {dest}", flush=True)


if __name__ == "__main__":
    main()
