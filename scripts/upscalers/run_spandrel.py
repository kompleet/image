#!/usr/bin/env python3
"""Runner générique pour modèles d'upscale chargés par spandrel (DRCT, DAT,
SwinIR, etc.). Charge le fichier de modèle présent dans --repo-dir et l'applique
par tuiles (pour tenir en VRAM), puis sauvegarde.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_MODEL_EXT = (".safetensors", ".pth", ".pt", ".ckpt")


def _find_model(repo: Path) -> Path:
    for p in sorted(repo.rglob("*")):
        if p.suffix.lower() in _MODEL_EXT:
            return p
    sys.exit(f"Aucun fichier de modèle ({_MODEL_EXT}) dans {repo}.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=int, default=4)
    ap.add_argument("--tile", type=int, default=192)
    ap.add_argument("--overlap", type=int, default=24)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image
    try:
        from spandrel import ModelLoader
    except ImportError:
        sys.exit("spandrel manquant. Réinstallez ce moteur depuis l'onglet Upscale.")

    model_path = _find_model(Path(args.repo_dir).resolve())
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[spandrel] chargement de {model_path.name} sur {device}…", flush=True)
    model = ModelLoader().load_from_file(str(model_path)).to(device).eval()
    sc = int(getattr(model, "scale", args.scale) or args.scale)
    print(f"[spandrel] modèle ×{sc}", flush=True)

    img = Image.open(args.input).convert("RGB")
    arr = np.asarray(img).astype("float32") / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    _, c, H, W = t.shape

    out = torch.zeros((1, c, H * sc, W * sc), device=device)
    wsum = torch.zeros((1, 1, H * sc, W * sc), device=device)
    tile, ov = args.tile, args.overlap
    step = max(16, tile - ov)
    ys = list(range(0, max(1, H), step))
    xs = list(range(0, max(1, W), step))
    print(f"[spandrel] {len(ys)*len(xs)} tuiles ({H}x{W} -> {H*sc}x{W*sc})…",
          flush=True)

    with torch.no_grad():
        for y in ys:
            for x in xs:
                y2, x2 = min(y + tile, H), min(x + tile, W)
                y1, x1 = max(0, y2 - tile), max(0, x2 - tile)
                patch = t[:, :, y1:y2, x1:x2]
                op = model(patch).clamp(0, 1)
                oy, ox = y1 * sc, x1 * sc
                out[:, :, oy:oy + op.shape[2], ox:ox + op.shape[3]] += op
                wsum[:, :, oy:oy + op.shape[2], ox:ox + op.shape[3]] += 1.0
    out = out / wsum.clamp(min=1.0)

    if args.scale == 2 and sc != 2:
        out = torch.nn.functional.interpolate(out, scale_factor=2 / sc,
                                              mode="bicubic", align_corners=False)
    res = (out.squeeze(0).clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255)
    res = res.round().astype("uint8")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(args.input).name
    Image.fromarray(res).save(dest)
    print(f"[spandrel] image écrite : {dest}", flush=True)


if __name__ == "__main__":
    main()
