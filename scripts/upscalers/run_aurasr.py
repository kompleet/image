#!/usr/bin/env python3
"""Runner AuraSR v2 — upscaler GAN (GigaGAN), une seule passe, léger.

Appelé par atelier/engine/upscalers.py. Le paquet `aura-sr` télécharge les poids
(fal/AuraSR-v2) automatiquement au premier chargement. ×4 natif ; ×2 par
réduction, ×8 par double passe.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)   # non utilisé (paquet pip)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    from PIL import Image
    try:
        from aura_sr import AuraSR
    except ImportError:
        sys.exit("Paquet 'aura-sr' manquant. Réinstallez AuraSR depuis l'onglet "
                 "Upscale (bouton Installer).")

    print("[AuraSR] chargement de fal/AuraSR-v2…", flush=True)
    aura = AuraSR.from_pretrained("fal/AuraSR-v2")

    img = Image.open(args.input).convert("RGB")
    print(f"[AuraSR] image {img.size} -> upscale ×4 (overlapped)…", flush=True)
    up = aura.upscale_4x_overlapped(img)

    if args.scale == 8:
        print("[AuraSR] seconde passe ×4 -> ×8…", flush=True)
        up = aura.upscale_4x_overlapped(up).resize(
            (up.width * 2, up.height * 2), Image.LANCZOS)
    elif args.scale == 2:
        up = up.resize((up.width // 2, up.height // 2), Image.LANCZOS)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(args.input).name
    up.save(dest)
    print(f"[AuraSR] image écrite : {dest}", flush=True)


if __name__ == "__main__":
    main()
