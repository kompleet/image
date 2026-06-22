#!/usr/bin/env python3
"""Installe les outils du Toolkit (Python embarqué, aucune commande à taper).

Outils :
  depth  -> Depth Anything V2 (Small) : carte de profondeur depuis une image.

Réutilise les helpers torch CUDA de setup_upscalers (build adaptée au GPU,
sans verrouiller de DLL). Lançable depuis l'interface ou en ligne :
    python scripts/setup_tools.py depth
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from atelier import settings  # noqa: E402
from setup_upscalers import ensure_torch_cuda, pin_numpy, sh  # noqa: E402

# Modèle léger (~100 Mo) : rapide, tourne même sur Pascal (GTX 10xx) et en CPU.
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Small-hf"


def install_depth():
    model_dir = settings.ROOT / "tools_repo" / "depth" / "model"
    ensure_torch_cuda()
    print("Installation de transformers…")
    sh([sys.executable, "-m", "pip", "install", "transformers>=4.45", "pillow"])
    print(f"\nTéléchargement du modèle de profondeur ({DEPTH_REPO})…")
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=DEPTH_REPO, local_dir=str(model_dir))
    pin_numpy()  # transformers peut réintroduire NumPy 2 -> on re-fige
    print("\n[OK] Depth Anything V2 installé. Disponible dans l'onglet Toolkit.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tool", choices=["depth"])
    args = ap.parse_args()
    settings.configure_hf_env()
    if args.tool == "depth":
        install_depth()


if __name__ == "__main__":
    main()
