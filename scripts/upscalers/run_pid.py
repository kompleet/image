#!/usr/bin/env python3
"""Runner NVIDIA PiD.

PiD (Pixel Diffusion Decoder, nv-tlabs/PiD) est avant tout un *décodeur de
latents* haute résolution, pas un upscaler d'image PNG « générique ». Son point
d'entrée et ses arguments varient selon le dépôt. Ce runner :
  1. liste les scripts Python du dépôt (affichés dans le journal),
  2. tente de détecter un vrai script CLI (avec argparse + __main__),
  3. l'exécute avec un jeu d'arguments probable.

Si rien ne sort, la liste affichée permet de câbler le bon script/arguments.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Mots du nom de fichier qui DISQUALIFIENT (modules, pas des points d'entrée).
_SKIP = ("utils", "__init__", "setup", "config", "dataset", "loader",
         "model", "module", "network", "layer", "blocks", "registry", "__main__")
# Mots qui suggèrent un point d'entrée d'inférence.
_GOOD = ("inference", "sample", "demo", "run", "main", "generate",
         "upscale", "super_res", "sr", "decode", "infer", "test_sr")


def _is_cli(path: Path) -> bool:
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return ("__main__" in txt and
            ("argparse" in txt or "ArgumentParser" in txt or "click" in txt))


def _score(path: Path) -> int:
    name = path.stem.lower()
    if any(s in name for s in _SKIP):
        return -1000
    s = 0
    if name in _GOOD:
        s += 6
    elif any(g in name for g in _GOOD):
        s += 3
    if _is_cli(path):
        s += 5
    # privilégie les scripts proches de la racine ou dans scripts/projects/tools
    parts = [p.lower() for p in path.parts]
    if any(d in parts for d in ("scripts", "projects", "tools", "demo")):
        s += 2
    s -= len(path.parts)  # moins profond = mieux
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=int, default=4)
    args = ap.parse_args()

    repo = Path(args.repo_dir).resolve()
    pys = sorted(p for p in repo.rglob("*.py")
                 if "__pycache__" not in p.parts)

    print("Scripts Python du dépôt PiD :")
    for p in pys:
        tag = " [CLI]" if _is_cli(p) else ""
        print(f"   - {p.relative_to(repo).as_posix()}{tag}")
    print()

    cli = [p for p in pys if _is_cli(p) and _score(p) > -1000]
    cli.sort(key=_score, reverse=True)
    if not cli:
        print("[PiD] Aucun script CLI détecté automatiquement.")
        print("[PiD] Copiez la liste ci-dessus pour câbler le bon point d'entrée.")
        raise SystemExit(2)

    script = cli[0]
    ckpt = repo / "ckpts"
    cmd = [sys.executable, str(script),
           "--input", str(Path(args.input).resolve()),
           "--output", str(Path(args.output_dir).resolve()),
           "--scale", str(args.scale)]
    if ckpt.is_dir():
        cmd += ["--ckpt", str(ckpt)]
    print(f"[PiD] Script détecté : {script.relative_to(repo).as_posix()}")
    print("$", " ".join(cmd), flush=True)
    code = subprocess.call(cmd, cwd=str(repo))
    if code != 0:
        print(f"\n[PiD] Échec (code {code}). L'API du dépôt diffère probablement.")
        print("[PiD] Envoyez la liste des scripts ci-dessus pour ajuster run_pid.py.")
    raise SystemExit(code)


if __name__ == "__main__":
    main()
