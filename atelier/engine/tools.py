"""Outils du Toolkit (PyTorch, installés à la demande).

depth : carte de profondeur via Depth Anything V2 (transformers). Installation
et exécution en sous-process (Python embarqué), comme les upscalers.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

from PIL import Image

from .. import hardware, settings

TOOLS_DIR = settings.ROOT / "tools_repo"
DEPTH_MODEL_DIR = TOOLS_DIR / "depth" / "model"

_IMG_EXT = (".png", ".jpg", ".jpeg", ".webp")


class ToolError(RuntimeError):
    pass


def depth_is_installed() -> bool:
    if not DEPTH_MODEL_DIR.is_dir():
        return False
    return any(DEPTH_MODEL_DIR.rglob("*.safetensors")) \
        or any(DEPTH_MODEL_DIR.rglob("*.bin"))


def install_depth_stream():
    """Installe l'outil de profondeur en streamant le journal (pour l'UI)."""
    setup = settings.ROOT / "scripts" / "setup_tools.py"
    cmd = [sys.executable, str(setup), "depth"]
    buf: list[str] = [f"$ {' '.join(cmd)}", ""]
    yield "\n".join(buf)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT),
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        buf.append(line.rstrip("\n"))
        yield "\n".join(buf[-500:])
    code = proc.wait()
    buf.append("")
    buf.append("✅ Installation terminée." if code == 0
               else f"❌ Échec (code {code}). Voir le journal ci-dessus.")
    yield "\n".join(buf[-500:])


def _gpu_index() -> int | None:
    prefs = settings.load_prefs()
    if prefs.get("gpu_index") is not None:
        return prefs["gpu_index"]
    prof = hardware.auto_profile()
    return prof.gpu.index if prof.gpu else None


def depth_map(
    image: Image.Image | str | Path,
    log: Callable[[str], None] | None = None,
) -> Path:
    if not depth_is_installed():
        raise ToolError("L'outil de profondeur n'est pas installé "
                        "(bouton « Installer » du Toolkit).")
    settings.ensure_dirs()
    if isinstance(image, (str, Path)):
        src = Path(image)
    else:
        src = settings.TMP_DIR / f"depth_src_{int(time.time()*1000)}.png"
        image.save(src)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out_dir = settings.TMP_DIR / f"depth_out_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = settings.ROOT / "scripts" / "tools" / "run_depth.py"
    cmd = [sys.executable, str(runner),
           "--model-dir", str(DEPTH_MODEL_DIR),
           "--input", str(src),
           "--output-dir", str(out_dir)]

    env = dict(os.environ)
    gi = _gpu_index()
    if gi is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gi)

    if log:
        log("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, cwd=str(settings.ROOT), env=env,
                            encoding="utf-8", errors="replace")
    assert proc.stdout is not None
    for line in proc.stdout:
        if log:
            log(line.rstrip("\n"))
    if proc.wait() != 0:
        raise ToolError("L'estimation de profondeur a échoué (voir le journal).")

    produced = sorted(p for p in out_dir.rglob("*")
                      if p.suffix.lower() in _IMG_EXT)
    if not produced:
        raise ToolError("Aucune carte de profondeur produite.")

    final = settings.OUTPUT_DIR / f"depth-{stamp}.png"
    Image.open(produced[0]).save(final)
    return final
