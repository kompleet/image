"""Pipeline de génération : assemble un GenRequest depuis la bibliothèque, les
préférences matérielles et les LoRA, puis lance stable-diffusion.cpp.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from .. import hardware, registry, settings
from . import sdcpp
from .sdcpp import GenRequest


def cancel() -> str:
    """Annule la génération en cours (termine le process sd-cli)."""
    return sdcpp.cancel_active()


def list_custom_models() -> list[str]:
    """Noms des fichiers de modèle déposés dans models/custom/ (téléchargés
    manuellement ailleurs)."""
    settings.ensure_dirs()
    out = []
    for p in sorted(settings.CUSTOM_DIR.glob("*")):
        if p.suffix.lower() in (".gguf", ".safetensors", ".sft", ".pth", ".ckpt"):
            out.append(p.name)
    return out


def custom_path(name: str | None) -> Path | None:
    if not name:
        return None
    p = settings.CUSTOM_DIR / name
    return p if p.is_file() else None


def list_loras() -> list[str]:
    """Noms des LoRA disponibles dans loras/ (sans extension)."""
    settings.ensure_dirs()
    out = []
    for p in sorted(settings.LORA_DIR.glob("*")):
        if p.suffix.lower() in (".safetensors", ".gguf", ".ckpt", ".pt"):
            out.append(p.stem)
    return out


def _component(model: registry.BaseModel, role: str) -> Path | None:
    comp = next((c for c in model.components if c.role == role), None)
    if comp is None:
        return None
    return registry.resolve_component_path(comp)


def _resolved_flags(prefs: dict) -> tuple[dict[str, bool], int | None]:
    """Flags d'optimisation effectifs + index GPU."""
    if prefs.get("auto_optimize", True):
        prof = hardware.auto_profile(prefs.get("gpu_index"))
        flags = prof.flags()
        gpu_index = prof.gpu.index if prof.gpu else None
    else:
        flags = dict(prefs.get("flags", {}))
        gpu_index = prefs.get("gpu_index")
    return flags, gpu_index


def _apply_loras(prompt: str, loras: list[tuple[str, float]]) -> str:
    """Ajoute la syntaxe <lora:nom:poids> au prompt (consommée par sd.cpp)."""
    tags = "".join(f" <lora:{name}:{weight:g}>" for name, weight in loras if name)
    return (prompt or "") + tags


def _write_prompt_sidecars(paths: list[Path], req: "GenRequest",
                           model: "registry.BaseModel", base_seed: int) -> None:
    """Écrit un .txt (style A1111) à côté de chaque image, pour retrouver le
    prompt et les réglages directement dans le dossier outputs/."""
    import time
    date = time.strftime("%Y-%m-%d %H:%M:%S")
    for i, p in enumerate(paths):
        seed = base_seed + i if (base_seed is not None and base_seed >= 0) \
            else base_seed
        lines = [req.prompt or ""]
        if req.negative:
            lines.append(f"Negative prompt: {req.negative}")
        lines.append(f"Model: {model.name} ({model.id})")
        lines.append(
            f"Steps: {req.steps}, CFG: {req.cfg_scale}, Sampler: {req.sampler}, "
            f"Scheduler: {req.schedule or 'auto'}, Size: {req.width}x{req.height}, "
            f"Seed: {seed}, Flow shift: {req.flow_shift:g}")
        if req.init_image:
            lines.append(f"img2img strength: {req.strength}")
        lines.append(f"Date: {date}")
        try:
            Path(p).with_suffix(".txt").write_text("\n".join(lines),
                                                   encoding="utf-8")
        except OSError:
            pass


def generate(
    model_id: str,
    prompt: str,
    negative: str,
    steps: int,
    cfg_scale: float,
    width: int,
    height: int,
    seed: int,
    batch_count: int,
    sampler: str | None = None,
    schedule: str = "auto",
    flow_shift: float = 0.0,
    init_image: Path | None = None,
    strength: float = 0.6,
    loras: list[tuple[str, float]] | None = None,
    diffusion_override: Path | None = None,
    vae_override: Path | None = None,
    encoder_override: Path | None = None,
    preview_path: Path | None = None,
    control_net: Path | None = None,
    control_image: Path | None = None,
    control_strength: float = 0.8,
    canny: bool = False,
    save_prompt: bool = True,
    log: Callable[[str], None] | None = None,
) -> list[Path]:
    prefs = settings.load_prefs()
    sd_cli = settings.find_sd_cli()
    if sd_cli is None:
        raise sdcpp.EngineError(
            "Binaire sd-cli introuvable. Lancez l'installation "
            "(install.bat) ou « python scripts/get_sdcpp.py ».")

    model = registry.get_base_model(model_id, prefs)
    if model is None:
        raise sdcpp.EngineError(f"Modèle inconnu : {model_id}")

    # Famille « checkpoint complet » (SDXL) : un seul fichier via -m.
    has_full = any(c.role == "model" for c in model.components)
    if has_full:
        model_path = Path(diffusion_override) if diffusion_override \
            else _component(model, "model")
        vae = Path(vae_override) if vae_override else _component(model, "vae")
        diffusion = enc = uncond = t5xxl = clip_l = None
        if model_path is None or not Path(model_path).is_file():
            raise sdcpp.EngineError(
                f"« {model.name} » : checkpoint manquant. Téléchargez-le "
                "(onglet Bibliothèque) ou fournissez un fichier local.")
    else:
        model_path = None
        diffusion = Path(diffusion_override) if diffusion_override else _component(model, "diffusion")
        vae = Path(vae_override) if vae_override else _component(model, "vae")
        enc = Path(encoder_override) if encoder_override else _component(model, "text_encoder")
        uncond = _component(model, "uncond")
        t5xxl = _component(model, "t5xxl")
        clip_l = _component(model, "clip_l")
        # Composants de texte requis : au moins un encodeur (llm OU t5xxl).
        need = {"diffusion": diffusion, "vae": vae}
        if not t5xxl:
            need["text_encoder"] = enc
        absent = [role for role, p in need.items() if p is None or not Path(p).is_file()]
        if t5xxl is not None and not Path(t5xxl).is_file():
            absent.append("t5xxl")
        if absent:
            raise sdcpp.EngineError(
                f"« {model.name} » : fichiers manquants ({', '.join(absent)}). "
                "Téléchargez le modèle (onglet Bibliothèque) ou fournissez des "
                "fichiers locaux valides.")

    flags, gpu_index = _resolved_flags(prefs)
    lora_dir = settings.LORA_DIR if loras else None
    final_prompt = _apply_loras(prompt, loras or [])

    req = GenRequest(
        diffusion_model=diffusion, vae=vae, model_path=model_path,
        text_encoder=enc, t5xxl=t5xxl, clip_l=clip_l, uncond_model=uncond,
        extra_flags=list(model.defaults.get("extra_flags", [])),
        prompt=final_prompt, negative=negative,
        steps=steps, cfg_scale=cfg_scale,
        sampler=sampler or model.defaults.get("sampler", "euler"),
        schedule="" if schedule in (None, "", "auto") else schedule,
        flow_shift=float(flow_shift or 0.0),
        width=width, height=height, seed=seed, batch_count=batch_count,
        init_image=init_image, strength=strength,
        lora_dir=lora_dir, preview_path=preview_path,
        control_net=control_net, control_image=control_image,
        control_strength=float(control_strength), canny=canny,
        flags=flags, gpu_index=gpu_index,
    )
    out = sdcpp.unique_output(model.family)
    cmd = sdcpp.build_gen_cmd(sd_cli, req, out)
    sdcpp.run(cmd, log=log, gpu_index=gpu_index)
    paths = sdcpp.collect_outputs(out, batch_count)
    if save_prompt and paths:
        _write_prompt_sidecars(paths, req, model, int(seed))
    return paths


def _feather_mask(h: int, w: int, fade: int):
    import numpy as np
    fy = np.ones(h, np.float32)
    fx = np.ones(w, np.float32)
    f = max(1, min(fade, h // 2, w // 2))
    ramp = np.linspace(0.05, 1.0, f, dtype=np.float32)
    fy[:f] = ramp; fy[-f:] = ramp[::-1]
    fx[:f] = ramp; fx[-f:] = ramp[::-1]
    return (fy[:, None] * fx[None, :])[:, :, None]


def creative_upscale(
    model_id: str,
    image,
    scale: int,
    prompt: str,
    creativity: float,
    log: Callable[[str], None] | None = None,
    tile: int = 1280,
    overlap: int = 160,
) -> Path:
    """Upscale « créatif » (façon Magnific) : pré-agrandissement Lanczos puis
    raffinage img2img à faible bruit qui ré-invente le détail.

    Comme sd.cpp ne découpe PAS la diffusion (le latent entier doit tenir en
    VRAM), on traite les grandes images PAR TUILES (img2img par tuile + recollage
    avec fondu, seed fixe pour la cohérence). `creativity` = strength img2img.
    """
    import numpy as np
    from PIL import Image
    import random as _random

    settings.ensure_dirs()
    im = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) \
        else image.convert("RGB")

    tw = max(256, int(round(im.width * scale / 16)) * 16)
    th = max(256, int(round(im.height * scale / 16)) * 16)
    if log:
        log(f"Pré-agrandissement {im.width}x{im.height} -> {tw}x{th} (Lanczos)…")
    base = im.resize((tw, th), Image.LANCZOS)

    prefs = settings.load_prefs()
    m = registry.get_base_model(model_id, prefs)
    if m is None:
        raise sdcpp.EngineError(f"Modèle de raffinage inconnu : {model_id}")
    d = m.defaults
    p = prompt or "highly detailed, sharp focus, intricate details, high quality"
    seed = _random.randint(0, 2**31 - 1)  # même seed pour toutes les tuiles
    common = dict(model_id=model_id, prompt=p, negative="",
                  steps=int(d.get("steps", 8)), cfg_scale=float(d.get("cfg_scale", 1.0)),
                  sampler=d.get("sampler", "euler"),
                  schedule=d.get("scheduler", "auto"), seed=seed, batch_count=1,
                  strength=float(creativity), save_prompt=False)

    # Petite cible -> une seule passe (tient en VRAM).
    if max(tw, th) <= 1536:
        bp = settings.TMP_DIR / "creative_base.png"
        base.save(bp)
        if log:
            log(f"Raffinage img2img via « {m.name} » (1 passe, créativité={creativity})…")
        outs = generate(width=tw, height=th, init_image=bp, log=log, **common)
        if not outs:
            raise sdcpp.EngineError("Le raffinage n'a produit aucune image.")
        return outs[0]

    # Grande cible -> tuiles.
    t = max(256, (tile // 16) * 16)
    step = max(16, t - overlap)
    ys = list(range(0, max(1, th), step))
    xs = list(range(0, max(1, tw), step))
    total = len(ys) * len(xs)
    if log:
        log(f"Raffinage par tuiles via « {m.name} » : {total} tuiles de {t}px "
            f"(le modèle se recharge à chaque tuile — c'est long).")

    acc = np.zeros((th, tw, 3), np.float32)
    wsum = np.zeros((th, tw, 1), np.float32)
    n = 0
    for y in ys:
        for x in xs:
            y2, x2 = min(y + t, th), min(x + t, tw)
            y1, x1 = max(0, y2 - t), max(0, x2 - t)
            tile_img = base.crop((x1, y1, x2, y2))
            tp = settings.TMP_DIR / "creative_tile.png"
            tile_img.save(tp)
            n += 1
            if log:
                log(f"  tuile {n}/{total}…")
            outs = generate(width=x2 - x1, height=y2 - y1, init_image=tp, **common)
            if not outs:
                raise sdcpp.EngineError("Une tuile n'a produit aucune image.")
            res = Image.open(outs[0]).convert("RGB").resize((x2 - x1, y2 - y1))
            arr = np.asarray(res, np.float32)
            mask = _feather_mask(y2 - y1, x2 - x1, overlap)
            acc[y1:y2, x1:x2] += arr * mask
            wsum[y1:y2, x1:x2] += mask

    final = (acc / np.clip(wsum, 1e-6, None)).clip(0, 255).astype("uint8")
    out_path = sdcpp.unique_output("creative")
    Image.fromarray(final).save(out_path)
    if log:
        log(f"Image finale : {out_path}")
    return out_path

