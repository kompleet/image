#!/usr/bin/env python3
"""Upscale créatif (façon Magnific) : SDXL + ControlNet Tile via diffusers.

Modèle RÉSIDENT sur le GPU (rapide, pas de rechargement par tuile), attention
SDPA (rapide), VAE tiling pour la mémoire. Pré-agrandit (Lanczos + affinage) puis
raffine PAR TUILES ancrées par le ControlNet Tile -> tuiles cohérentes, fondu
cosinus. Lancé en sous-process pour ne pas verrouiller les DLL torch dans Gradio.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass


def _feather(h, w, fade, np):
    fy = np.ones(h, np.float32)
    fx = np.ones(w, np.float32)
    f = max(1, min(fade, h // 2, w // 2))
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0.0, np.pi, f, dtype=np.float32))
    fy[:f] = ramp; fy[-f:] = ramp[::-1]
    fx[:f] = ramp; fx[-f:] = ramp[::-1]
    return (fy[:, None] * fx[None, :])[:, :, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--controlnet", required=True)
    ap.add_argument("--vae", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--creativity", type=float, default=0.35)   # strength img2img
    ap.add_argument("--cn-scale", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--cfg", type=float, default=5.0)
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--overlap", type=int, default=160)
    ap.add_argument("--preview-path", default="")   # aperçu temps réel (par tuile)
    args = ap.parse_args()

    import numpy as np
    import torch
    from PIL import Image, ImageFilter
    try:
        from diffusers import (AutoencoderKL, ControlNetModel,
                               DPMSolverMultistepScheduler,
                               StableDiffusionXLControlNetImg2ImgPipeline)
    except ImportError:
        sys.exit("diffusers manquant. Réinstallez l'outil depuis l'onglet Upscale.")

    cuda = torch.cuda.is_available()
    dev = torch.cuda.get_device_name(0) if cuda else "CPU"
    print(f"[upscale] torch {torch.__version__} | CUDA: {cuda} | device: {dev}",
          flush=True)
    if not cuda:
        sys.exit("PyTorch ne voit PAS le GPU (build CPU). Réinstallez l'outil "
                 "(bouton « Installer ») pour obtenir un PyTorch CUDA.")
    dtype = torch.float16

    print("[upscale] chargement SDXL + ControlNet Tile…", flush=True)
    controlnet = ControlNetModel.from_pretrained(args.controlnet, torch_dtype=dtype)
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype)
    pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_single_file(
        args.base_model, controlnet=controlnet, vae=vae, torch_dtype=dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)
    pipe.set_progress_bar_config(disable=True)
    # RÉSIDENT GPU (rapide), attention SDPA (défaut), VAE tiling (mémoire).
    pipe.to("cuda")
    try:
        pipe.enable_vae_tiling()
    except Exception:  # noqa: BLE001
        pass
    total = torch.cuda.get_device_properties(0).total_memory
    print(f"[upscale] {total/1e9:.0f} Go VRAM -> modèle résident GPU (SDPA).",
          flush=True)

    prompt = args.prompt or ("high quality, sharp focus, fine intricate details, "
                             "crisp textures, photorealistic, 8k, masterpiece")
    negative = ("blurry, soft, low quality, jpeg artifacts, oversaturated, "
                "deformed, oversmooth, plastic")

    img = Image.open(args.input).convert("RGB")
    tw = max(512, int(round(img.width * args.scale / 8)) * 8)
    th = max(512, int(round(img.height * args.scale / 8)) * 8)
    cap = 4096
    if max(tw, th) > cap:
        r = cap / max(tw, th)
        tw = max(512, int(round(tw * r / 8)) * 8)
        th = max(512, int(round(th * r / 8)) * 8)
        print(f"[upscale] cible plafonnée à {tw}x{th} (max {cap}px).", flush=True)
    print(f"[upscale] {img.width}x{img.height} -> {tw}x{th}…", flush=True)
    base = img.resize((tw, th), Image.LANCZOS).filter(
        ImageFilter.UnsharpMask(radius=2, percent=90, threshold=2))

    def _sharpen(im):
        return im.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=2))

    common = dict(prompt=prompt, negative_prompt=negative,
                  num_inference_steps=int(args.steps), strength=float(args.creativity),
                  guidance_scale=float(args.cfg),
                  controlnet_conditioning_scale=float(args.cn_scale))

    def refine(tile_img):
        try:
            return pipe(image=tile_img, control_image=tile_img, **common).images[0]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            sys.exit("VRAM insuffisante. Baissez le facteur (×2) ou la résolution.")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / (Path(args.input).stem + "_upscaled.png")

    if max(tw, th) <= args.tile:
        print("[upscale] raffinage en 1 passe…", flush=True)
        _sharpen(refine(base).resize((tw, th))).save(dest)
        print(f"[upscale] image écrite : {dest}", flush=True)
        return

    t = max(512, (args.tile // 8) * 8)
    step = max(64, t - args.overlap)
    ys = list(range(0, max(1, th), step))
    xs = list(range(0, max(1, tw), step))
    total_tiles = len(ys) * len(xs)
    print(f"[upscale] {total_tiles} tuiles de {t}px…", flush=True)
    acc = np.zeros((th, tw, 3), np.float32)
    wsum = np.zeros((th, tw, 1), np.float32)

    def _write_preview():
        if not args.preview_path:
            return
        cur = (acc / np.clip(wsum, 1e-6, None)).clip(0, 255).astype("uint8")
        prev = Image.fromarray(cur)
        if max(prev.size) > 1280:  # aperçu léger pour rester fluide
            r = 1280 / max(prev.size)
            prev = prev.resize((int(prev.width * r), int(prev.height * r)))
        try:
            prev.save(args.preview_path)
        except OSError:
            pass

    n = 0
    for y in ys:
        for x in xs:
            y2, x2 = min(y + t, th), min(x + t, tw)
            y1, x1 = max(0, y2 - t), max(0, x2 - t)
            n += 1
            print(f"[upscale]   tuile {n}/{total_tiles}…", flush=True)
            res = refine(base.crop((x1, y1, x2, y2))).resize((x2 - x1, y2 - y1))
            arr = np.asarray(res, np.float32)
            mask = _feather(y2 - y1, x2 - x1, args.overlap, np)
            acc[y1:y2, x1:x2] += arr * mask
            wsum[y1:y2, x1:x2] += mask
            _write_preview()   # aperçu : image assemblée jusqu'ici
    final = (acc / np.clip(wsum, 1e-6, None)).clip(0, 255).astype("uint8")
    _sharpen(Image.fromarray(final)).save(dest)
    print(f"[upscale] image écrite : {dest}", flush=True)


if __name__ == "__main__":
    main()
