"""Onglet Upscale : PiD (rapide ~2K, sd.cpp), SDXL+ControlNet Tile (créatif,
PyTorch, résident GPU) ou Flux.2 Klein tuilé (créatif, sd.cpp léger).
"""
from __future__ import annotations

import gradio as gr

from .. import downloader, registry, settings
from ..engine import generate as gen_engine
from ..engine import tools


def build_creative_tab(tab_id="creative"):
    """Construit l'onglet Upscale. Renvoie le composant Image d'entrée (pour que
    l'onglet de génération puisse y envoyer une image)."""
    with gr.Tab("✨ Upscale", id=tab_id):
        gr.Markdown(
            "### Upscale — 100% GPU\n"
            "**⚡ PiD** : décodeur NVIDIA (sd.cpp), ~2K en 4 pas, rapide.  \n"
            "**🎨 SDXL + ControlNet Tile** (PyTorch) : façon Magnific, modèle "
            "**résident** sur le GPU → rapide + tuiles cohérentes.  \n"
            "**🟣 Flux Klein (tuilé, sd.cpp)** : léger (rien à installer) mais "
            "lent (recharge par tuile).")

        method = gr.Radio(
            [("⚡ PiD (rapide)", "pid"),
             ("🎨 SDXL + ControlNet Tile (recommandé)", "sdxl"),
             ("🟣 Flux Klein (léger, lent)", "klein")],
            value="sdxl", label="Méthode")

        with gr.Accordion("⚙️ Installer SDXL + ControlNet Tile (en 1 clic)",
                          open=not tools.upscale_is_installed()):
            gr.Markdown(
                "PyTorch + diffusers (~9 Go : SDXL + ControlNet Tile + VAE). "
                "Modèle résident sur le GPU → rapide. Aucune commande à taper.")
            sdxl_log = gr.Textbox(label="Journal d'installation", lines=8,
                                  autoscroll=True, elem_classes="log-box")
            sdxl_btn = gr.Button("⬇️ Installer (SDXL + ControlNet Tile)")

            def _install_sdxl():
                for msg in tools.install_upscale_stream():
                    yield msg

            sdxl_btn.click(_install_sdxl, outputs=[sdxl_log])

        with gr.Accordion("⚙️ Installer PiD (en 1 clic)",
                          open=False):
            gr.Markdown("Via sd.cpp (pas de PyTorch). Décodeur PiD + Gemma-2-2B "
                        "+ VAE FLUX.1.")
            pid_log = gr.Textbox(label="Journal d'installation", lines=8,
                                 autoscroll=True, elem_classes="log-box")
            pid_btn = gr.Button("⬇️ Installer PiD")

            def _install_pid():
                lines: list[str] = []
                for msg in downloader.download_pid(log=lines.append):
                    lines.append(msg)
                    yield "\n".join(lines)

            pid_btn.click(_install_pid, outputs=[pid_log])

        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image à agrandir", type="pil", height=380)
                prompt = gr.Textbox(
                    label="Prompt (optionnel — guide le détail)", lines=2,
                    placeholder="highly detailed, sharp focus, intricate details")
                scale = gr.Radio([2, 4, 8], value=2,
                                 label="Facteur (×8 = très long)")
                with gr.Group() as sdxl_opts:
                    creativity = gr.Slider(0.15, 0.7, value=0.4, step=0.05,
                                           label="Créativité (détail inventé — ↑ = plus)")
                    cn_scale = gr.Slider(0.2, 1.0, value=0.5, step=0.05,
                                         label="Fidélité (↓ = plus de détail inventé)")
                with gr.Group(visible=False) as klein_opts:
                    k_steps = gr.Slider(2, 12, value=4, step=1,
                                        label="Pas par tuile (Klein)")
                with gr.Row():
                    run = gr.Button("✨ Upscaler", variant="primary",
                                    size="lg", scale=3)
                    stop = gr.Button("⏹️ Annuler", variant="stop", scale=1)
            with gr.Column(scale=4):
                result = gr.Image(label="Aperçu (résolution complète dans outputs/)",
                                  height=620, format="png", show_download_button=True)
                logbox = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                    elem_classes="log-box")

        def _toggle(m):
            return (gr.update(visible=(m != "pid")),     # scale
                    gr.update(visible=(m == "sdxl")),    # sdxl_opts
                    gr.update(visible=(m == "klein")))   # klein_opts

        method.change(_toggle, inputs=[method],
                      outputs=[scale, sdxl_opts, klein_opts])

        def do_upscale(method, image, prompt, scale, creativity, cn_scale, k_steps,
                       progress=gr.Progress()):
            if image is None:
                raise gr.Error("Fournissez une image.")
            logs: list[str] = []
            try:
                if method == "pid":
                    progress(0.1, desc="PiD (GPU)…")
                    out = gen_engine.pid_upscale(image, prompt=prompt or "",
                                                 log=logs.append)
                elif method == "sdxl":
                    progress(0.05, desc="Upscale SDXL…")
                    out = tools.creative_upscale(
                        image, scale=int(scale), prompt=prompt or "",
                        creativity=float(creativity), cn_scale=float(cn_scale),
                        log=logs.append)
                else:
                    m = registry.get_base_model("flux2-klein-9b",
                                                settings.load_prefs())
                    if m is None or not registry.model_is_ready(m):
                        raise gr.Error("Flux.2 Klein n'est pas téléchargé "
                                       "(onglet Catalogue de modèles).")
                    progress(0.05, desc="Upscale Klein tuilé…")
                    out = gen_engine.klein_tiled_upscale(
                        image, scale=int(scale), prompt=prompt or "",
                        steps=int(k_steps), log=logs.append)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return None, "\n".join(logs)
            progress(1.0, desc="Terminé")
            from PIL import Image as _PILImage
            try:
                im = _PILImage.open(out)
                logs.append(f"\n✅ Image pleine résolution ({im.width}x{im.height}) "
                            f"enregistrée : {out}")
                disp = im
                if max(im.size) > 1600:
                    r = 1600 / max(im.size)
                    disp = im.resize((max(1, int(im.width * r)),
                                      max(1, int(im.height * r))))
                return disp, "\n".join(logs)
            except Exception:  # noqa: BLE001
                return str(out), "\n".join(logs)

        evt = run.click(
            do_upscale,
            inputs=[method, image, prompt, scale, creativity, cn_scale, k_steps],
            outputs=[result, logbox])

        def _cancel():
            tools.cancel()        # sous-process SDXL/SAM
            gen_engine.cancel()   # process sd.cpp (PiD / Klein)

        stop.click(_cancel, outputs=None, cancels=[evt])

    return image
