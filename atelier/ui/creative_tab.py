"""Onglet Upscale : PiD (rapide ~2K) ou Flux.2 Klein tuilé (créatif), 100% GPU
via stable-diffusion.cpp. Plus de dépendance SDXL/diffusers.
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
            "### Upscale — 100% GPU (sd.cpp, offload RAM)\n"
            "**⚡ PiD** : décodeur NVIDIA, agrandit vers ~2K en 4 pas (rapide).  \n"
            "**🟣 Flux Klein (tuilé)** : ré-invente le détail par tuiles via "
            "**Flux.2 Klein** (façon Magnific) — facteur libre, plus lent (le "
            "modèle se recharge à chaque tuile). Aucun PyTorch.")

        method = gr.Radio(
            [("⚡ Rapide — PiD (~2K)", "pid"),
             ("🟣 Créatif — Flux Klein (tuilé)", "klein")],
            value="pid", label="Méthode")

        with gr.Accordion("⚙️ Installer PiD (en 1 clic)",
                          open=not registry.pid_ready()):
            gr.Markdown(
                "Via sd.cpp (GPU natif, **pas de PyTorch**). Télécharge le "
                "décodeur PiD + l'encodeur Gemma-2-2B + la VAE FLUX.1.")
            pid_log = gr.Textbox(label="Journal d'installation", lines=8,
                                 autoscroll=True, elem_classes="log-box")
            pid_btn = gr.Button("⬇️ Installer PiD")

            def _install_pid():
                lines: list[str] = []
                for msg in downloader.download_pid(log=lines.append):
                    lines.append(msg)
                    yield "\n".join(lines)

            pid_btn.click(_install_pid, outputs=[pid_log])

        gr.Markdown(
            "<small>🟣 *Flux Klein* utilise le modèle **Flux.2 Klein 9B** : "
            "téléchargez-le simplement dans l'onglet **Catalogue de modèles** (rien de "
            "plus à installer).</small>")

        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image à agrandir", type="pil", height=380)
                prompt = gr.Textbox(
                    label="Prompt (optionnel — guide le détail)", lines=2,
                    placeholder="highly detailed, sharp focus, intricate details")
                with gr.Group(visible=False) as klein_opts:
                    scale = gr.Radio([2, 4, 8], value=2,
                                     label="Facteur (×8 = très long)")
                    k_steps = gr.Slider(2, 12, value=4, step=1,
                                        label="Pas par tuile (↑ = + de détail/dérive)")
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
            return gr.update(visible=(m == "klein"))

        method.change(_toggle, inputs=[method], outputs=[klein_opts])

        def do_upscale(method, image, prompt, scale, k_steps,
                       progress=gr.Progress()):
            if image is None:
                raise gr.Error("Fournissez une image.")
            logs: list[str] = []
            try:
                if method == "pid":
                    progress(0.1, desc="PiD (GPU)…")
                    out = gen_engine.pid_upscale(image, prompt=prompt or "",
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

        evt = run.click(do_upscale,
                        inputs=[method, image, prompt, scale, k_steps],
                        outputs=[result, logbox])

        stop.click(lambda: gen_engine.cancel(), outputs=None, cancels=[evt])

    return image
