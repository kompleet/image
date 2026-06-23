"""Onglet Upscale créatif (façon Magnific) : pré-agrandissement + raffinage
img2img par tuiles via Flux.2 Klein. Réutilise le moteur de génération.
"""
from __future__ import annotations

import gradio as gr

from .. import registry, settings
from ..engine import generate as gen_engine

REFINER_ID = "flux2-klein-9b"


def build_creative_tab(tab_id="creative"):
    """Construit l'onglet Upscale créatif. Renvoie le composant Image d'entrée
    (pour que les onglets de génération puissent y envoyer une image)."""
    with gr.Tab("✨ Upscale créatif", id=tab_id):
        gr.Markdown(
            "### Upscale créatif (façon Magnific / Topaz Wonder)\n"
            "Pré-agrandit l'image, puis **ré-invente le détail** via **Flux.2 "
            "Klein** (img2img **par tuiles**, recollées avec fondu). Le curseur "
            "*Créativité* dose le détail ajouté.\n\n"
            "> ⚠️ Le modèle se recharge à chaque tuile : **×2 = 4 tuiles** (rapide), "
            "**×4 = 16 tuiles** (long), **×8 = très long**.")
        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image à agrandir", type="pil", height=380)
                scale = gr.Radio([2, 4, 8], value=2,
                                 label="Facteur (×8 = très long, beaucoup de tuiles)")
                creativity = gr.Slider(0.1, 0.6, value=0.3, step=0.05,
                                       label="Créativité (détail ajouté / dérive)")
                prompt = gr.Textbox(
                    label="Prompt (optionnel — guide le détail)", lines=2,
                    placeholder="highly detailed, sharp focus, intricate details")
                run = gr.Button("✨ Upscaler", variant="primary", size="lg")
            with gr.Column(scale=4):
                result = gr.Image(label="Résultat", height=620, format="png",
                                  show_download_button=True)
                logbox = gr.Textbox(label="Journal", lines=12, autoscroll=True,
                                    elem_classes="log-box")

        def do_creative(image, scale, creativity, prompt, progress=gr.Progress()):
            if image is None:
                raise gr.Error("Fournissez une image.")
            m = registry.get_base_model(REFINER_ID, settings.load_prefs())
            if m is None or not registry.model_is_ready(m):
                raise gr.Error("Flux.2 Klein n'est pas téléchargé "
                               "(onglet Bibliothèque).")
            logs: list[str] = []
            progress(0.05, desc="Upscale créatif…")
            try:
                out = gen_engine.creative_upscale(
                    model_id=REFINER_ID, image=image, scale=int(scale),
                    prompt=prompt or "", creativity=float(creativity),
                    log=logs.append)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return None, "\n".join(logs)
            progress(1.0, desc="Terminé")
            return str(out), "\n".join(logs)

        run.click(do_creative, inputs=[image, scale, creativity, prompt],
                  outputs=[result, logbox])

    return image
