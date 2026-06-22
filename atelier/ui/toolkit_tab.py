"""Onglet Toolkit : outils utilitaires (carte de profondeur, …)."""
from __future__ import annotations

import gradio as gr

from ..engine import tools


def build_toolkit_tab(tab_id="toolkit"):
    with gr.Tab("🧰 Toolkit", id=tab_id):
        gr.Markdown(
            "### Outils\n"
            "**Carte de profondeur (depth map)** — estime la profondeur d'une "
            "image avec *Depth Anything V2*. Le résultat (clair = proche, sombre "
            "= loin) s'utilise directement comme **image de contrôle ControlNet** "
            "(téléchargez-le puis chargez-le dans l'accordéon ControlNet d'un "
            "modèle).")

        with gr.Accordion("⚙️ Installer l'outil de profondeur (en 1 clic)",
                          open=not tools.depth_is_installed()):
            gr.Markdown(
                "Repose sur PyTorch + transformers (~100 Mo de modèle). Cliquez "
                "pour installer — **aucune commande à taper**. Le premier "
                "téléchargement peut prendre un moment.")
            install_log = gr.Textbox(label="Journal d'installation", lines=10,
                                     autoscroll=True, elem_classes="log-box")
            install_btn = gr.Button("⬇️ Installer Depth Anything V2")

            def _install():
                for msg in tools.install_depth_stream():
                    yield msg

            install_btn.click(_install, outputs=[install_log])

        with gr.Row():
            with gr.Column(scale=3):
                image = gr.Image(label="Image source", type="pil", height=420)
                run = gr.Button("🌐 Générer la carte de profondeur",
                                variant="primary", size="lg")
            with gr.Column(scale=4):
                result = gr.Image(label="Carte de profondeur", height=560,
                                  format="png", show_download_button=True)
                logbox = gr.Textbox(label="Journal", lines=10, autoscroll=True,
                                    elem_classes="log-box")

        def do_depth(img, progress=gr.Progress()):
            if img is None:
                raise gr.Error("Fournissez une image.")
            logs: list[str] = []
            progress(0.1, desc="Estimation de profondeur…")
            try:
                out = tools.depth_map(img, log=logs.append)
            except Exception as exc:  # noqa: BLE001
                logs.append(f"\n[ERREUR] {exc}")
                return None, "\n".join(logs)
            progress(1.0, desc="Terminé")
            return str(out), "\n".join(logs)

        run.click(do_depth, inputs=[image], outputs=[result, logbox])
