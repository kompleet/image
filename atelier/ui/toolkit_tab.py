"""Onglet Toolkit : profondeur, normales, suppression d'arrière-plan."""
from __future__ import annotations

import gradio as gr

from ..engine import tools


def _installer_block(title: str, note: str, stream_fn, installed: bool):
    """Accordéon d'installation 1 clic commun aux outils."""
    with gr.Accordion(f"⚙️ Installer {title} (en 1 clic)", open=not installed):
        gr.Markdown(note)
        log = gr.Textbox(label="Journal d'installation", lines=10,
                         autoscroll=True, elem_classes="log-box")
        btn = gr.Button(f"⬇️ Installer {title}")

        def _install():
            for msg in stream_fn():
                yield msg

        btn.click(_install, outputs=[log])


def build_toolkit_tab(tab_id="toolkit"):
    with gr.Tab("🧰 Toolkit", id=tab_id):
        gr.Markdown(
            "### Outils utilitaires\n"
            "Cartes de **profondeur** et de **normales** (utilisables comme image "
            "de contrôle ControlNet), et **suppression d'arrière-plan** "
            "(PNG transparent).")

        with gr.Tabs():
            # ---------- Profondeur & Normales ----------
            with gr.Tab("🌐 Profondeur / Normales"):
                gr.Markdown(
                    "*Depth Anything V2* — **profondeur** (clair = proche) et "
                    "**normales** (relief, dérivé de la profondeur). Téléchargez "
                    "le résultat pour l'utiliser comme contrôle ControlNet.")
                _installer_block(
                    "Depth Anything V2",
                    "Repose sur PyTorch + transformers (~100 Mo de modèle). "
                    "Couvre profondeur **et** normales. Aucune commande à taper.",
                    tools.install_depth_stream, tools.depth_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        d_image = gr.Image(label="Image source", type="pil",
                                           height=420)
                        d_strength = gr.Slider(
                            0.5, 6.0, value=2.0, step=0.5,
                            label="Intensité du relief (normales)")
                        with gr.Row():
                            d_depth = gr.Button("Profondeur", variant="primary")
                            d_normal = gr.Button("Normales", variant="primary")
                    with gr.Column(scale=4):
                        d_result = gr.Image(label="Résultat", height=520,
                                            format="png", show_download_button=True)
                        d_log = gr.Textbox(label="Journal", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def do_depth(img, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error("Fournissez une image.")
                    logs: list[str] = []
                    progress(0.1, desc="Profondeur…")
                    try:
                        out = tools.depth_map(img, log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERREUR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Terminé")
                    return str(out), "\n".join(logs)

                def do_normal(img, strength, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error("Fournissez une image.")
                    logs: list[str] = []
                    progress(0.1, desc="Normales…")
                    try:
                        out = tools.normal_map(img, strength=float(strength),
                                               log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERREUR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Terminé")
                    return str(out), "\n".join(logs)

                d_depth.click(do_depth, inputs=[d_image],
                              outputs=[d_result, d_log])
                d_normal.click(do_normal, inputs=[d_image, d_strength],
                               outputs=[d_result, d_log])

            # ---------- Suppression d'arrière-plan ----------
            with gr.Tab("✂️ Sans arrière-plan"):
                gr.Markdown(
                    "*RMBG-1.4* — détoure le sujet et renvoie un **PNG "
                    "transparent**.  \n"
                    "⚠️ Modèle sous licence **non commerciale** (BRIA RMBG-1.4).")
                _installer_block(
                    "RMBG-1.4",
                    "Repose sur PyTorch + transformers (~176 Mo de modèle). "
                    "Aucune commande à taper.",
                    tools.install_bg_stream, tools.bg_is_installed())

                with gr.Row():
                    with gr.Column(scale=3):
                        b_image = gr.Image(label="Image source", type="pil",
                                           height=420)
                        b_run = gr.Button("✂️ Détourer", variant="primary",
                                          size="lg")
                    with gr.Column(scale=4):
                        b_result = gr.Image(label="Sujet détouré (PNG transparent)",
                                            height=520, format="png",
                                            show_download_button=True,
                                            image_mode="RGBA")
                        b_log = gr.Textbox(label="Journal", lines=8,
                                           autoscroll=True, elem_classes="log-box")

                def do_bg(img, progress=gr.Progress()):
                    if img is None:
                        raise gr.Error("Fournissez une image.")
                    logs: list[str] = []
                    progress(0.1, desc="Détourage…")
                    try:
                        out = tools.bg_remove(img, log=logs.append)
                    except Exception as exc:  # noqa: BLE001
                        logs.append(f"\n[ERREUR] {exc}")
                        return None, "\n".join(logs)
                    progress(1.0, desc="Terminé")
                    return str(out), "\n".join(logs)

                b_run.click(do_bg, inputs=[b_image], outputs=[b_result, b_log])
