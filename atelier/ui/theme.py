"""Thème et habillage CSS d'Atelier (clair, moderne, orienté artiste)."""
from __future__ import annotations

import gradio as gr


def theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.violet,
        secondary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_lg,
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#f6f7fb",
        block_background_fill="#ffffff",
        block_border_width="1px",
        block_title_text_weight="600",
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_400",
    )


CSS = """
.gradio-container { max-width: 1400px !important; margin: auto; }
#atelier-header { text-align: left; padding: 4px 0 2px 0; }
#atelier-header h1 { font-size: 1.7rem; margin: 0; letter-spacing: .5px;
                     color: #5b21b6; }
#atelier-header .sub { color: #6b7280; font-size: .85rem; margin-top: 2px; }
.model-card { border:1px solid #e5e7eb; border-radius:14px; padding:14px 16px;
              background:#ffffff; margin-bottom:10px;
              box-shadow:0 1px 2px rgba(16,24,40,.04); }
.model-card h3 { margin:0 0 4px 0; }
.tag { display:inline-block; background:#f1ecfb; color:#6d28d9; border-radius:999px;
       padding:2px 10px; font-size:.72rem; margin-right:6px; }
.status-ok { color:#16a34a; font-weight:600; }
.status-missing { color:#d97706; font-weight:600; }
.log-box textarea { font-family: ui-monospace, monospace; font-size:.8rem; }
footer { display:none !important; }
"""
