"""Constructeur de prompt structuré pour Ideogram 4.

Reproduit fidèlement le schéma JSON sur lequel Ideogram 4 est entraîné
(cf. doc officielle « prompting »). L'ORDRE DES CLÉS est strict et vérifié par
le CaptionVerifier du modèle :

  high_level_description
  style_description :
     photo     -> aesthetics, lighting, photo, medium, [color_palette]
     art_style -> aesthetics, lighting, medium, art_style, [color_palette]
  compositional_deconstruction : background, elements[]
     obj  -> type, [bbox], desc, [color_palette]
     text -> type, [bbox], text, desc, [color_palette]

bbox : [y_min, x_min, y_max, x_max] en coordonnées normalisées 0–1000 (optionnel).
Sérialisation compacte (séparateurs sans espace, ensure_ascii=False), hex MAJ.
"""
from __future__ import annotations

import json
import re


def parse_colors(raw: str | None, limit: int = 16) -> list[str]:
    """Extrait des couleurs #RRGGBB (mises en MAJUSCULES) d'une chaîne libre."""
    if not raw:
        return []
    out = []
    for c in re.findall(r"#?[0-9a-fA-F]{6}", raw):
        c = c if c.startswith("#") else "#" + c
        out.append(c.upper())
    return out[:limit]


def _clamp1000(v) -> int:
    try:
        return max(0, min(1000, int(round(float(v)))))
    except (TypeError, ValueError):
        return 0


def build_prompt(
    mode: str,
    high_level: str,
    aesthetics: str,
    lighting: str,
    medium: str,
    style_or_photo: str,
    background: str,
    global_colors: str,
    element_rows: list[list],
) -> str:
    """Assemble le prompt JSON Ideogram 4. `mode` = 'photo' ou 'art_style'."""
    is_photo = (mode == "photo")

    # --- style_description (ordre des clés dépendant du type) --------------
    style: dict = {"aesthetics": aesthetics or "", "lighting": lighting or ""}
    if is_photo:
        style["photo"] = style_or_photo or ""
        style["medium"] = medium or "photograph"
    else:
        style["medium"] = medium or ""
        style["art_style"] = style_or_photo or ""
    palette = parse_colors(global_colors, 16)
    if palette:
        style["color_palette"] = palette

    # --- éléments ----------------------------------------------------------
    elements: list[dict] = []
    for row in element_rows or []:
        row = list(row) + [""] * (8 - len(row))
        etype, text, desc, x1, y1, x2, y2, colors = row[:8]
        etype = "text" if str(etype).strip().lower().startswith("t") else "obj"
        desc, text = str(desc or ""), str(text or "")
        if not desc.strip() and not text.strip():
            continue  # ligne vide

        el: dict = {"type": etype}
        bbox = [_clamp1000(y1), _clamp1000(x1), _clamp1000(y2), _clamp1000(x2)]
        if any(bbox):  # bbox optionnel : omis si tout à zéro
            el["bbox"] = bbox
        if etype == "text":
            el["text"] = text
            el["desc"] = desc
        else:
            el["desc"] = desc
        ecols = parse_colors(colors if isinstance(colors, str) else "", 5)
        if ecols:
            el["color_palette"] = ecols
        elements.append(el)

    caption = {
        "high_level_description": high_level or "",
        "style_description": style,
        "compositional_deconstruction": {
            "background": background or "",
            "elements": elements,
        },
    }
    # Encodage attendu par le modèle : compact + non-ASCII littéral conservé.
    return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)


# Exemple pré-rempli (colonnes : type, texte, description, x1, y1, x2, y2, couleurs)
EXAMPLE_ELEMENTS = [
    ["text", "FEDERALL", "titre en haut, lettres dorées", 100, 50, 900, 180, "#E7C84B"],
    ["obj", "", "chat roux pelucheux assis au centre", 300, 300, 700, 850, ""],
]
ELEMENT_HEADERS = ["type (obj/text)", "texte", "description",
                   "x1", "y1", "x2", "y2", "couleurs (#hex)"]
