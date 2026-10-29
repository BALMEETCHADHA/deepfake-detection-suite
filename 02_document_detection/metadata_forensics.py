"""
metadata_forensics.py
----------------------
Heuristic evidence from document metadata alone (no image analysis).
These are supporting signals, not proof by themselves — a document edited
in Photoshop isn't necessarily fake, but an inconsistent authoring trail is
worth surfacing to the user, same philosophy as explain.py in Step 1.
"""

from __future__ import annotations
from typing import List, Tuple

_SUSPICIOUS_TOOL_KEYWORDS = [
    "stable diffusion", "midjourney", "dall-e", "dalle", "runway",
    "generative", "ai-generated", "synthesia", "deepfake",
]


def build_pdf_metadata_evidence(meta: dict) -> Tuple[List[str], float]:
    """Returns (evidence_lines, suspicion_score 0..1)."""
    evidence = []
    suspicion = 0.0

    producer = (meta.get("producer") or "").lower()
    creator = (meta.get("creator") or "").lower()
    combined = f"{producer} {creator}"

    hit = next((kw for kw in _SUSPICIOUS_TOOL_KEYWORDS if kw in combined), None)
    if hit:
        evidence.append(
            f"PDF producer/creator metadata references a generative-AI tool ('{hit}')."
        )
        suspicion += 0.5
    else:
        evidence.append(
            f"PDF producer='{meta.get('producer', 'unknown')}', "
            f"creator='{meta.get('creator', 'unknown')}' — no generative-AI tool signature found."
        )

    creation = meta.get("creationDate")
    mod = meta.get("modDate")
    if creation and mod and creation != mod:
        evidence.append(
            f"Creation date ({creation}) differs from modification date ({mod}) — "
            "document was edited after creation (not inherently suspicious, but worth noting)."
        )

    return evidence, min(suspicion, 1.0)


def build_office_metadata_evidence(meta: dict) -> Tuple[List[str], float]:
    evidence = []
    suspicion = 0.0

    creator = (meta.get("creator") or "").lower()
    last_mod_by = (meta.get("last_modified_by") or "").lower()
    application = (meta.get("application") or "").lower()
    combined = f"{creator} {last_mod_by} {application}"

    hit = next((kw for kw in _SUSPICIOUS_TOOL_KEYWORDS if kw in combined), None)
    if hit:
        evidence.append(f"Office metadata references a generative-AI tool ('{hit}').")
        suspicion += 0.5
    else:
        evidence.append(
            f"Author='{meta.get('creator', 'unknown')}', "
            f"last modified by='{meta.get('last_modified_by', 'unknown')}', "
            f"application='{meta.get('application', 'unknown')}' — no generative-AI tool signature found."
        )

    if creator and last_mod_by and creator != last_mod_by:
        evidence.append(
            f"Document author ('{meta.get('creator')}') differs from the last person who "
            f"modified it ('{meta.get('last_modified_by')}') — could be normal collaboration, "
            "could indicate later tampering."
        )
        suspicion += 0.1

    return evidence, min(suspicion, 1.0)
