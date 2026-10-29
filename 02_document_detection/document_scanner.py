"""
document_scanner.py
--------------------
Low-level extraction: given a PDF / DOCX / PPTX file, pull out
(a) every embedded raster image, as PIL Images, and
(b) document metadata (author/tool/dates) for forensic evidence.

Supported formats: .pdf, .docx, .pptx
(.doc/.ppt legacy binary formats are NOT supported — convert them to
 .docx/.pptx first, e.g. with LibreOffice: `soffice --headless --convert-to docx file.doc`)
"""

from __future__ import annotations
import io
import os
import zipfile
import xml.etree.ElementTree as ET
from typing import List, Tuple

from PIL import Image

try:
    import pymupdf as fitz
except ImportError:  # pragma: no cover
    import fitz  # older package name


# ---------------------------------------------------------------------- #
# PDF
# ---------------------------------------------------------------------- #
def extract_images_from_pdf(path: str) -> List[Tuple[str, Image.Image]]:
    """Returns list of (source_label, PIL.Image) for every embedded image."""
    images = []
    doc = fitz.open(path)
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            try:
                base = doc.extract_image(xref)
                pil_img = Image.open(io.BytesIO(base["image"]))
                label = f"page{page_index + 1}_img{img_index + 1}.{base.get('ext', 'png')}"
                images.append((label, pil_img))
            except Exception:
                continue

    # Fallback: if a PDF has zero extractable embedded images (e.g. the
    # "photo" is actually the whole page background / a scan), rasterize
    # each page instead so visual content still gets analyzed.
    if not images:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(dpi=150)
            pil_img = Image.open(io.BytesIO(pix.tobytes("png")))
            images.append((f"page{page_index + 1}_rendered.png", pil_img))

    doc.close()
    return images


def extract_metadata_pdf(path: str) -> dict:
    doc = fitz.open(path)
    meta = dict(doc.metadata) if doc.metadata else {}
    meta["page_count"] = len(doc)
    doc.close()
    return meta


# ---------------------------------------------------------------------- #
# DOCX / PPTX  (both are OOXML = a zip archive of XML + media files)
# ---------------------------------------------------------------------- #
_OOXML_MEDIA_DIRS = {".docx": "word/media/", ".pptx": "ppt/media/"}


def extract_images_from_office(path: str) -> List[Tuple[str, Image.Image]]:
    ext = os.path.splitext(path)[1].lower()
    media_dir = _OOXML_MEDIA_DIRS.get(ext)
    if media_dir is None:
        raise ValueError(f"Unsupported office format: {ext}")

    images = []
    with zipfile.ZipFile(path) as z:
        media_files = [n for n in z.namelist() if n.startswith(media_dir)]
        for name in media_files:
            try:
                data = z.read(name)
                pil_img = Image.open(io.BytesIO(data))
                pil_img.load()  # force decode now, fail fast on non-image (e.g. .emf/.wmf)
                images.append((os.path.basename(name), pil_img))
            except Exception:
                continue  # skip vector/unsupported formats (emf, wmf, etc.)
    return images


def extract_metadata_office(path: str) -> dict:
    """Reads docProps/core.xml and docProps/app.xml straight from the zip —
    no python-docx/python-pptx dependency needed, works for both formats."""
    ns_core = {
        "cp": "http://schemas.openxmlformats.org/package/2006/metadata/core-properties",
        "dc": "http://purl.org/dc/elements/1.1/",
        "dcterms": "http://purl.org/dc/terms/",
    }
    meta = {}
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if "docProps/core.xml" in names:
            root = ET.fromstring(z.read("docProps/core.xml"))
            meta["creator"] = _find_text(root, "dc:creator", ns_core)
            meta["last_modified_by"] = _find_text(root, "cp:lastModifiedBy", ns_core)
            meta["created"] = _find_text(root, "dcterms:created", ns_core)
            meta["modified"] = _find_text(root, "dcterms:modified", ns_core)
            meta["revision"] = _find_text(root, "cp:revision", ns_core)
        if "docProps/app.xml" in names:
            root = ET.fromstring(z.read("docProps/app.xml"))
            app_tag = root.find(
                "{http://schemas.openxmlformats.org/officeDocument/2006/extended-properties}Application"
            )
            meta["application"] = app_tag.text if app_tag is not None else None
    return meta


def _find_text(root, tag, ns):
    el = root.find(tag, ns)
    return el.text if el is not None else None


# ---------------------------------------------------------------------- #
# Unified entry point
# ---------------------------------------------------------------------- #
def extract(path: str) -> Tuple[List[Tuple[str, Image.Image]], dict]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return extract_images_from_pdf(path), extract_metadata_pdf(path)
    elif ext in (".docx", ".pptx"):
        return extract_images_from_office(path), extract_metadata_office(path)
    else:
        raise ValueError(
            f"Unsupported document type '{ext}'. Supported: .pdf, .docx, .pptx "
            "(convert legacy .doc/.ppt to these first)."
        )
