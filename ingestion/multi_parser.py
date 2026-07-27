import fitz  # PyMuPDF
import os
import logging
from collections import Counter
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def parse_file(file_path: str, filename: str) -> List[Dict[str, Any]]:
    ext = os.path.splitext(filename)[1].lower()
    if ext == ".pdf":
        return _parse_pdf(file_path, filename)
    elif ext in [".docx", ".doc"]:
        return _parse_docx(file_path, filename)
    else:
        return _parse_text(file_path, filename)


def _parse_pdf(file_path: str, filename: str) -> List[Dict[str, Any]]:
    doc = fitz.open(file_path)
    pages = []
    all_doc_sizes = []

    # Pass 1: Character-weighted font size collection (weights by char length to prevent
    # footnote markers / page numbers from biasing the mode)
    for page in doc:
        for b in page.get_text("dict").get("blocks", []):
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    txt = span.get("text", "").strip()
                    if txt and len(txt) >= 3:  # skip very short spans (page numbers, markers)
                        sz = round(span.get("size", 10.0), 1)
                        all_doc_sizes.extend([sz] * len(txt))

    # Resolve modal size; fall back to median if top-2 sizes are too close (no clear body mode)
    if all_doc_sizes:
        size_counts = Counter(all_doc_sizes)
        top_two = size_counts.most_common(2)
        if len(top_two) > 1 and top_two[0][1] < top_two[1][1] * 3:
            # No dominant mode — use sorted median as stable baseline
            sorted_sizes = sorted(all_doc_sizes)
            modal_size = sorted_sizes[len(sorted_sizes) // 2]
        else:
            modal_size = top_two[0][0]
    else:
        modal_size = 10.0

    # Pass 2: Extract text blocks, tagging outlier-size lines as headings
    for i, page in enumerate(doc):
        blocks = page.get_text("dict").get("blocks", [])
        lines = []
        for b in blocks:
            block_text = ""
            for line in b.get("lines", []):
                line_spans = line.get("spans", [])
                if not line_spans:
                    continue
                line_str = "".join(s.get("text", "") for s in line_spans).strip()
                max_span_size = max((s.get("size", 10.0) for s in line_spans), default=10.0)

                if line_str:
                    if max_span_size > (modal_size * 1.15) and len(line_str.split()) < 14 and not line_str.startswith("#"):
                        block_text += f"\n# {line_str}\n"
                    else:
                        block_text += f" {line_str}"

            if block_text.strip():
                lines.append(block_text.strip())

        full_page_text = "\n\n".join(lines)
        if full_page_text.strip():
            pages.append({
                "content": full_page_text.strip(),
                "metadata": {"source": filename, "page": i + 1, "type": "pdf"}
            })
    return pages


def _parse_docx(file_path: str, filename: str) -> List[Dict[str, Any]]:
    # 1. Try python-docx if installed
    try:
        import docx
        doc = docx.Document(file_path)
        elements = []
        body_elements = list(doc.element.body)
        consumed_indices: set = set()

        for idx, element in enumerate(body_elements):
            if idx in consumed_indices:
                continue

            tag_name = element.tag.split("}")[-1] if "}" in element.tag else element.tag

            if tag_name == "p":
                p = docx.text.paragraph.Paragraph(element, doc)
                txt = p.text.strip()
                if not txt:
                    continue
                style_name = p.style.name.lower() if p.style and p.style.name else ""
                if "heading" in style_name and not txt.startswith("#"):
                    elements.append(f"# {txt}")
                else:
                    elements.append(txt)

            elif tag_name == "tbl":
                table = docx.table.Table(element, doc)

                # Look for preceding caption
                caption = ""
                if idx > 0 and body_elements[idx - 1].tag.endswith("p"):
                    prev_txt = docx.text.paragraph.Paragraph(body_elements[idx - 1], doc).text.strip()
                    if any(prev_txt.lower().startswith(prefix) for prefix in ["table", "tab.", "figure", "fig."]):
                        caption = prev_txt

                # Look for trailing caption and mark it consumed so it isn't emitted again
                if not caption and idx + 1 < len(body_elements) and body_elements[idx + 1].tag.endswith("p"):
                    next_txt = docx.text.paragraph.Paragraph(body_elements[idx + 1], doc).text.strip()
                    if any(next_txt.lower().startswith(prefix) for prefix in ["table", "tab.", "figure", "fig."]):
                        caption = next_txt
                        consumed_indices.add(idx + 1)

                table_rows = []
                for row_idx, row in enumerate(table.rows):
                    # Use object identity to deduplicate merged (rowspan/colspan) cells
                    seen_cell_ids: set = set()
                    row_cells = []
                    for cell in row.cells:
                        if id(cell) not in seen_cell_ids:
                            seen_cell_ids.add(id(cell))
                            row_cells.append(cell.text.strip().replace("\n", " "))
                    if any(row_cells):
                        table_rows.append("| " + " | ".join(row_cells) + " |")
                        if row_idx == 0:
                            table_rows.append("| " + " | ".join(["---"] * len(row_cells)) + " |")

                if table_rows:
                    table_md = "\n".join(table_rows)
                    if caption:
                        table_md = f"**{caption}**\n{table_md}"
                    elements.append(table_md)

        full_text = "\n\n".join(elements)
        if full_text.strip():
            return [{"content": full_text.strip(), "metadata": {"source": filename, "page": 1, "type": "docx"}}]
    except Exception:
        pass

    # 2. Native stdlib zipfile + XML fallback (zero dependencies required)
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        with zipfile.ZipFile(file_path, "r") as z:
            xml_content = z.read("word/document.xml")
        root = ET.fromstring(xml_content)
        texts = [node.text for node in root.iter() if node.tag.endswith("}t") and node.text]
        full_text = " ".join(texts).strip()
        if full_text:
            return [{"content": full_text, "metadata": {"source": filename, "page": 1, "type": "docx"}}]
    except Exception as e:
        logger.warning(f"Stdlib Word XML fallback failed for '{filename}': {e}")

    return _parse_text(file_path, filename)


def _parse_text(file_path: str, filename: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read().strip()
    if not content:
        return []
    return [{"content": content, "metadata": {"source": filename, "page": 1, "type": "text"}}]


