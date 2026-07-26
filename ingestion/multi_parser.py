import fitz  # PyMuPDF
import os
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class MultiParser:
    """Multi-format document parser supporting PDF, Word (.docx), Markdown, and TXT files."""
    
    @staticmethod
    def parse_file(file_path: str, filename: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            return MultiParser._parse_pdf(file_path, filename)
        elif ext in [".docx", ".doc"]:
            return MultiParser._parse_docx(file_path, filename)
        else:
            return MultiParser._parse_text(file_path, filename)

    @staticmethod
    def _parse_pdf(file_path: str, filename: str) -> List[Dict[str, Any]]:
        doc = fitz.open(file_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({
                    "content": text,
                    "metadata": {
                        "source": filename,
                        "page": i + 1,
                        "type": "pdf"
                    }
                })
        return pages

    @staticmethod
    def _parse_docx(file_path: str, filename: str) -> List[Dict[str, Any]]:
        # 1. Try python-docx if installed
        try:
            import docx
            doc = docx.Document(file_path)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    row_text = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                    if row_text:
                        paragraphs.append(row_text)
            full_text = "\n\n".join(paragraphs)
            if full_text:
                return [{"content": full_text, "metadata": {"source": filename, "page": 1, "type": "docx"}}]
        except Exception:
            pass

        # 2. Native stdlib zipfile + XML fallback (zero dependencies required)
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            with zipfile.ZipFile(file_path, 'r') as z:
                xml_content = z.read('word/document.xml')
            root = ET.fromstring(xml_content)
            texts = [node.text for node in root.iter() if node.tag.endswith('}t') and node.text]
            full_text = " ".join(texts).strip()
            if full_text:
                return [{"content": full_text, "metadata": {"source": filename, "page": 1, "type": "docx"}}]
        except Exception as e:
            logger.warning(f"Stdlib Word XML fallback failed for '{filename}': {e}")

        return MultiParser._parse_text(file_path, filename)

    @staticmethod
    def _parse_text(file_path: str, filename: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read().strip()
        if not content:
            return []
        return [{
            "content": content,
            "metadata": {
                "source": filename,
                "page": 1,
                "type": "text"
            }
        }]
