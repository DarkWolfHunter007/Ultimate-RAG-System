import fitz  # PyMuPDF
from typing import List, Dict, Any
import os

class MultiParser:
    """Multi-format document parser supporting PDF, Markdown, and TXT files."""
    
    @staticmethod
    def parse_file(file_path: str, filename: str) -> List[Dict[str, Any]]:
        ext = os.path.splitext(filename)[1].lower()
        if ext == ".pdf":
            return MultiParser._parse_pdf(file_path, filename)
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
