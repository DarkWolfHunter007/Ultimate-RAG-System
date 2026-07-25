import re
from typing import List, Dict, Any

class SemanticChunker:
    """Paragraph and sentence aware text chunker with heading context preservation."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size  # target words per chunk
        self.chunk_overlap = chunk_overlap

    def chunk_documents(self, parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        chunks = []
        global_chunk_id = 0

        for page in parsed_pages:
            content = page["content"]
            meta = page["metadata"]
            
            # Split text by paragraphs / double newlines first
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
            
            current_words = []
            current_heading = ""

            for para in paragraphs:
                # Detect heading lines (e.g. # Heading or ALL CAPS / short line ending with no period)
                if para.startswith("#") or (len(para.split()) < 8 and not para.endswith(".")):
                    current_heading = para.lstrip("#").strip()

                words = para.split()
                if len(current_words) + len(words) > self.chunk_size and current_words:
                    chunk_text = " ".join(current_words)
                    chunks.append({
                        "chunk_id": f"chk_{global_chunk_id}",
                        "content": chunk_text,
                        "metadata": {
                            **meta,
                            "chunk_index": global_chunk_id,
                            "heading": current_heading,
                            "word_count": len(current_words)
                        }
                    })
                    global_chunk_id += 1
                    # Retain overlap
                    current_words = current_words[-self.chunk_overlap:] + words
                else:
                    current_words.extend(words)

            if current_words:
                chunk_text = " ".join(current_words)
                chunks.append({
                    "chunk_id": f"chk_{global_chunk_id}",
                    "content": chunk_text,
                    "metadata": {
                        **meta,
                        "chunk_index": global_chunk_id,
                        "heading": current_heading,
                        "word_count": len(current_words)
                    }
                })
                global_chunk_id += 1

        return chunks
