import re
import uuid
from typing import List, Dict, Any

class SemanticChunker:
    """Hierarchical Parent-Child Aware Chunker. 
    Indexes small child sub-chunks for high vector precision, while linking back to full parent content for LLM synthesis.
    """

    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 40, child_size: int = 100):
        self.chunk_size = chunk_size  # Parent chunk size (words)
        self.chunk_overlap = chunk_overlap
        self.child_size = child_size  # Child sub-chunk size for high vector precision

    def _create_child_chunks(self, parent_id: str, parent_text: str, parent_meta: Dict[str, Any], clean_source: str) -> List[Dict[str, Any]]:
        words = parent_text.split()
        if len(words) <= self.child_size:
            return [{
                "chunk_id": parent_id,
                "content": parent_text,
                "metadata": {**parent_meta, "parent_id": parent_id, "parent_content": parent_text}
            }]

        child_chunks = []
        c_idx = 0
        for i in range(0, len(words), self.child_size):
            sub_words = words[i:i + self.child_size + 20]
            if sub_words:
                c_text = " ".join(sub_words)
                cid = f"{parent_id}_c{c_idx}"
                child_chunks.append({
                    "chunk_id": cid,
                    "content": c_text,
                    "metadata": {
                        **parent_meta,
                        "parent_id": parent_id,
                        "parent_content": parent_text,
                        "child_index": c_idx
                    }
                })
                c_idx += 1
        return child_chunks

    def chunk_documents(self, parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        global_parent_id = 0

        for page in parsed_pages:
            content = page["content"]
            meta = page["metadata"]
            source_raw = str(meta.get("source", "doc"))
            clean_source = re.sub(r'[^a-zA-Z0-9_-]', '_', source_raw)
            
            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
            
            current_words = []
            current_heading = ""

            for para in paragraphs:
                if para.startswith("#") or (len(para.split()) < 8 and not para.endswith(".")):
                    current_heading = para.lstrip("#").strip()

                words = para.split()
                if len(current_words) + len(words) > self.chunk_size and current_words:
                    parent_text = " ".join(current_words)
                    uid = uuid.uuid4().hex[:6]
                    pid = f"chk_{clean_source}_{global_parent_id}_{uid}"
                    pmeta = {
                        **meta,
                        "chunk_index": global_parent_id,
                        "heading": current_heading,
                        "word_count": len(current_words)
                    }

                    children = self._create_child_chunks(pid, parent_text, pmeta, clean_source)
                    all_chunks.extend(children)

                    global_parent_id += 1
                    current_words = current_words[-self.chunk_overlap:] + words
                else:
                    current_words.extend(words)

            if current_words:
                parent_text = " ".join(current_words)
                uid = uuid.uuid4().hex[:6]
                pid = f"chk_{clean_source}_{global_parent_id}_{uid}"
                pmeta = {
                    **meta,
                    "chunk_index": global_parent_id,
                    "heading": current_heading,
                    "word_count": len(current_words)
                }

                children = self._create_child_chunks(pid, parent_text, pmeta, clean_source)
                all_chunks.extend(children)
                global_parent_id += 1

        return all_chunks
