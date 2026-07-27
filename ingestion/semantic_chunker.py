import re
import uuid
from typing import List, Dict, Any

class SemanticChunker:
    """Sentence & Heading Aware Hierarchical Parent-Child Chunker.
    Ensures complete sentences are preserved without mid-sentence splits,
    prefixes section heading context to sub-chunks, and links children to full parent text for LLM synthesis.
    """

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50, child_size: int = 150):
        self.chunk_size = chunk_size      # Target parent chunk size in words
        self.chunk_overlap = chunk_overlap  # Target overlap in words
        self.child_size = child_size      # Target child chunk size in words

    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text by sentence boundaries (. ! ? \n) without breaking sentences in half."""
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n\s*\n', text)
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences if sentences else [text.strip()]

    def _create_child_chunks(self, parent_id: str, parent_text: str, parent_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        sentences = self._split_into_sentences(parent_text)
        words_count = sum(len(s.split()) for s in sentences)

        if words_count <= self.child_size:
            return [{
                "chunk_id": parent_id,
                "content": parent_text,
                "metadata": {**parent_meta, "parent_id": parent_id, "parent_content": parent_text}
            }]

        child_chunks = []
        current_sentences = []
        current_word_count = 0
        c_idx = 0

        heading_prefix = f"[{parent_meta['heading']}] " if parent_meta.get("heading") else ""

        for sent in sentences:
            sent_words = len(sent.split())
            if current_word_count + sent_words > self.child_size and current_sentences:
                child_text = heading_prefix + " ".join(current_sentences)
                cid = f"{parent_id}_c{c_idx}"
                child_chunks.append({
                    "chunk_id": cid,
                    "content": child_text,
                    "metadata": {
                        **parent_meta,
                        "parent_id": parent_id,
                        "parent_content": parent_text,
                        "child_index": c_idx
                    }
                })
                c_idx += 1
                current_sentences = current_sentences[-1:]  # Keep last sentence for overlap
                current_word_count = sum(len(s.split()) for s in current_sentences)

            current_sentences.append(sent)
            current_word_count += sent_words

        if current_sentences:
            child_text = heading_prefix + " ".join(current_sentences)
            cid = f"{parent_id}_c{c_idx}"
            child_chunks.append({
                "chunk_id": cid,
                "content": child_text,
                "metadata": {
                    **parent_meta,
                    "parent_id": parent_id,
                    "parent_content": parent_text,
                    "child_index": c_idx
                }
            })

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
            current_sentences = []
            current_word_count = 0
            current_heading = ""

            for para in paragraphs:
                if para.startswith("#") or (len(para.split()) < 8 and not para.endswith(".")):
                    current_heading = para.lstrip("#").strip()
                    continue

                para_sentences = self._split_into_sentences(para)
                for sent in para_sentences:
                    sent_words = len(sent.split())
                    if current_word_count + sent_words > self.chunk_size and current_sentences:
                        parent_text = " ".join(current_sentences)
                        uid = uuid.uuid4().hex[:6]
                        pid = f"chk_{clean_source}_{global_parent_id}_{uid}"
                        pmeta = {
                            **meta,
                            "chunk_index": global_parent_id,
                            "heading": current_heading,
                            "word_count": current_word_count
                        }

                        children = self._create_child_chunks(pid, parent_text, pmeta)
                        all_chunks.extend(children)

                        global_parent_id += 1
                        # Retain last few sentences for overlap
                        current_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                        current_word_count = sum(len(s.split()) for s in current_sentences)

                    current_sentences.append(sent)
                    current_word_count += sent_words

            if current_sentences:
                parent_text = " ".join(current_sentences)
                uid = uuid.uuid4().hex[:6]
                pid = f"chk_{clean_source}_{global_parent_id}_{uid}"
                pmeta = {
                    **meta,
                    "chunk_index": global_parent_id,
                    "heading": current_heading,
                    "word_count": current_word_count
                }

                children = self._create_child_chunks(pid, parent_text, pmeta)
                all_chunks.extend(children)
                global_parent_id += 1

        return all_chunks
