import re
import uuid
import logging
from typing import Any

logger = logging.getLogger(__name__)

# Improved regex: splits on .!? followed by space+Capital, avoiding decimal and section number splits.
_SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9])')

_ABBR_SET = frozenset([
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "eg", "ie", "etc",
    "fig", "no", "vol", "sec", "ref", "para", "ch", "eq", "approx", "max", "min", "cl",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "st", "nd", "rd", "th"
])


class SemanticChunker:
    """
    Context-Aware Hierarchical Semantic Chunker.
    Preserves document structure (lists, tables, code blocks, paragraphs) with newline joining
    and adaptive sliding-window sentence overlap.
    """

    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50, child_size: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.child_size = child_size

    def _split_into_sentences(self, text: str) -> list[str]:
        paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
        sentences = []

        for para in paragraphs:
            # Preserve lists, tables, and code blocks as unbroken units
            if para.startswith("|") or para.startswith("```") or re.match(r'^\s*[-*]\s+', para):
                sentences.append(para)
                continue

            raw = [s.strip() for s in _SENTENCE_SPLIT_RE.split(para) if s.strip()]

            i = 0
            while i < len(raw) - 1:
                frag = raw[i]
                last_word = re.split(r'\s+', frag)[-1].rstrip('.').lower()
                if last_word in _ABBR_SET or len(last_word) <= 2:
                    raw[i + 1] = frag + " " + raw[i + 1]
                    raw.pop(i)
                else:
                    i += 1
            sentences.extend(raw)

        return sentences if sentences else [text.strip()]

    def _create_child_chunks(self, parent_id: str, parent_text: str, parent_meta: dict[str, Any]) -> list[dict[str, Any]]:
        sentences = self._split_into_sentences(parent_text)
        words_count = sum(len(s.split()) for s in sentences)

        parent_chunk = {
            "chunk_id": parent_id,
            "content": parent_text,
            "metadata": {**parent_meta, "parent_id": parent_id, "chunk_type": "parent"}
        }

        if words_count <= self.child_size:
            parent_chunk["metadata"]["chunk_type"] = "child"
            return [parent_chunk]

        chunks = [parent_chunk]
        current_sentences: list[str] = []
        current_word_count = 0
        c_idx = 0

        breadcrumb = ""
        src = parent_meta.get("source", "")
        heading = parent_meta.get("heading", "")
        if heading:
            breadcrumb = f"[Context: {heading}]\n"
        elif src:
            breadcrumb = f"[Source: {src}]\n"

        def _emit_child(sents: list[str], idx: int) -> dict[str, Any]:
            # Join with newline to preserve lists, tables, and paragraph formatting
            body = "\n".join(sents)
            content = (breadcrumb + body) if not body.startswith("[") else body
            return {
                "chunk_id": f"{parent_id}_c{idx}",
                "content": content,
                "metadata": {
                    **parent_meta,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "child_index": idx
                }
            }

        for sent in sentences:
            sent_words = len(sent.split())
            if current_word_count + sent_words > self.child_size and current_sentences:
                chunks.append(_emit_child(current_sentences, c_idx))
                c_idx += 1
                current_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences[-1:]
                current_word_count = sum(len(s.split()) for s in current_sentences)

            current_sentences.append(sent)
            current_word_count += sent_words

        if current_sentences:
            chunks.append(_emit_child(current_sentences, c_idx))

        return chunks

    def chunk_documents(self, parsed_pages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        all_chunks = []
        global_parent_id = 0

        for page in parsed_pages:
            content = page["content"]
            meta = page["metadata"]
            source_raw = str(meta.get("source", "doc"))
            clean_source = re.sub(r'[^a-zA-Z0-9_-]', '_', source_raw)
            is_markdown = meta.get("type") == "text" and source_raw.lower().endswith(".md")

            raw_blocks = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
            current_sentences: list[str] = []
            current_word_count = 0
            current_heading = ""

            def _emit_parent(sents: list[str], p_idx: int, heading: str, w_count: int):
                # Join with newline to preserve document structure
                parent_text = "\n".join(sents)
                uid = uuid.uuid4().hex[:6]
                pid = f"chk_{clean_source}_{p_idx}_{uid}"
                pmeta = {
                    **meta,
                    "chunk_index": p_idx,
                    "heading": heading,
                    "word_count": w_count
                }
                return self._create_child_chunks(pid, parent_text, pmeta)

            for block in raw_blocks:
                is_heading = block.startswith("#")
                if not is_heading and is_markdown:
                    is_heading = len(block.split()) < 8 and not block.endswith(".")

                if is_heading:
                    current_heading = block.lstrip("#").strip()
                    if current_sentences:
                        all_chunks.extend(_emit_parent(current_sentences, global_parent_id, current_heading, current_word_count))
                        global_parent_id += 1
                        current_sentences = []
                        current_word_count = 0
                    continue

                if block.startswith("|") or block.startswith("```"):
                    if current_sentences:
                        all_chunks.extend(_emit_parent(current_sentences, global_parent_id, current_heading, current_word_count))
                        global_parent_id += 1
                        current_sentences = []
                        current_word_count = 0

                    all_chunks.extend(_emit_parent([block], global_parent_id, current_heading, len(block.split())))
                    global_parent_id += 1
                    continue

                for sent in self._split_into_sentences(block):
                    sent_words = len(sent.split())
                    if current_word_count + sent_words > self.chunk_size and current_sentences:
                        all_chunks.extend(_emit_parent(current_sentences, global_parent_id, current_heading, current_word_count))
                        global_parent_id += 1
                        current_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                        current_word_count = sum(len(s.split()) for s in current_sentences)

                    current_sentences.append(sent)
                    current_word_count += sent_words

            if current_sentences:
                all_chunks.extend(_emit_parent(current_sentences, global_parent_id, current_heading, current_word_count))
                global_parent_id += 1

        return all_chunks
