import re
import uuid
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Known abbreviations whose trailing period must NOT trigger a sentence split.
# Stored as a frozenset for O(1) lookup in the repair pass.
_ABBR_SET = frozenset([
    # Titles / honorifics
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
    # Latin / common
    "vs", "eg", "ie", "etc",
    # Document / publishing
    "fig", "no", "vol", "sec", "ref", "para", "ch", "eq", "approx", "max", "min", "cl",
    # Months
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # Roman numerals I–XX
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx",
])

# Pass 1: naive sentence-boundary split.
#   Splits on: !? followed by whitespace, blank lines, or . followed by whitespace + uppercase/digit.
#   Deliberately oversplits — the repair pass below fixes false positives.
_NAIVE_SPLIT_RE = re.compile(r'(?<=[!?])\s+|\n\s*\n|\.(?=\s+[A-Z0-9])\s*')



class SemanticChunker:
    """Sentence & Heading Aware Hierarchical Parent-Child Chunker."""

    def __init__(self, chunk_size: int = 400, chunk_overlap: int = 50, child_size: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.child_size = child_size

    def _split_into_sentences(self, text: str) -> List[str]:
        # Pass 1: naive split on sentence boundaries (may oversplit on abbreviations)
        raw = [s.strip() for s in _NAIVE_SPLIT_RE.split(text) if s.strip()]
        if not raw:
            return [text.strip()] if text.strip() else []

        # Pass 2: repair false splits — if a fragment's last word is a known abbreviation
        # or a short token (≤3 chars, covers "23", "4a", single-letter variables), rejoin
        # it with the next fragment because the period was not a sentence boundary.
        merged: List[str] = []
        i = 0
        while i < len(raw):
            frag = raw[i]
            last_word = re.split(r'\s+', frag)[-1].rstrip('.').lower()
            if i + 1 < len(raw) and (last_word in _ABBR_SET or len(last_word) <= 3):
                # Rejoin: the period was inside an abbreviation or short token
                raw[i + 1] = frag + ". " + raw[i + 1]
            else:
                merged.append(frag)
            i += 1

        return merged if merged else [text.strip()]

    def _create_child_chunks(self, parent_id: str, parent_text: str, parent_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        sentences = self._split_into_sentences(parent_text)
        words_count = sum(len(s.split()) for s in sentences)

        # Parent chunk — always emitted so VectorIndexer can resolve full context by ID.
        # chunk_type="parent" marks it to be excluded from search but fetchable by ID.
        parent_chunk = {
            "chunk_id": parent_id,
            "content": parent_text,
            "metadata": {**parent_meta, "parent_id": parent_id, "chunk_type": "parent"}
        }

        if words_count <= self.child_size:
            # Short parent — it IS the child; mark as child so it surfaces in search.
            parent_chunk["metadata"]["chunk_type"] = "child"
            return [parent_chunk]

        # Long parent: store parent + all children. Children carry parent_id for lookup.
        chunks = [parent_chunk]
        current_sentences: List[str] = []
        current_word_count = 0
        c_idx = 0
        heading_prefix = f"[{parent_meta['heading']}] " if parent_meta.get("heading") else ""

        for sent in sentences:
            sent_words = len(sent.split())
            if current_word_count + sent_words > self.child_size and current_sentences:
                child_text = heading_prefix + " ".join(current_sentences)
                cid = f"{parent_id}_c{c_idx}"
                chunks.append({
                    "chunk_id": cid,
                    "content": child_text,
                    "metadata": {
                        **parent_meta,
                        "parent_id": parent_id,
                        "chunk_type": "child",
                        "child_index": c_idx
                    }
                })
                c_idx += 1
                current_sentences = current_sentences[-1:]
                current_word_count = sum(len(s.split()) for s in current_sentences)

            current_sentences.append(sent)
            current_word_count += sent_words

        if current_sentences:
            child_text = heading_prefix + " ".join(current_sentences)
            cid = f"{parent_id}_c{c_idx}"
            chunks.append({
                "chunk_id": cid,
                "content": child_text,
                "metadata": {
                    **parent_meta,
                    "parent_id": parent_id,
                    "chunk_type": "child",
                    "child_index": c_idx
                }
            })

        return chunks

    def chunk_documents(self, parsed_pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        all_chunks = []
        global_parent_id = 0

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"\n=== [SEMANTIC CHUNKER START] Processing {len(parsed_pages)} document page(s) ===")

        for page in parsed_pages:
            content = page["content"]
            meta = page["metadata"]
            source_raw = str(meta.get("source", "doc"))
            clean_source = re.sub(r'[^a-zA-Z0-9_-]', '_', source_raw)

            # BUG-03 fix: only apply the shape-heuristic heading fallback for markdown files.
            # For PDF (type="pdf") headings are already tagged with "#" by multi_parser.
            # For plain TXT, the 8-word heuristic silently misfires on short sentences.
            is_markdown = meta.get("type") == "text" and source_raw.lower().endswith(".md")

            paragraphs = [p.strip() for p in re.split(r'\n\s*\n', content) if p.strip()]
            current_sentences: List[str] = []
            current_word_count = 0
            current_heading = ""

            for para in paragraphs:
                is_heading = para.startswith("#")
                if not is_heading and is_markdown:
                    is_heading = len(para.split()) < 8 and not para.endswith(".")
                if is_heading:
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
                        all_chunks.extend(self._create_child_chunks(pid, parent_text, pmeta))
                        global_parent_id += 1
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
                all_chunks.extend(self._create_child_chunks(pid, parent_text, pmeta))
                global_parent_id += 1

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[SEMANTIC CHUNKER 200 OK] Generated {len(all_chunks)} chunks across {global_parent_id} parent block(s) for {len(parsed_pages)} page(s)")

        return all_chunks
