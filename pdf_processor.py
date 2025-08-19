
"""
AFH1 Structure-Aware PDF Processor for RAG
------------------------------------------
- Extracts text (page-by-page when possible) from AFH1 PDF.
- Splits by Chapter -> Section to preserve semantics.
- Creates ~550-token chunks with 60-token overlap inside sections.
- Adds rich metadata (chapter/section ids and titles, page ranges, edition/publish date if found).
- Optionally creates "micro-chunks" (1–3 sentences) to power acronym/definition lookups.

Dependencies:
    pip install pdfminer.six
"""

from __future__ import annotations
import os
import re
import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional

from pdfminer.high_level import extract_text, extract_pages
from pdfminer.layout import LTTextContainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("afh1_pdf_processor")

# ---------------------------
# Helpers & Data Structures
# ---------------------------

@dataclass
class Chunk:
    text: str
    source: str
    chunk_id: int
    doc_type: str
    chapter_number: Optional[int]
    chapter_title: Optional[str]
    section_id: Optional[str]
    section_title: Optional[str]
    start_page: Optional[int]
    end_page: Optional[int]
    edition: Optional[str]
    total_chunks: int = 0
    granularity: str = "section_chunk"  # or "micro_chunk"


AFH_CHAPTER_RE = re.compile(r'(?m)^\s*Chapter\s+(\d+)\s*[—-]\s*(.+?)\s*$')
AFH_SECTION_RE = re.compile(r'(?m)^\s*Section\s+([A-Z0-9\-]+)\s*[—-]\s*(.+?)\s*$')

# Simple sentence splitter that respects common punctuation.
SENTENCE_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z(])')

# Extract probable edition/publish string (e.g., "1 November 2024", "15 February 2025")
DATE_RE = re.compile(
    r'(?i)\b(\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b'
)

def _safe_extract_pages(file_path: str) -> List[str]:
    """Return a list of page texts using pdfminer.extract_pages (layout traversal)."""
    pages = []
    try:
        for page_layout in extract_pages(file_path):
            lines = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    lines.append(element.get_text())
            pages.append("".join(lines))
        if not pages:
            raise RuntimeError("No pages extracted via layout traversal.")
        return pages
    except Exception as e:
        logger.warning(f"Falling back to whole-document extract_text: {e}")
        # Fallback: single blob
        text = extract_text(file_path) or ""
        return [text]

def _find_edition_text(head_text: str) -> Optional[str]:
    """Try to locate a publish/edition date string in the front matter."""
    m = DATE_RE.search(head_text[:5000])
    return m.group(0) if m else None

def _iter_structure(full_text: str) -> List[Tuple[Tuple[int, str], List[Tuple[str, str, int, int]]]]:
    """
    Locate chapter and section boundaries.
    Returns a list of:
       [ ((chapter_number, chapter_title), [ (section_id, section_title, start_idx, end_idx), ... ] ), ... ]
    Where indices are string offsets into full_text.
    """
    chapters = []
    for m in AFH_CHAPTER_RE.finditer(full_text):
        chapters.append((m.start(), int(m.group(1)), m.group(2).strip()))
    # Append sentinel end
    chapters.append((len(full_text), None, None))

    result = []
    for i in range(len(chapters)-1):
        chap_start, chap_num, chap_title = chapters[i]
        chap_end = chapters[i+1][0]

        chap_text = full_text[chap_start:chap_end]

        # find sections within this chapter span, with absolute offsets
        sections = []
        for sm in AFH_SECTION_RE.finditer(chap_text):
            abs_start = chap_start + sm.start()
            sections.append((abs_start, sm.group(1).strip(), sm.group(2).strip()))
        # Add sentinel end for sections
        section_bounds = []
        if sections:
            sections.append((chap_end, None, None))
            for j in range(len(sections)-1):
                s_start, sid, stitle = sections[j]
                s_end = sections[j+1][0]
                section_bounds.append((sid, stitle, s_start, s_end))
        else:
            # No explicit sections; treat whole chapter as one section
            section_bounds.append(("CHAP", chap_title or f"Chapter {chap_num}", chap_start, chap_end))

        result.append(((chap_num, chap_title), section_bounds))

    return result

def _map_offsets_to_pages(pages: List[str], full_text: str) -> List[int]:
    """
    Build a list mapping from character offset to page index.
    We compute cumulative lengths of page texts and then binary-search page indices.
    Returns a list of cumulative end offsets per page.
    """
    cum = []
    total = 0
    for pg in pages:
        total += len(pg)
        cum.append(total)
    return cum

def _offset_to_page(cum_ends: List[int], offset: int) -> int:
    """Binary search: which page contains this character offset?"""
    import bisect
    idx = bisect.bisect_left(cum_ends, offset)
    return idx + 1  # 1-based page numbers

def _clean(text: str) -> str:
    # Keep paragraphs, remove excessive spaces
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def _sentences(text: str) -> List[str]:
    text = _clean(text)
    if not text:
        return []
    # Protect em-dash enumerations by replacing with period+space surrogate to avoid joining
    sents = SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s.strip()]

def _tokens_count(text: str) -> int:
    # Approximate tokens as words for simplicity
    return len(text.split())

def _pack_sentences(sent_list: List[str], target_size: int = 550, overlap: int = 60) -> List[str]:
    """Greedy pack sentences into chunks ~target_size with token overlap."""
    chunks = []
    cur = []
    cur_len = 0
    for s in sent_list:
        sl = _tokens_count(s)
        if cur and cur_len + sl > target_size:
            chunk_text = " ".join(cur).strip()
            if _tokens_count(chunk_text) >= 50:
                chunks.append(chunk_text)
            # Start next chunk with overlap of last sentences
            # Compute approx words for overlap by popping sentences from end
            if overlap > 0:
                words_needed = overlap
                tail = []
                for t in reversed(cur):
                    tw = _tokens_count(t)
                    if words_needed <= 0:
                        break
                    tail.append(t)
                    words_needed -= tw
                cur = list(reversed(tail))
                cur_len = sum(_tokens_count(x) for x in cur)
            else:
                cur = []
                cur_len = 0
            
        cur.append(s)
        cur_len += sl
    if cur:
        chunk_text = " ".join(cur).strip()
        if _tokens_count(chunk_text) >= 50:
            chunks.append(chunk_text)
        return chunks
    
def _make_micro_chunks(text: str) -> List[str]:
    """Create small 1–3 sentence micro-chunks to improve acronym/definition recall."""
    sents = _sentences(text)
    micro = []
    i = 0
    while i < len(sents):
        block = sents[i:i+3]
        bt = " ".join(block).strip()
        if 10 <= _tokens_count(bt) <= 120:
            micro.append(bt)
        i += 3
    return micro

class AFH1PDFProcessor:
    def __init__(self, section_chunk_size: int = 550, section_overlap: int = 60,
                 make_micro_chunks: bool = True):
        self.section_chunk_size = section_chunk_size
        self.section_overlap = section_overlap
        self.make_micro_chunks = make_micro_chunks

    def process_pdf(self, file_path: str) -> List[Dict]:
        """
        Full, structure-aware processing for AFH1.
        Returns a list of dicts ready for upsert into your vector DB.
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
        
        # 1) Extract text per page (fallback to single blob)
        page_texts = _safe_extract_pages(file_path)
        full_text = "".join(page_texts)
        logger.info(f"Extracted {len(page_texts)} pages, {len(full_text)} characters")

        # 2) Edition (publish date) detection from first pages
        edition = _find_edition_text(page_texts[0] if page_texts else full_text) or None
        if edition:
            logger.info(f"Detected edition/publish date: {edition}")

        # 3) Build page offset map
        cum_ends = _map_offsets_to_pages(page_texts, full_text)

        # 4) Find chapter/section bounds
        structure = _iter_structure(full_text)
        logger.info(f"Detected {len(structure)} chapters")

        # 5) Chunk within sections
        out: List[Chunk] = []
        chunk_id = 0
        doc_type = os.path.splitext(os.path.basename(file_path))[0].upper() or "PDF"

        for (chap_num, chap_title), section_list in structure:
            for sid, stitle, s_start, s_end in section_list:
                sec_text = full_text[s_start:s_end]
                sentences = _sentences(sec_text)
                if not sentences:
                    continue
                section_chunks = _pack_sentences(sentences, self.section_chunk_size, self.section_overlap)

                # page mapping
                start_pg = _offset_to_page(cum_ends, s_start)
                end_pg = _offset_to_page(cum_ends, s_end-1) if s_end > s_start else start_pg

                for ch in section_chunks:
                    out.append(Chunk(
                        text=ch,
                        source=file_path,
                        chunk_id=chunk_id,
                        doc_type=doc_type,
                        chapter_number=chap_num,
                        chapter_title=chap_title,
                        section_id=sid,
                        section_title=stitle,
                        start_page=start_pg,
                        end_page=end_pg,
                        edition=edition,
                        total_chunks=0,
                        granularity="section_chunk"
                    ))
                    chunk_id += 1

                # Optional micro-chunks
                if self.make_micro_chunks:
                    micro = _make_micro_chunks(sec_text)
                    for mch in micro:
                        out.append(Chunk(
                            text=mch,
                            source=file_path,
                            chunk_id=chunk_id,
                            doc_type=doc_type,
                            chapter_number=chap_num,
                            chapter_title=chap_title,
                            section_id=sid,
                            section_title=stitle,
                            start_page=start_pg,
                            end_page=end_pg,
                            edition=edition,
                            total_chunks=0,
                            granularity="micro_chunk"
                        ))
                        chunk_id += 1

        # 6) Finalize total count
        total = len(out)
        for i in range(total):
            out[i].total_chunks = total

        logger.info(f"Created {total} chunks (including micro-chunks={self.make_micro_chunks})")
        return [asdict(x) for x in out]
    
    def get_processor_stats(self) -> dict:
        """
        Get information about processor configuration and performance.
        
        Returns:
            Dictionary with processor settings and statistics
        """
        return {
            "chunk_size": self.section_chunk_size,
            "overlap": self.section_overlap,
            "make_micro_chunks": self.make_micro_chunks,
            "description": "AFH1 Structure-Aware PDF processor with enhanced metadata",
            "features": [
                "Chapter/Section detection",
                "Page mapping",
                "Edition detection",
                "Micro-chunking",
                "Rich metadata extraction"
            ]
        }


# Convenience function
def process(file_path: str, section_chunk_size: int = 550, section_overlap: int = 60,
            make_micro_chunks: bool = True) -> List[Dict]:
    return AFH1PDFProcessor(section_chunk_size, section_overlap, make_micro_chunks).process_pdf(file_path)


if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser(description="AFH1 Structure-Aware PDF Processor")
    parser.add_argument("file", help="Path to AFH1 PDF")
    parser.add_argument("--chunk", type=int, default=550, help="Section chunk size (words)")
    parser.add_argument("--overlap", type=int, default=60, help="Overlap (words)")
    parser.add_argument("--no-micro", action="store_true", help="Disable micro-chunk generation")
    args = parser.parse_args()

    chunks = process(args.file, args.chunk, args.overlap, not args.no_micro)
    print(json.dumps(chunks[:3], indent=2))  # preview first 3
    print(f"... total chunks: {len(chunks)}")
