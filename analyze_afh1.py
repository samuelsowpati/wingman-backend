#!/usr/bin/env python3
"""
Analyze AFH1 PDF content for improved chunking and indexing
"""

from pdf_processor import AFH1PDFProcessor

# Initialize the processor
pdf_processor = AFH1PDFProcessor()
import re

def analyze_afh1_structure():
    """Analyze the structure and content of AFH1 for optimization"""
    
    print("🔍 Analyzing AFH1 PDF structure...")
    # Process the PDF and get all chunks
    chunks = pdf_processor.process_pdf('afh1.pdf')
    if not chunks:
        print("❌ Failed to process PDF")
        return
        
    # Combine all chunk texts for analysis
    text = "\n\n".join(chunk['text'] for chunk in chunks)
    
    if not text:
        print("❌ Failed to extract text")
        return
    
    # Basic stats
    print(f"\n📊 BASIC STATISTICS:")
    print(f"Total characters: {len(text):,}")
    print(f"Total words: {len(text.split()):,}")
    print(f"Total lines: {len(text.split(chr(10))):,}")
    
    # Find chapter structure
    print(f"\n📖 CHAPTER STRUCTURE:")
    chapter_patterns = [
        r'Chapter\s+(\d+)\s*[-:]?\s*([^\n]+)',
        r'CHAPTER\s+(\d+)\s*[-:]?\s*([^\n]+)',
        r'\n\s*(\d+)\.\s*([A-Z][A-Z\s]{10,})',
    ]
    
    chapters = []
    for pattern in chapter_patterns:
        matches = re.finditer(pattern, text, re.IGNORECASE)
        for match in matches:
            if len(match.groups()) >= 2:
                num = match.group(1)
                title = match.group(2).strip()
                pos = match.start()
                chapters.append((pos, num, title, pattern))
    
    # Sort by position and remove duplicates
    chapters = sorted(list(set(chapters)), key=lambda x: x[0])
    
    print(f"Found {len(chapters)} chapter/section markers:")
    for i, (pos, num, title, pattern) in enumerate(chapters[:20]):  # Show first 20
        print(f"  {num}: {title[:60]}...")
        if i == 19 and len(chapters) > 20:
            print(f"  ... and {len(chapters) - 20} more")
    
    # Analyze content density
    print(f"\n📈 CONTENT DENSITY ANALYSIS:")
    sample_size = 10000
    chunks = [text[i:i+sample_size] for i in range(0, len(text), sample_size)]
    
    avg_words_per_chunk = sum(len(chunk.split()) for chunk in chunks) / len(chunks)
    print(f"Average words per {sample_size} char chunk: {avg_words_per_chunk:.1f}")
    
    # Find key section types
    print(f"\n🎯 KEY SECTION TYPES:")
    section_patterns = {
        'Leadership': r'leadership|leader|leading',
        'History': r'history|historical|heritage',
        'Core Values': r'core values|integrity|service|excellence',
        'Doctrine': r'doctrine|strategy|tactical',
        'Organization': r'organization|structure|command',
        'Training': r'training|education|development',
        'Operations': r'operations|mission|combat',
        'Technology': r'technology|equipment|aircraft',
    }
    
    for section_type, pattern in section_patterns.items():
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        print(f"  {section_type}: {matches} mentions")
    
    # Sample content from different parts
    print(f"\n📝 CONTENT SAMPLES:")
    
    # Beginning
    print(f"\nBEGINNING (first 500 chars):")
    print(repr(text[:500]))
    
    # Middle
    middle_start = len(text) // 2
    print(f"\nMIDDLE (around char {middle_start:,}):")
    print(repr(text[middle_start:middle_start+500]))
    
    # End
    print(f"\nEND (last 500 chars):")
    print(repr(text[-500:]))
    
    # Analyze chunk structure
    print(f"\n📦 CHUNK ANALYSIS:")
    section_chunks = [c for c in chunks if c['granularity'] == 'section_chunk']
    micro_chunks = [c for c in chunks if c['granularity'] == 'micro_chunk']
    
    print(f"Total chunks: {len(chunks)}")
    print(f"Section chunks: {len(section_chunks)}")
    print(f"Micro chunks: {len(micro_chunks)}")
    
    # Analyze chapter distribution
    chapters_found = sorted(list(set((c['chapter_number'], c['chapter_title']) 
                                   for c in chunks if c['chapter_number'] is not None)))
    print(f"\nChapters found in chunks:")
    for num, title in chapters_found:
        print(f"  Chapter {num}: {title}")
    
    # Sample chunk metadata
    if chunks:
        print(f"\n📋 Sample Chunk Metadata:")
        sample = chunks[0]
        print(f"  Document Type: {sample['doc_type']}")
        print(f"  Edition: {sample['edition']}")
        print(f"  Chapter: {sample['chapter_number']} - {sample['chapter_title']}")
        print(f"  Section: {sample['section_id']} - {sample['section_title']}")
        print(f"  Pages: {sample['start_page']} to {sample['end_page']}")
        print(f"  Granularity: {sample['granularity']}")
    
    return text, chunks, chapters

if __name__ == "__main__":
    analyze_afh1_structure()
