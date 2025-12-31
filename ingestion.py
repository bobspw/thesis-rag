"""
Multi-Source Ingestion Module for Thesis Research Assistant
============================================================

Handles extraction and chunking from multiple source types:
- PDF research papers (arXiv, conference papers)
- Web articles (transformer-circuits.pub, blogs)
- YouTube transcripts (Neel Nanda, Karpathy, 3B1B)

Author: Ajay Pravin Mahale
Thesis: Explainable AI for LLMs

All sources are chunked with page/section tracking for proper citation.
"""

import os
import re
from dataclasses import dataclass
from pypdf import PdfReader


@dataclass
class DocumentChunk:
    """
    A chunk of text with citation metadata.

    Works for all source types:
    - PDFs: page_number is the actual page
    - Web articles: page_number represents section number
    - YouTube: page_number represents approximate timestamp section
    """
    text: str
    page_number: int
    source_file: str
    chunk_index: int
    source_type: str = "pdf"  # "pdf", "web", "youtube"

    def __repr__(self):
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"Chunk({self.source_file}, p.{self.page_number}, '{preview}')"


class PDFIngester:
    """Handles PDF research papers."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_pages(self, pdf_path: str) -> list[tuple[int, str]]:
        """Extract text from each page of a PDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        reader = PdfReader(pdf_path)
        pages = []

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            text = self._clean_text(text)
            pages.append((i + 1, text))

        return pages

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def chunk_text_with_pages(self, pages: list[tuple[int, str]]) -> list[tuple[str, int]]:
        """Split text into overlapping chunks."""
        chunks_with_pages = []

        for page_num, page_text in pages:
            if not page_text.strip():
                continue

            start = 0
            while start < len(page_text):
                end = start + self.chunk_size
                chunk = page_text[start:end]

                if len(chunk) < self.chunk_size // 4 and chunks_with_pages:
                    prev_text, prev_page = chunks_with_pages[-1]
                    chunks_with_pages[-1] = (prev_text + " " + chunk, prev_page)
                else:
                    chunks_with_pages.append((chunk, page_num))

                start += self.chunk_size - self.chunk_overlap

        return chunks_with_pages

    def ingest_pdf(self, pdf_path: str) -> list[DocumentChunk]:
        """Ingest a single PDF."""
        source_file = os.path.basename(pdf_path)
        pages = self.extract_pages(pdf_path)
        chunks_with_pages = self.chunk_text_with_pages(pages)

        documents = []
        for i, (text, page_num) in enumerate(chunks_with_pages):
            doc = DocumentChunk(
                text=text,
                page_number=page_num,
                source_file=source_file,
                chunk_index=i,
                source_type="pdf"
            )
            documents.append(doc)

        return documents


class TextIngester:
    """Handles plain text files (web articles, YouTube transcripts)."""

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_text_file(self, filepath: str, source_type: str = "web") -> list[DocumentChunk]:
        """
        Ingest a text file.

        For text files, we create artificial "pages" based on sections
        or chunk boundaries for citation purposes.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        source_file = os.path.basename(filepath)

        # Split by major sections (double newlines) or chunk by size
        sections = self._split_into_sections(content)

        documents = []
        chunk_index = 0

        for section_num, section_text in enumerate(sections, 1):
            # Further chunk if section is too large
            chunks = self._chunk_section(section_text)

            for chunk in chunks:
                if len(chunk.strip()) < 50:  # Skip tiny chunks
                    continue

                doc = DocumentChunk(
                    text=chunk,
                    page_number=section_num,  # Section number as "page"
                    source_file=source_file,
                    chunk_index=chunk_index,
                    source_type=source_type
                )
                documents.append(doc)
                chunk_index += 1

        return documents

    def _split_into_sections(self, text: str) -> list[str]:
        """Split text into major sections."""
        # Split on multiple newlines (section breaks)
        sections = re.split(r'\n{3,}', text)

        # If no clear sections, split by double newlines
        if len(sections) < 3:
            sections = re.split(r'\n\n+', text)

        # Filter empty sections
        sections = [s.strip() for s in sections if s.strip()]

        return sections

    def _chunk_section(self, text: str) -> list[str]:
        """Chunk a section if it's too large."""
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('. ')
                if last_period > self.chunk_size // 2:
                    chunk = chunk[:last_period + 1]
                    end = start + last_period + 1

            chunks.append(chunk.strip())
            start = end - self.chunk_overlap

        return chunks


class MultiSourceIngester:
    """
    Unified ingester for all thesis research sources.

    Handles:
    - data/pdfs/*.pdf          → PDF research papers
    - data/web_articles/*.txt  → Web articles and blog posts
    - data/youtube_transcripts/*.txt → YouTube video transcripts
    """

    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 300):
        self.pdf_ingester = PDFIngester(chunk_size, chunk_overlap)
        self.text_ingester = TextIngester(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def ingest_all(self, data_dir: str = "data") -> list[DocumentChunk]:
        """
        Ingest all sources from the data directory.

        Expected structure:
        data/
        ├── pdfs/                    # PDF papers
        ├── web_articles/            # Web content
        └── youtube_transcripts/     # Video transcripts
        """
        all_chunks = []

        # 1. Ingest PDFs
        pdf_dir = os.path.join(data_dir, "pdfs")
        if os.path.exists(pdf_dir):
            pdf_chunks = self._ingest_pdfs(pdf_dir)
            all_chunks.extend(pdf_chunks)

        # 2. Ingest web articles
        web_dir = os.path.join(data_dir, "web_articles")
        if os.path.exists(web_dir):
            web_chunks = self._ingest_text_dir(web_dir, "web")
            all_chunks.extend(web_chunks)

        # 3. Ingest YouTube transcripts
        yt_dir = os.path.join(data_dir, "youtube_transcripts")
        if os.path.exists(yt_dir):
            yt_chunks = self._ingest_text_dir(yt_dir, "youtube")
            all_chunks.extend(yt_chunks)

        return all_chunks

    def _ingest_pdfs(self, pdf_dir: str) -> list[DocumentChunk]:
        """Ingest all PDFs in a directory."""
        chunks = []

        pdf_files = sorted([f for f in os.listdir(pdf_dir) if f.lower().endswith('.pdf')])

        if pdf_files:
            print(f"\n📄 Ingesting {len(pdf_files)} PDF papers...")

        for filename in pdf_files:
            filepath = os.path.join(pdf_dir, filename)
            try:
                doc_chunks = self.pdf_ingester.ingest_pdf(filepath)
                chunks.extend(doc_chunks)
                print(f"  ✓ {filename}: {len(doc_chunks)} chunks")
            except Exception as e:
                print(f"  ✗ {filename}: {e}")

        return chunks

    def _ingest_text_dir(self, text_dir: str, source_type: str) -> list[DocumentChunk]:
        """Ingest all text files in a directory."""
        chunks = []

        text_files = sorted([f for f in os.listdir(text_dir) if f.endswith('.txt')])

        type_labels = {"web": "🌐 web articles", "youtube": "🎥 YouTube transcripts"}

        if text_files:
            print(f"\n{type_labels.get(source_type, source_type)}: {len(text_files)} files...")

        for filename in text_files:
            filepath = os.path.join(text_dir, filename)
            try:
                doc_chunks = self.text_ingester.ingest_text_file(filepath, source_type)
                chunks.extend(doc_chunks)
                print(f"  ✓ {filename}: {len(doc_chunks)} chunks")
            except Exception as e:
                print(f"  ✗ {filename}: {e}")

        return chunks


# Backward compatibility - keep PDFIngester as default
def create_ingester(chunk_size: int = 1500, chunk_overlap: int = 300):
    """Create a multi-source ingester."""
    return MultiSourceIngester(chunk_size, chunk_overlap)


if __name__ == "__main__":
    ingester = MultiSourceIngester(chunk_size=1500, chunk_overlap=300)
    print("Multi-Source Ingester initialized")
    print(f"  Chunk size: {ingester.chunk_size} chars")
    print(f"  Overlap: {ingester.chunk_overlap} chars")
    print("\nSupported sources:")
    print("  - data/pdfs/*.pdf")
    print("  - data/web_articles/*.txt")
    print("  - data/youtube_transcripts/*.txt")
