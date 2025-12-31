"""
Thesis Research Assistant - Clean Terminal Interface
=====================================================

Author: Ajay Pravin Mahale
Thesis: Explainable AI for LLMs
"""

import argparse
import os
import sys
import textwrap

from ingestion import MultiSourceIngester, DocumentChunk
from vector_store import VectorStore
from qa_chain import RetrievalQAChain


DEFAULT_STORE_PATH = "data/vector_store"


def clear_screen():
    """Clear terminal screen."""
    os.system('clear' if os.name != 'nt' else 'cls')


def print_header():
    """Print clean header."""
    print()
    print("╔" + "═" * 50 + "╗")
    print("║" + "  THESIS RESEARCH ASSISTANT  ".center(50) + "║")
    print("║" + "  Explainable AI for LLMs  ".center(50) + "║")
    print("╚" + "═" * 50 + "╝")
    print()


def wrap_text(text: str, width: int = 60) -> str:
    """Wrap text for clean display."""
    paragraphs = text.split('\n\n')
    wrapped = []
    for p in paragraphs:
        p = ' '.join(p.split())  # Normalize whitespace
        if p:
            wrapped.append(textwrap.fill(p, width=width))
    return '\n\n'.join(wrapped)


def format_sources(citations) -> str:
    """Format sources cleanly."""
    seen = set()
    sources = []
    for c in citations:
        key = (c.source_file, c.page_number)
        if key not in seen:
            seen.add(key)
            name = c.source_file.replace('.pdf', '').replace('.txt', '')
            # Shorten long names
            if '_' in name:
                parts = name.split('_')
                if len(parts) >= 3:
                    name = f"{parts[1]} ({parts[-1]})"
            sources.append(f"{name}, p.{c.page_number}")
    return " | ".join(sources[:4])  # Max 4 sources


def check_ollama():
    """Verify Ollama is running."""
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False


def cmd_ingest(args):
    """Ingest all research sources."""
    data_dir = args.pdf_dir or "data"
    store_path = args.store_path or DEFAULT_STORE_PATH
    
    print_header()
    print("  Indexing your research sources...\n")
    
    ingester = MultiSourceIngester(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    store = VectorStore()
    
    all_chunks = ingester.ingest_all(data_dir)
    
    if not all_chunks:
        print("  ✗ No content found to index")
        sys.exit(1)
    
    print(f"\n  Total: {len(all_chunks)} chunks")
    print("  Embedding (this takes a few minutes)...")
    
    store.add_documents(all_chunks)
    store.save(store_path)
    
    print(f"\n  ✓ Ready! Run: python main.py chat")


def cmd_chat(args):
    """Interactive research session."""
    store_path = args.store_path or DEFAULT_STORE_PATH
    
    if not os.path.exists(store_path):
        print("\n  ✗ No index found. Run 'python main.py ingest' first.\n")
        sys.exit(1)
    
    store = VectorStore()
    store.load(store_path)
    
    chain = RetrievalQAChain(
        vector_store=store,
        model=args.model,
        num_chunks=args.num_chunks
    )
    
    clear_screen()
    print_header()
    print(f"  {store.index.ntotal} chunks indexed | Model: {args.model}")
    print()
    print("  Commands: 'exit' to quit, 'clear' to clear screen")
    print("─" * 52)
    
    while True:
        try:
            print()
            question = input("  You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  Good luck with the thesis! 👋\n")
            break
        
        if not question:
            continue
        
        if question.lower() in ('quit', 'exit', 'q'):
            print("\n  Good luck with the thesis! 👋\n")
            break
        
        if question.lower() == 'clear':
            clear_screen()
            print_header()
            print(f"  {store.index.ntotal} chunks indexed")
            print("─" * 52)
            continue
        
        print("  ...")
        
        response = chain.query(question)
        
        # Clean output
        print()
        print("─" * 52)
        print()
        
        # Wrap and display answer
        answer = wrap_text(response.answer, width=50)
        for line in answer.split('\n'):
            print(f"  {line}")
        
        print()
        print("─" * 52)
        
        # Show sources on one line
        if response.citations:
            sources = format_sources(response.citations)
            print(f"  📚 {sources}")
        
        print("─" * 52)


def cmd_query(args):
    """Single query mode."""
    store_path = args.store_path or DEFAULT_STORE_PATH
    
    store = VectorStore()
    store.load(store_path)
    
    chain = RetrievalQAChain(
        vector_store=store,
        model=args.model,
        num_chunks=args.num_chunks
    )
    
    response = chain.query(args.question)
    
    print()
    print("─" * 52)
    print()
    answer = wrap_text(response.answer, width=50)
    for line in answer.split('\n'):
        print(f"  {line}")
    print()
    print("─" * 52)
    if response.citations:
        sources = format_sources(response.citations)
        print(f"  📚 {sources}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Thesis Research Assistant")
    subparsers = parser.add_subparsers(dest='command')
    
    # Ingest
    ingest_parser = subparsers.add_parser('ingest')
    ingest_parser.add_argument('pdf_dir', nargs='?')
    ingest_parser.add_argument('--store-path')
    ingest_parser.add_argument('--chunk-size', type=int, default=1500)
    ingest_parser.add_argument('--chunk-overlap', type=int, default=300)
    
    # Query
    query_parser = subparsers.add_parser('query')
    query_parser.add_argument('question')
    query_parser.add_argument('--store-path')
    query_parser.add_argument('--model', default='llama3.2')
    query_parser.add_argument('--num-chunks', type=int, default=5)
    
    # Chat
    chat_parser = subparsers.add_parser('chat')
    chat_parser.add_argument('--store-path')
    chat_parser.add_argument('--model', default='llama3.2')
    chat_parser.add_argument('--num-chunks', type=int, default=5)
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        sys.exit(1)
    
    if not check_ollama():
        print("\n  ✗ Ollama not running. Start with: ollama serve\n")
        sys.exit(1)
    
    if args.command == 'ingest':
        cmd_ingest(args)
    elif args.command == 'query':
        cmd_query(args)
    elif args.command == 'chat':
        cmd_chat(args)


if __name__ == "__main__":
    main()
