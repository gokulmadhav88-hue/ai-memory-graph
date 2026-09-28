import logging
import sys

from src.ingestion.parser import parse_document

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

chunks = parse_document(sys.argv[1])
print(f"\n{len(chunks)} chunks\n")
for c in chunks:
    print(f"--- chunk {c.chunk_index} | section: {c.section} | {len(c.text.split())} words ---")
    print(c.text)
    print()