"""All parser settings live here. Tune numbers here, not in the logic.

Bump PARSER_VERSION whenever a change alters the parser's output
(new chunk size, new cleaning rule, changed split).
"""

PARSER_VERSION = "0.3.0"

# --- Input validation (rules 1, 5) ---
SUPPORTED_EXTENSIONS = {".txt", ".pdf"}   # Tier 1. Add ".docx", ".csv", ".xlsx" in Tier 2
MAX_FILE_SIZE_MB = 20

# --- Cleaning (rules 12, 16) ---
HEADER_FOOTER_PAGE_RATIO = 0.5    # a line on more than 50% of pages counts as a header/footer
CLEANING_LOSS_WARNING = 0.7       # warn if cleaning removes more than 70% of the text

HEADER_LINES = 2                  # how many top-of-page lines can be a header
FOOTER_LINES = 2                  # how many bottom-of-page lines can be a footer
MIN_PAGES_FOR_FURNITURE = 3       # need this many pages to detect repeated headers/footers

# --- Chunking (rules 19, 20) ---
MIN_CHUNK_WORDS = 30
TARGET_CHUNK_WORDS = 400
MAX_CHUNK_WORDS = 600
OVERLAP_SENTENCES = 1

# --- Junk / duplicate chunk removal (rule 22) ---
MIN_ALPHA_RATIO = 0.5             # drop chunks where under 50% of characters are letters
DUPLICATE_SIMILARITY = 0.9        # chunks this similar count as duplicates
CUT_REFERENCES = True             # drop everything after a "References" heading (rule 15)