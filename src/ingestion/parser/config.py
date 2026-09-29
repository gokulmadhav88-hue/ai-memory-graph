"""All parser settings live here. Tune numbers here, not in the logic.

Bump PARSER_VERSION whenever a change alters the parser's output.
"""

PARSER_VERSION = "0.5.0"

# --- Input validation (rules 1, 5) ---
SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv", ".xlsx"}
MAX_FILE_SIZE_MB = 20

# --- Cleaning (rules 12, 16) ---
HEADER_LINES = 2
FOOTER_LINES = 2
MIN_PAGES_FOR_FURNITURE = 3
HEADER_FOOTER_PAGE_RATIO = 0.5
CLEANING_LOSS_WARNING = 0.7
CUT_REFERENCES = True

# --- Chunking (rules 19, 20) ---
MIN_CHUNK_WORDS = 30
TARGET_CHUNK_WORDS = 400
MAX_CHUNK_WORDS = 600
OVERLAP_SENTENCES = 1

# --- Junk / duplicate chunk removal (rule 22) ---
MIN_ALPHA_RATIO = 0.5
DUPLICATE_SIMILARITY = 0.95

# --- Tables (rules 24-27) ---
MAX_TABLE_COLS_FOR_SENTENCES = 8
TABLE_ROWS_PER_CHUNK = 10
MAX_TABLE_ROWS = 500
# a table where fewer than this share of non-header cells contain any letters
# is treated as numeric-heavy: still stored, but flagged (new gap fix)
MIN_ALPHA_CELL_RATIO_FOR_ENTITIES = 0.3
