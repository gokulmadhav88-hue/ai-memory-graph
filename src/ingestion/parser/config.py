"""All parser settings live here. Tune numbers here, not in the logic.

Bump PARSER_VERSION whenever a change alters the parser's output.
"""

PARSER_VERSION = "0.6.0"

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

# --- Tier 3: figures (rules 28-30) ---
# OFF by default. A plain parse_document(path) call behaves exactly as in Tier 1/2:
# figures are counted and skipped. Turn on by also passing a describe_image callback.
ENABLE_FIGURE_DESCRIPTIONS = False
MAX_FIGURES_PER_DOC = 20          # cost cap: one document can't run up unlimited vision calls
MIN_IMAGE_SIZE_PX = 50            # skip icons, bullets, dividers

# --- Tier 3: OCR for scanned PDFs (alternative to rule 4's "skip with a warning") ---
ENABLE_OCR = False                # OFF by default: scanned PDFs are still just skipped
OCR_RENDER_DPI = 200              # higher = better OCR accuracy, slower, bigger images
OCR_MIN_CHARS = 20                # below this, treat OCR output as noise, not real text
