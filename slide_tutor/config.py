import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env file from project root so users don't need to export env vars manually
load_dotenv()

APP_NAME = "Lecture Slide Tutor"
SCHEMA_VERSION = "v1"

DATA_ROOT = Path(".slide_tutor_data")
# Deprecated: MODULES_DIR is the legacy Phase 1.0 cache path.
# New code should use PROJECTS_DIR and project-first APIs in slide_tutor.projects.
MODULES_DIR = DATA_ROOT / "modules"  # legacy — kept for old cache compatibility
PROJECTS_DIR = DATA_ROOT / "projects"

SUPPORTED_EXTENSIONS = {".pdf"}

PAGE_IMAGE_WIDTH = 1400
THUMBNAIL_WIDTH = 260

MANIFEST_FILENAME = "manifest.json"
PROJECT_FILENAME = "project.json"
SLIDES_FILENAME = "slides.json"
CHUNKS_FILENAME = "chunks.json"

# ── Page-generation artifact cache (PIO-8) ──
# Artifacts are project-scoped but content-addressed by source file hash and
# parser/batch contract.  They are separate from active generations so a file
# add, replacement, or reorder can reuse unaffected page batches safely.
PAGE_ARTIFACTS_DIRNAME = "page_artifacts"
PAGE_BATCH_ARTIFACTS_DIRNAME = "batches"
GENERATION_LIFECYCLE_FILENAME = "generation_lifecycle.json"

TEXT_LOW_CHAR_THRESHOLD = 40

# ── LLM config (env vars with defaults for local Ollama) ──
LLM_PROVIDER = os.getenv("SLIDE_TUTOR_LLM_PROVIDER", "openai_compatible")
LLM_BASE_URL = os.getenv("SLIDE_TUTOR_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("SLIDE_TUTOR_API_KEY", "ollama")
LLM_MODEL = os.getenv("SLIDE_TUTOR_MODEL", "llama3.2")
LIGHTWEIGHT_MODEL = os.getenv("SLIDE_TUTOR_LIGHTWEIGHT_MODEL", LLM_MODEL)
QA_PLANNER_MODEL = os.getenv("SLIDE_TUTOR_QA_PLANNER_MODEL", LIGHTWEIGHT_MODEL)
QA_AGENT_MODEL = os.getenv("SLIDE_TUTOR_QA_AGENT_MODEL", QA_PLANNER_MODEL)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


# Native tool calling is opt-in until a configured Provider is live-verified.
QA_TOOL_CALLING_ENABLED = _env_bool("SLIDE_TUTOR_QA_TOOL_CALLING", False)
QA_TOOL_CALLING_MAX_CALLS = max(
    1, int(os.getenv("SLIDE_TUTOR_QA_TOOL_CALLING_MAX_CALLS", "4"))
)

# Bounded multi-round Agent v2 is separately gated from v1 single-turn tools.
QA_AGENT_ENABLED = _env_bool("SLIDE_TUTOR_QA_AGENT", False)
QA_AGENT_MAX_ROUNDS = max(1, int(os.getenv("SLIDE_TUTOR_QA_AGENT_MAX_ROUNDS", "3")))
QA_AGENT_MAX_CALLS_PER_ROUND = max(
    1, int(os.getenv("SLIDE_TUTOR_QA_AGENT_MAX_CALLS_PER_ROUND", "4"))
)
QA_AGENT_MAX_TOTAL_CALLS = max(
    1, int(os.getenv("SLIDE_TUTOR_QA_AGENT_MAX_TOTAL_CALLS", "6"))
)
QA_AGENT_MAX_SECONDS = float(os.getenv("SLIDE_TUTOR_QA_AGENT_MAX_SECONDS", "45"))
QA_AGENT_MAX_CONTROLLER_TOKENS = max(
    0, int(os.getenv("SLIDE_TUTOR_QA_AGENT_MAX_CONTROLLER_TOKENS", "0"))
)

# ── Retrieval config ──
RETRIEVAL_TOP_K = 5       # semantic matches to supplement page context
RETRIEVAL_CANDIDATE_K = max(
    1, int(os.getenv("SLIDE_TUTOR_RETRIEVAL_CANDIDATE_K", "12"))
)
RETRIEVAL_RRF_K = max(1, int(os.getenv("SLIDE_TUTOR_RETRIEVAL_RRF_K", "60")))
RETRIEVAL_TITLE_LEXICAL_WEIGHT = float(
    os.getenv("SLIDE_TUTOR_RETRIEVAL_TITLE_LEXICAL_WEIGHT", "1.5")
)
RETRIEVAL_BODY_LEXICAL_WEIGHT = float(
    os.getenv("SLIDE_TUTOR_RETRIEVAL_BODY_LEXICAL_WEIGHT", "1.0")
)
RETRIEVAL_DENSE_WEIGHT = float(
    os.getenv("SLIDE_TUTOR_RETRIEVAL_DENSE_WEIGHT", "1.0")
)
MAX_CONTEXT_CHARS = 6000  # ~1500 tokens, safe for most local models
RETRIEVAL_SCHEMA_VERSION = "v2"
CHUNK_TEXT_VERSION = "title-clean-v1"
LEXICAL_INDEX_VERSION = "bm25-v1"
LEXICAL_TOKENIZER_VERSION = "unicode-cjk-bigram-v1"
RETRIEVAL_TITLE_INDEX_VERSION = "title-bm25-v2-dual-query"

# ── Index paths ──
INDEX_DIRNAME = "index"
UNDERSTANDING_DIRNAME = "understanding"

# Embedding model for semantic retrieval (sentence-transformers).
# Falls back to TF-IDF if the model is not installed or fails to load.
#
# Default: paraphrase-multilingual-MiniLM-L12-v2 — better for Chinese/Japanese/
# English retrieval.  Override with SLIDE_TUTOR_EMBEDDING_MODEL=all-MiniLM-L6-v2
# for an English-optimised lightweight model.
EMBEDDING_MODEL = os.getenv(
    "SLIDE_TUTOR_EMBEDDING_MODEL",
    "paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_MODEL_REVISION = os.getenv("SLIDE_TUTOR_EMBEDDING_MODEL_REVISION") or None
EMBEDDING_QUERY_PREFIX = os.getenv("SLIDE_TUTOR_EMBEDDING_QUERY_PREFIX", "")
EMBEDDING_PASSAGE_PREFIX = os.getenv("SLIDE_TUTOR_EMBEDDING_PASSAGE_PREFIX", "")
EMBEDDING_NORMALIZE = False
EMBEDDING_LOCAL_FILES_ONLY = os.getenv(
    "SLIDE_TUTOR_EMBEDDING_LOCAL_FILES_ONLY", "true",
).strip().lower() not in {"0", "false", "no"}
EMBEDDING_PREFLIGHT_TIMEOUT_SECONDS = float(
    os.getenv("SLIDE_TUTOR_EMBEDDING_PREFLIGHT_TIMEOUT_SECONDS", "15")
)
RETRIEVAL_EMBEDDING_BATCH_SIZE = int(
    os.getenv("SLIDE_TUTOR_RETRIEVAL_EMBEDDING_BATCH_SIZE", "32")
)

# ── Language config ──
DEFAULT_TARGET_LANGUAGE = "中文"

# ── Marker config ──
# Whether to auto-parse pages with Marker on first Q&A interaction.
# Set to False to disable Marker and always use PyMuPDF fallback.
#
# When Marker is enabled but not installed, the system silently falls back
# to PyMuPDF (PageUnderstanding.quality = "fallback").
MARKER_ENABLED = True
MARKER_LOCAL_FILES_ONLY = os.getenv(
    "SLIDE_TUTOR_MARKER_LOCAL_FILES_ONLY", "true",
).strip().lower() not in {"0", "false", "no"}
MARKER_BATCH_PAGES = max(
    1, int(os.getenv("SLIDE_TUTOR_MARKER_BATCH_PAGES", "8"))
)


def ensure_data_root() -> None:
    MODULES_DIR.mkdir(parents=True, exist_ok=True)
    PROJECTS_DIR.mkdir(parents=True, exist_ok=True)
