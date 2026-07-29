import logging
import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
from env_loader import load_project_env

# 测试环境通过该开关跳过加载本机 .env，避免开发者的生产配置
# （如 API_TRUSTED_HOSTS、API_PUBLIC_MODE）泄漏进测试并影响结果。
if os.getenv("CHATBOT_SKIP_DOTENV", "").strip().lower() in {"1", "true", "yes"}:
    LOADED_ENV_FILES: list[Path] = []
else:
    LOADED_ENV_FILES = load_project_env(BASE_DIR)

_logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        _logger.warning("环境变量 %s=%r 不是合法的浮点数，使用默认值 %s", name, raw, default)
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        _logger.warning("环境变量 %s=%r 不是合法的整数，使用默认值 %s", name, raw, default)
        return default

USERS_DIR = BASE_DIR / "users"
KNOWLEDGE_DIR = BASE_DIR / "knowledge_base"
KNOWLEDGE_DOCS_DIR = KNOWLEDGE_DIR / "documents"
KNOWLEDGE_INDEX_PATH = KNOWLEDGE_DIR / "knowledge.index"
KNOWLEDGE_CHUNKS_PATH = KNOWLEDGE_DIR / "chunks.json"
STYLE_DIR = BASE_DIR / "style_base"
STYLE_DOCS_DIR = STYLE_DIR / "documents"
STYLE_INDEX_PATH = STYLE_DIR / "style.index"
STYLE_CHUNKS_PATH = STYLE_DIR / "chunks.json"

SHORT_TERM_LIMIT              = 10
MID_TERM_LIMIT                = 50
LONG_TERM_EXPIRY_DAYS         = 30
INTEREST_SIMILARITY_THRESHOLD = 0.90
INTEREST_RETRIEVAL_THRESHOLD  = 0.75

SCORE_EMOTION_MULTIPLIER     = 2.0
SCORE_NEGATIVE_BONUS         = 1.0
SCORE_MEMORY_KEYWORD_BONUS   = 2.0
SCORE_PERSONAL_KEYWORD_BONUS = 1.5
SCORE_LONG_TERM_THRESHOLD     = 4.0
SCORE_MID_TERM_THRESHOLD      = 2.0

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-small-zh-v1.5")
CHAT_MODEL_NAME      = os.getenv("CHAT_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
HF_TOKEN             = os.getenv("HF_TOKEN") or None

# Multilingual emotion classification runs directly on the original text, so
# Chinese and other non-English input no longer needs a translation fallback.
EMOTION_MODEL_NAME = os.getenv(
    "EMOTION_MODEL_NAME", "tabularisai/multilingual-emotion-classification"
)
EMOTION_CONFIDENCE_THRESHOLD = min(
    1.0, max(0.0, _env_float("EMOTION_CONFIDENCE_THRESHOLD", 0.60))
)
EMOTION_MODEL_MULTI_LABEL = os.getenv("EMOTION_MODEL_MULTI_LABEL", "true").lower() == "true"
# 多语言情绪模型没有独立的 anxiety 标签。开启后，当模型判定为 fear 且文本带有
# 明显的焦虑线索（对未来的担忧、反刍、躯体化）时，细分为 anxiety。
EMOTION_ANXIETY_REFINEMENT = os.getenv("EMOTION_ANXIETY_REFINEMENT", "true").lower() == "true"

# Optional local multilingual NLI supplement for the rule-based safety layer.
# It is disabled by default because the model adds a sizeable CPU/RAM footprint.
SAFETY_SEMANTIC_ENABLED = os.getenv("SAFETY_SEMANTIC_ENABLED", "false").lower() == "true"
SAFETY_SEMANTIC_PRELOAD = os.getenv("SAFETY_SEMANTIC_PRELOAD", "true").lower() == "true"
SAFETY_SEMANTIC_MODEL_NAME = os.getenv(
    "SAFETY_SEMANTIC_MODEL_NAME", "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
)
SAFETY_SEMANTIC_THRESHOLD = min(
    1.0, max(0.0, _env_float("SAFETY_SEMANTIC_THRESHOLD", 0.78))
)

DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai_compatible")
DEFAULT_API_MODEL = os.getenv("LLM_API_MODEL", "deepseek-chat")
DEFAULT_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.deepseek.com")
DEFAULT_TEMPERATURE = _env_float("LLM_TEMPERATURE", 0.8)
DEFAULT_TOP_P = _env_float("LLM_TOP_P", 0.9)
DEFAULT_MAX_NEW_TOKENS = _env_int("LLM_MAX_NEW_TOKENS", 300)
# JSON array of server-managed API fallback targets. Credentials remain in the
# provider-specific environment variables and are never accepted here.
LLM_FALLBACKS_JSON = os.getenv("LLM_FALLBACKS_JSON", "").strip()

# API 稳定性与成本控制。限额默认关闭，单价由实际使用的服务商账单填写。
API_REQUEST_TIMEOUT_SECONDS = _env_float("API_REQUEST_TIMEOUT_SECONDS", 60.0)
API_MAX_RETRIES = max(0, _env_int("API_MAX_RETRIES", 2))
API_RETRY_BACKOFF_SECONDS = max(0.0, _env_float("API_RETRY_BACKOFF_SECONDS", 1.0))
API_MAX_REQUESTS_PER_MINUTE = max(0, _env_int("API_MAX_REQUESTS_PER_MINUTE", 0))
API_DAILY_BUDGET_USD = max(0.0, _env_float("API_DAILY_BUDGET_USD", 0.0))
API_MONTHLY_BUDGET_USD = max(0.0, _env_float("API_MONTHLY_BUDGET_USD", 0.0))
API_INPUT_COST_PER_1M_TOKENS = max(0.0, _env_float("API_INPUT_COST_PER_1M_TOKENS", 0.0))
API_OUTPUT_COST_PER_1M_TOKENS = max(0.0, _env_float("API_OUTPUT_COST_PER_1M_TOKENS", 0.0))

# RAG release gate. Disabled by default until a representative evaluation set
# has been prepared; enabling it keeps the last published index on failure.
RAG_RELEASE_GATE_ENABLED = os.getenv("RAG_RELEASE_GATE_ENABLED", "false").lower() == "true"
RAG_RELEASE_MIN_CASES = max(1, _env_int("RAG_RELEASE_MIN_CASES", 5))
RAG_RELEASE_MIN_PASS_RATE = min(100.0, max(0.0, _env_float("RAG_RELEASE_MIN_PASS_RATE", 80.0)))
RAG_RELEASE_MIN_SOURCE_RECALL = min(100.0, max(0.0, _env_float("RAG_RELEASE_MIN_SOURCE_RECALL", 80.0)))
RAG_RELEASE_MIN_KEYWORD_COVERAGE = min(100.0, max(0.0, _env_float("RAG_RELEASE_MIN_KEYWORD_COVERAGE", 80.0)))

# Persistent operational telemetry. Event bodies and user content are never stored.
OBSERVABILITY_RETENTION_DAYS = max(1, _env_int("OBSERVABILITY_RETENTION_DAYS", 90))

# Public deployment hardening. Keep false for local development; set true only
# behind HTTPS/Tunnel after configuring the matching API_* values.
API_PUBLIC_MODE = os.getenv("API_PUBLIC_MODE", "false").lower() == "true"
API_TRUSTED_HOSTS = os.getenv("API_TRUSTED_HOSTS", "localhost,127.0.0.1,[::1],testserver")
API_ENABLE_DOCS = os.getenv("API_ENABLE_DOCS", "true").lower() == "true"
API_MAX_REQUEST_BYTES = max(1_024, _env_int("API_MAX_REQUEST_BYTES", 21 * 1024 * 1024))
API_AUTH_MAX_ATTEMPTS = max(1, _env_int("API_AUTH_MAX_ATTEMPTS", 8))
API_AUTH_WINDOW_SECONDS = max(60, _env_int("API_AUTH_WINDOW_SECONDS", 900))
API_TRUST_PROXY_HEADERS = os.getenv("API_TRUST_PROXY_HEADERS", "false").lower() == "true"

# Administrator-only operations dashboard and alert thresholds.
API_OPERATIONS_USER_IDS = os.getenv("API_OPERATIONS_USER_IDS", "")
# Comma-separated user IDs allowed to publish and manage the shared RAG corpus.
# When omitted, the operations allowlist is used as the secure default.
API_RAG_ADMIN_USER_IDS = os.getenv("API_RAG_ADMIN_USER_IDS", "")
OPS_ALERT_MIN_REQUESTS = max(1, _env_int("OPS_ALERT_MIN_REQUESTS", 10))
OPS_ALERT_HTTP_FAILURE_RATE = min(100.0, max(0.0, _env_float("OPS_ALERT_HTTP_FAILURE_RATE", 10.0)))
OPS_ALERT_AVERAGE_LATENCY_MS = max(1, _env_int("OPS_ALERT_AVERAGE_LATENCY_MS", 5000))
OPS_ALERT_PROVIDER_FAILURES = max(1, _env_int("OPS_ALERT_PROVIDER_FAILURES", 3))
OPS_ALERT_JOB_FAILURES = max(1, _env_int("OPS_ALERT_JOB_FAILURES", 1))

LOCAL_MODEL_DTYPE = os.getenv("LOCAL_MODEL_DTYPE", "auto").lower()
LOCAL_MODEL_ATTN_IMPLEMENTATION = os.getenv("LOCAL_MODEL_ATTN_IMPLEMENTATION", "").strip() or None
LOCAL_MODEL_LOW_CPU_MEM_USAGE = os.getenv("LOCAL_MODEL_LOW_CPU_MEM_USAGE", "true").lower() == "true"
LOCAL_MODEL_COMPILE = os.getenv("LOCAL_MODEL_COMPILE", "false").lower() == "true"
LOCAL_MODEL_CPU_THREADS = _env_int("LOCAL_MODEL_CPU_THREADS", 0)
LOCAL_MODEL_MEMORY_CHECK = os.getenv("LOCAL_MODEL_MEMORY_CHECK", "true").lower() == "true"
LOCAL_MODEL_MEMORY_SAFETY_FACTOR = _env_float("LOCAL_MODEL_MEMORY_SAFETY_FACTOR", 1.25)
LOCAL_MODEL_PARAMETER_COUNT_B = _env_float("LOCAL_MODEL_PARAMETER_COUNT_B", 0.0)
LOCAL_MODEL_ALLOW_CPU_OFFLOAD = os.getenv("LOCAL_MODEL_ALLOW_CPU_OFFLOAD", "false").lower() == "true"
LOCAL_MODEL_GPU_MAX_MEMORY_GB = _env_float("LOCAL_MODEL_GPU_MAX_MEMORY_GB", 5.5)
LOCAL_MODEL_CPU_MAX_MEMORY_GB = _env_float("LOCAL_MODEL_CPU_MAX_MEMORY_GB", 12.0)

KNOWLEDGE_ENABLED = os.getenv("KNOWLEDGE_ENABLED", "false").lower() == "true"
KNOWLEDGE_TOP_K = _env_int("KNOWLEDGE_TOP_K", 4)
KNOWLEDGE_RETRIEVAL_THRESHOLD = _env_float("KNOWLEDGE_RETRIEVAL_THRESHOLD", 0.35)
KNOWLEDGE_CHUNK_SIZE = _env_int("KNOWLEDGE_CHUNK_SIZE", 700)
KNOWLEDGE_CHUNK_OVERLAP = _env_int("KNOWLEDGE_CHUNK_OVERLAP", 120)
KNOWLEDGE_CANDIDATE_MULTIPLIER = _env_int("KNOWLEDGE_CANDIDATE_MULTIPLIER", 4)
KNOWLEDGE_MAX_PER_SOURCE = _env_int("KNOWLEDGE_MAX_PER_SOURCE", 2)
KNOWLEDGE_MAX_CONTEXT_CHARS = _env_int("KNOWLEDGE_MAX_CONTEXT_CHARS", 4000)
KNOWLEDGE_MIN_CHUNK_CHARS = _env_int("KNOWLEDGE_MIN_CHUNK_CHARS", 40)

STYLE_ENABLED = os.getenv("STYLE_ENABLED", "true").lower() == "true"
STYLE_TOP_K = _env_int("STYLE_TOP_K", 3)
STYLE_RETRIEVAL_THRESHOLD = _env_float("STYLE_RETRIEVAL_THRESHOLD", 0.30)
# 按文件名前缀过滤时，需要检索更大的候选池，否则最近邻可能全属于其他风格。
STYLE_PREFIX_CANDIDATE_MULTIPLIER = max(1, _env_int("STYLE_PREFIX_CANDIDATE_MULTIPLIER", 8))
STYLE_CHUNK_SIZE = _env_int("STYLE_CHUNK_SIZE", 900)
STYLE_CHUNK_OVERLAP = _env_int("STYLE_CHUNK_OVERLAP", 120)

# 同时允许的最大会话缓存数（防止长期运行的服务内存无限增长）
MAX_CACHED_SESSIONS = _env_int("MAX_CACHED_SESSIONS", 1000)

INTEREST_PATTERNS = [
    "我喜欢", "我爱", "我讨厌", "我不喜欢", "我想成为",
    "我的梦想", "我的目标", "我习惯",
    "I like", "I love", "I enjoy", "I hate",
    "I prefer", "My dream is", "I want to become", "I always",
]
MEMORY_KEYWORDS   = ["喜欢", "讨厌", "目标", "梦想", "计划", "习惯", "我想", "我不喜欢"]
PERSONAL_KEYWORDS = ["我是", "我叫", "我的名字", "我今年", "我来自"]

# 合法 user_id 字符集：字母、数字、下划线、短横线，1-128 字符
_USER_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-]{1,128}$")
