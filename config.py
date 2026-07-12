import os
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
from env_loader import load_project_env

LOADED_ENV_FILES = load_project_env(BASE_DIR)
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

DEFAULT_LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local_hf")
DEFAULT_API_MODEL = os.getenv("LLM_API_MODEL", "deepseek-chat")
DEFAULT_API_BASE_URL = os.getenv("LLM_API_BASE_URL", "https://api.deepseek.com")
DEFAULT_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.8"))
DEFAULT_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))
DEFAULT_MAX_NEW_TOKENS = int(os.getenv("LLM_MAX_NEW_TOKENS", "300"))

KNOWLEDGE_ENABLED = os.getenv("KNOWLEDGE_ENABLED", "false").lower() == "true"
KNOWLEDGE_TOP_K = int(os.getenv("KNOWLEDGE_TOP_K", "4"))
KNOWLEDGE_RETRIEVAL_THRESHOLD = float(os.getenv("KNOWLEDGE_RETRIEVAL_THRESHOLD", "0.35"))
KNOWLEDGE_CHUNK_SIZE = int(os.getenv("KNOWLEDGE_CHUNK_SIZE", "700"))
KNOWLEDGE_CHUNK_OVERLAP = int(os.getenv("KNOWLEDGE_CHUNK_OVERLAP", "120"))

STYLE_ENABLED = os.getenv("STYLE_ENABLED", "false").lower() == "true"
STYLE_TOP_K = int(os.getenv("STYLE_TOP_K", "3"))
STYLE_RETRIEVAL_THRESHOLD = float(os.getenv("STYLE_RETRIEVAL_THRESHOLD", "0.30"))
STYLE_CHUNK_SIZE = int(os.getenv("STYLE_CHUNK_SIZE", "900"))
STYLE_CHUNK_OVERLAP = int(os.getenv("STYLE_CHUNK_OVERLAP", "120"))

# 同时允许的最大会话缓存数（防止长期运行的服务内存无限增长）
MAX_CACHED_SESSIONS = int(os.getenv("MAX_CACHED_SESSIONS", "1000"))

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

