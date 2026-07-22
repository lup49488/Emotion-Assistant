"""
conftest.py
在任何测试模块导入 chatbot.py 之前，把 torch / langdetect / goemotions_local /
numpy 等重依赖替换为轻量 stub，注入到 sys.modules。

这样测试环境不需要真的安装 GPU 版 torch、下载 Qwen 模型权重，
也能验证所有不依赖大模型推理的纯逻辑函数。

若本机已经装有这些库（比如在真实部署环境跑测试），stub 会被跳过，
直接用真实库，不影响正确性。
"""

import os
import sys
import types

import pytest

# 必须在任何项目模块（尤其是 config）被导入之前设置：让测试忽略本机 .env，
# 使测试在干净的默认配置下运行，不受开发者部署配置影响。
os.environ.setdefault("CHATBOT_SKIP_DOTENV", "1")


@pytest.fixture(autouse=True)
def _isolate_storage_backend(monkeypatch):
    """Keep legacy tests independent from a developer's local SQLite settings."""
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    monkeypatch.delenv("SQLITE_DATABASE_PATH", raising=False)


@pytest.fixture(autouse=True)
def _reset_session_cache():
    """chatbot.session_store 是进程级单例，会把每个用户的 SessionState 缓存在内存里。

    测试通常只把 session_store.USERS_DIR 指到 tmp_path，但内存缓存不会跟着清空，
    导致一个测试写入的用户状态（对话、稳定资料等）泄漏到之后的测试。
    这里在每个测试结束后清空缓存，保证测试之间互不影响。
    """
    yield
    chatbot_module = sys.modules.get("chatbot")
    store = getattr(chatbot_module, "session_store", None)
    if store is None:
        return
    with store._registry_lock:
        store._sessions.clear()
        store._active_counts.clear()
        store._locks.clear()


def _install_stub(name: str, **attrs):
    """如果模块未安装，则注册一个同名 stub 到 sys.modules。"""
    if name in sys.modules:
        return
    try:
        __import__(name)
        return  # 真实库已存在，不覆盖
    except ImportError:
        pass

    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    sys.modules[name] = module


# ── torch stub ──────────────────────────────────────────
def _torch_cuda_is_available():
    return False


_install_stub(
    "torch",
    cuda=types.SimpleNamespace(is_available=_torch_cuda_is_available),
    bfloat16="bfloat16-stub",
    inference_mode=lambda: _NullContext(),
)


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


# torch.inference_mode 需要是一个可调用的上下文管理器工厂
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules["torch"].cuda = types.SimpleNamespace(is_available=lambda: False)
sys.modules["torch"].bfloat16 = "bfloat16-stub"
sys.modules["torch"].inference_mode = lambda: _NullContext()


# ── langdetect stub ─────────────────────────────────────
def _detect_stub(text: str) -> str:
    return "en"


_install_stub("langdetect", detect=_detect_stub)


# ── goemotions_local stub ───────────────────────────────
def _predict_zh_stub(text: str):
    return ("neutral", 0.5)


def _predict_en_stub(text: str):
    return ("neutral", 0.5)


_install_stub(
    "goemotions_local",
    predict_emotion_zh=_predict_zh_stub,
    predict_emotion_en=_predict_en_stub,
)


# ── numpy：通常环境里有，若没有也 stub 最小子集 ──────────
def _install_numpy_stub():
    if "numpy" in sys.modules:
        return
    try:
        __import__("numpy")
        return
    except ImportError:
        pass

    class _FakeArray(list):
        @property
        def size(self):
            return len(self)

    def asarray(data, dtype=None):
        return _FakeArray(data)

    module = types.ModuleType("numpy")
    module.asarray = asarray
    module.float32 = "float32-stub"
    sys.modules["numpy"] = module


_install_numpy_stub()
