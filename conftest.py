"""
conftest.py
在任何测试模块导入 chatbot.py 之前，把 torch / langdetect / goemotions_local /
numpy 等重依赖替换为轻量 stub，注入到 sys.modules。

这样测试环境不需要真的安装 GPU 版 torch、下载 Qwen 模型权重，
也能验证所有不依赖大模型推理的纯逻辑函数。

若本机已经装有这些库（比如在真实部署环境跑测试），stub 会被跳过，
直接用真实库，不影响正确性。
"""

import sys
import types

import pytest


@pytest.fixture(autouse=True)
def _isolate_storage_backend(monkeypatch):
    """Keep legacy tests independent from a developer's local SQLite settings."""
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    monkeypatch.delenv("SQLITE_DATABASE_PATH", raising=False)


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
