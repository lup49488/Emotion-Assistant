"""Measure what an embedding model actually costs on this machine.

Estimates from parameter counts ignore runtime overhead, so before committing to
a model on a memory-constrained box, load it here and read the real numbers.

Usage:
    python tools/check_embedding_footprint.py intfloat/multilingual-e5-small
    python tools/check_embedding_footprint.py --cpu BAAI/bge-m3

Reports resident memory before and after loading, VRAM when the model lands on a
GPU, the embedding dimension (which determines whether existing FAISS indexes
must be rebuilt), and encode latency. Pass --cpu to measure what a CPU-only
server would see from a machine that has a GPU.
"""

from __future__ import annotations

import ctypes
import gc
import sys
import time
from pathlib import Path


def resident_mb() -> float:
    """Current resident set size in MB, without adding a psutil dependency.

    Reads /proc on Linux rather than ``resource.getrusage``, whose ru_maxrss is a
    high-water mark: that would make the "before load" reading wrong once an
    earlier model had already been resident.
    """
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024
    if sys.platform == "win32":
        class _Counters(ctypes.Structure):
            _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        kernel32 = ctypes.WinDLL("kernel32")
        psapi = ctypes.WinDLL("psapi")
        # Without an explicit restype the HANDLE comes back as a 32-bit int and
        # gets truncated, so the call silently reports zero.
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_Counters), ctypes.c_ulong,
        ]
        counters = _Counters()
        counters.cb = ctypes.sizeof(_Counters)
        if not psapi.GetProcessMemoryInfo(
            kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
        ):
            raise OSError(ctypes.get_last_error())
        return counters.WorkingSetSize / 1024 / 1024
    import resource  # macOS reports bytes.

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024


SAMPLES = [
    "我最近特别难过，觉得自己很没用。",
    "I feel really sad and worthless lately.",
    "TCP 和 UDP 有什么区别？",
    "What is the difference between TCP and UDP?",
]


def cuda_allocated_mb() -> float | None:
    """VRAM held by this process, or None when the model is not on a GPU.

    Without this a GPU run looks deceptively cheap: sentence-transformers moves
    the weights to CUDA automatically, so RSS never accounts for them.
    """
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.memory_allocated() / 1024 / 1024


def check(model_name: str, *, device: str | None = None) -> None:
    from sentence_transformers import SentenceTransformer

    before = resident_mb()
    vram_before = cuda_allocated_mb()
    start = time.perf_counter()
    model = SentenceTransformer(model_name, device=device)
    load_seconds = time.perf_counter() - start
    after_load = resident_mb()
    vram_after_load = cuda_allocated_mb()

    start = time.perf_counter()
    vectors = model.encode(SAMPLES, normalize_embeddings=True)
    encode_seconds = time.perf_counter() - start
    after_encode = resident_mb()
    vram_after_encode = cuda_allocated_mb()

    # Mirrors llm_providers.get_embedding_dimension: the method was renamed, and
    # the old name warns on newer sentence-transformers.
    dimension = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )

    print(f"\n=== {model_name} ===")
    print(f"  device              : {model.device}")
    print(f"  embedding dimension : {dimension}")
    print(f"  RSS before load     : {before:8.1f} MB")
    print(f"  RSS after load      : {after_load:8.1f} MB   (+{after_load - before:.1f})")
    print(f"  RSS after encode    : {after_encode:8.1f} MB   (+{after_encode - after_load:.1f})")
    if vram_after_load is not None:
        print(f"  VRAM before load    : {vram_before:8.1f} MB")
        print(f"  VRAM after load     : {vram_after_load:8.1f} MB   (+{vram_after_load - vram_before:.1f})")
        print(f"  VRAM after encode   : {vram_after_encode:8.1f} MB   (+{vram_after_encode - vram_after_load:.1f})")
    print(f"  load time           : {load_seconds:8.2f} s")
    print(f"  encode {len(SAMPLES)} texts     : {encode_seconds:8.3f} s")

    # A cross-lingual sanity check: the paired Chinese and English sentences
    # should score far higher against each other than against the unrelated pair.
    import numpy as np

    similarity = np.asarray(vectors) @ np.asarray(vectors).T
    emotional, technical, unrelated = (
        similarity[0][1], similarity[2][3], similarity[0][3],
    )
    print(f"  zh/en pair 1 (情绪)  : {emotional:.3f}")
    print(f"  zh/en pair 2 (技术)  : {technical:.3f}")
    print(f"  unrelated (情绪/技术): {unrelated:.3f}")
    # Raw cosine is not comparable across models: some families place every pair
    # in a narrow high band, so a bigger number can still mean weaker separation.
    # Margin over the unrelated pair is what a retrieval threshold actually sees.
    print(f"  margin, 情绪 - 无关  : {emotional - unrelated:+.3f}")
    print(f"  margin, 技术 - 无关  : {technical - unrelated:+.3f}")
    print(
        "  NOTE: retrieval thresholds are calibrated per model. An unrelated "
        f"baseline of {unrelated:.3f} means KNOWLEDGE/STYLE_RETRIEVAL_THRESHOLD "
        "must be re-tuned before this model is used."
    )

    del model
    gc.collect()


def main() -> int:
    args = sys.argv[1:]
    # A CPU-only server cannot fall back to a GPU, so measuring the deployment
    # target from a workstation means pinning the device explicitly.
    device = "cpu" if "--cpu" in args else None
    names = [arg for arg in args if not arg.startswith("--")]
    if not names:
        print(__doc__)
        return 1
    print(f"baseline RSS (interpreter + imports): {resident_mb():.1f} MB")
    for name in names:
        try:
            check(name, device=device)
        except Exception as exc:
            print(f"\n=== {name} ===\n  FAILED: {exc}")
    print(
        "\nNote: freed memory is not always returned to the OS, so loading several "
        "models in one run inflates later readings. Measure one model per run for "
        "a clean number."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
