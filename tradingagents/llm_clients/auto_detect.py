"""Auto-detect the best LLM backend based on available hardware.

Logic:
  GPU present (nvidia-smi or CUDA) AND Ollama reachable → ollama + Qwen
  Otherwise                                              → google + Gemini 2.5 Flash

Qwen model selection is based on detected VRAM:
  < 10 GB   → deep: qwen2.5:7b   / quick: qwen2.5:7b
  10–14 GB  → deep: qwen2.5:14b  / quick: qwen2.5:7b
  14–22 GB  → deep: qwen2.5:32b  / quick: qwen2.5:14b
  22+ GB    → deep: qwen2.5:72b  / quick: qwen2.5:14b
"""

import subprocess
import urllib.request
import urllib.error
from typing import TypedDict

OLLAMA_HOST = "http://127.0.0.1:11434"
OLLAMA_TIMEOUT = 2  # seconds

_GEMINI_CONFIG = {
    "llm_provider": "google",
    "deep_think_llm": "gemini-2.5-flash",
    "quick_think_llm": "gemini-2.5-flash",
}

# VRAM thresholds in GiB → (deep_model, quick_model)
_VRAM_TIERS: list[tuple[float, str, str]] = [
    (22.0, "qwen2.5:72b",  "qwen2.5:14b"),
    (14.0, "qwen2.5:32b",  "qwen2.5:14b"),
    (10.0, "qwen2.5:14b",  "qwen2.5:7b"),
    ( 0.0, "qwen2.5:7b",   "qwen2.5:7b"),
]


class ProviderConfig(TypedDict):
    llm_provider: str
    deep_think_llm: str
    quick_think_llm: str


def gpu_vram_gb() -> float:
    """Return total VRAM of the first NVIDIA GPU in GiB, or 0.0 if none found."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
        mib = float(out.decode().strip().splitlines()[0])
        return mib / 1024
    except Exception:
        pass

    # Fallback: PyTorch CUDA (may not be installed)
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            mib = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
            return mib / 1024
    except Exception:
        pass

    return 0.0


def ollama_reachable(host: str = OLLAMA_HOST, timeout: int = OLLAMA_TIMEOUT) -> bool:
    """Return True if the Ollama API is responding."""
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=timeout)
        return True
    except Exception:
        return False


def _qwen_config(vram_gb: float) -> ProviderConfig:
    for threshold, deep, quick in _VRAM_TIERS:
        if vram_gb >= threshold:
            return {"llm_provider": "ollama", "deep_think_llm": deep, "quick_think_llm": quick}
    return {"llm_provider": "ollama", "deep_think_llm": "qwen2.5:7b", "quick_think_llm": "qwen2.5:7b"}


def detect_provider() -> ProviderConfig:
    """Return the best available provider config for this machine."""
    vram = gpu_vram_gb()
    if vram > 0 and ollama_reachable():
        return _qwen_config(vram)
    return _GEMINI_CONFIG  # type: ignore[return-value]


def detect_provider_verbose() -> tuple[ProviderConfig, str]:
    """Same as detect_provider() but also returns a human-readable reason string."""
    vram = gpu_vram_gb()
    if vram <= 0:
        return _GEMINI_CONFIG, "no NVIDIA GPU detected → Gemini"  # type: ignore[return-value]
    if not ollama_reachable():
        return _GEMINI_CONFIG, f"GPU found ({vram:.0f} GB VRAM) but Ollama not reachable → Gemini"  # type: ignore[return-value]
    cfg = _qwen_config(vram)
    reason = (
        f"GPU found ({vram:.0f} GB VRAM), Ollama running → "
        f"{cfg['deep_think_llm']} / {cfg['quick_think_llm']}"
    )
    return cfg, reason
