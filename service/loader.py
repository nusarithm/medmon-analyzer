"""Shared pipeline loader.

The three services all wanted the same thing: load a HF pipeline once, keep
it, and run it with sane settings for a small CPU box. That lives here so
truncation, thread count and quantisation are set in one place.

Model ids come from the environment, so swapping in a lighter model is a
config change rather than a code change.
"""
import os

import torch
from transformers import pipeline

# A 2-vCPU box does worse when torch oversubscribes threads.
torch.set_num_threads(int(os.getenv("TORCH_THREADS", "2")))

# int8 dynamic quantisation cuts the linear weights to a quarter and is 2-3x
# faster - but only where the CPU has AVX2. Without it the fbgemm kernels do
# not fall back, they abort the process with SIGILL (verified on the homelab
# VM, whose QEMU CPU model exposes no AVX at all), so check before enabling.
QUANTIZE = os.getenv("QUANTIZE", "0") == "1"


def _quantization_supported() -> bool:
    try:
        with open("/proc/cpuinfo") as f:
            return any(
                "avx2" in line.split() for line in f if line.startswith("flags")
            )
    except OSError:
        return True  # not Linux/x86 (e.g. Apple silicon): qnnpack handles it


if QUANTIZE and not _quantization_supported():
    print("QUANTIZE=1 ignored: CPU has no AVX2, fbgemm would crash the process")
    QUANTIZE = False

# 512 is the token limit of these BERT-family models, applied by the tokenizer.
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "512"))

_CACHE: dict = {}


def get_pipeline(task: str, model: str, **kwargs):
    """Return a cached pipeline, loading it on first use."""
    key = (task, model)
    if key not in _CACHE:
        pipe = pipeline(
            task,
            model=model,
            tokenizer=model,
            device=-1,  # CPU
            **kwargs,
        )
        if QUANTIZE:
            pipe.model = torch.quantization.quantize_dynamic(
                pipe.model, {torch.nn.Linear}, dtype=torch.qint8
            )
        # transformers 5 dropped call-time truncation for token-classification;
        # that pipeline truncates from the tokenizer's own limit instead. Pin it
        # here so every task cuts at the same place.
        pipe.tokenizer.model_max_length = MAX_LENGTH
        _CACHE[key] = pipe
    return _CACHE[key]


def run(task: str, model: str, text, **kwargs):
    """Run a pipeline with truncation handled by the tokenizer.

    `text` may be a single string or a list - a list is batched, which is
    several times faster than one call per document.
    """
    pipe = get_pipeline(task, model, **kwargs)
    if task in ("ner", "token-classification"):
        # Passing truncation here raises TypeError: this pipeline forwards no
        # tokenizer kwargs. model_max_length above already caps the input.
        return pipe(text)
    return pipe(text, truncation=True, max_length=MAX_LENGTH)


def loaded_models() -> list:
    """Which pipelines are currently resident, for /health."""
    return [f"{task}:{model}" for task, model in _CACHE]
