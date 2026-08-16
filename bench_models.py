"""Compare the current annotator models against lighter candidates.

Measures resident memory and latency for each, and prints the predicted
labels on sample Indonesian news text so label mappings can be checked
rather than assumed.
"""
import gc
import os
import time

import torch
from transformers import pipeline

torch.set_num_threads(2)  # match the 2-vCPU target box

SAMPLES = [
    "Presiden Prabowo Subianto menegaskan digitalisasi bantuan sosial telah berjalan di 153 kabupaten.",
    "Harga saham anjlok tajam setelah laporan keuangan mengecewakan, investor panik melepas kepemilikan.",
    "Saya sangat senang dengan pelayanan yang ramah dan cepat, produknya juga berkualitas bagus sekali.",
]


def rss_mb():
    try:
        import resource
        v = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return v / 1e6 if os.uname().sysname == "Darwin" else v / 1e3
    except Exception:
        return -1


def bench(tag, task, model, **kw):
    gc.collect()
    before = rss_mb()
    t0 = time.time()
    try:
        pipe = pipeline(task, model=model, tokenizer=model, **kw)
    except Exception as e:
        print(f"  {tag:<12} {model}\n     GAGAL memuat: {str(e)[:110]}")
        return
    load = time.time() - t0
    peak = rss_mb()

    # warmup
    pipe(SAMPLES[0], truncation=True, max_length=512)

    t0 = time.time()
    outs = [pipe(s, truncation=True, max_length=512) for s in SAMPLES]
    per = (time.time() - t0) / len(SAMPLES)

    print(f"  {tag:<12} {model}")
    print(f"     muat {load:5.1f}s | RSS puncak {peak:6.0f} MB (+{peak-before:.0f}) | {per*1000:6.0f} ms/teks")
    for s, o in zip(SAMPLES, outs):
        head = o[0] if isinstance(o, list) and o and isinstance(o[0], dict) else o
        print(f"     {s[:52]:<54} -> {head}")
    del pipe
    gc.collect()


print("=" * 78)
print("SENTIMENT")
print("=" * 78)
bench("sekarang", "sentiment-analysis", "masnasri-a/indobert-sentiment-analysis")
bench("kandidat", "sentiment-analysis", "savioruz/indobert-lite-p1-smsa")

print()
print("=" * 78)
print("EMOTION")
print("=" * 78)
bench("sekarang", "text-classification", "w11wo/indonesian-roberta-base-prdect-id")
bench("kandidat", "text-classification", "albarpambagio/distilbert-base-indonesian-finetuned-PRDECT-ID")
