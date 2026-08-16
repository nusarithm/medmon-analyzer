# Indonesian News Annotation Pipeline

Batch worker, no HTTP API. It reads un-annotated news from Elasticsearch
newest-first, runs three models over each document, and writes the result
back into the same document under `annotate` — the field
`media-monitoring-svc` aggregates on.

- **Emotion**: `w11wo/indonesian-roberta-base-prdect-id` → Happy, Sadness, Anger, Love, Fear
- **Sentiment**: `masnasri-a/indobert-sentiment-analysis` → negatif, netral, positif
- **NER**: `cahya/bert-base-indonesian-NER` → PER, ORG, LOC

## Cara kerja

Elasticsearch *is* the queue. The selector is `must_not exists annotate`
sorted by `scraped_at` desc, so:

- a document leaves the queue as soon as it is annotated — no cursor, no offset,
- a crash costs at most one batch; just start the process again,
- new scraped articles are picked up first, backfill drains behind them.

The in-process `queue.Queue(maxsize=2)` only overlaps the ES fetch with model
inference; ids already in flight are excluded from the next fetch so a batch is
never handed out twice.

```
ES search (newest un-annotated) ──► queue ──► emotion + sentiment + NER ──► bulk update annotate
```

## Instalasi

```bash
pip install -r requirements.txt
cp .env.example .env   # isi kredensial Elasticsearch
```

## Menjalankan

```bash
python main.py          # daemon: polls forever, sleeps IDLE_SLEEP when idle
python main.py --once   # backfill: exits when nothing is left un-annotated
python test_pipeline.py # self-check (no ES, no model loading)
```

Stop with Ctrl+C or SIGTERM: the current batch finishes writing, then it exits.

## Hasil di Elasticsearch

```json
"annotate": {
  "emotion":   {"label": "Anger", "score": 0.91},
  "sentiment": {"label": "negatif", "score": 0.88},
  "entities":  [{"entity_group": "PER", "word": "Prabowo", "score": 0.99, "start": 10, "end": 17}],
  "annotated_at": "2026-08-16T12:00:00+00:00"
}
```

A document that cannot be annotated (empty body, model error) gets
`{"error": "...", "annotated_at": ...}` instead — the marker is what keeps a
broken document from being retried forever.

Dengan `NER_ENABLED=0`, `entities` selalu kosong dan dokumen ditandai
`"ner_skipped": true`. Untuk mengulang NER-nya nanti, hapus field `annotate`
pada dokumen bertanda itu lalu jalankan lagi:

```json
POST online-news-*/_update_by_query
{"query": {"term": {"annotate.ner_skipped": true}},
 "script": "ctx._source.remove('annotate')"}
```

## Deploy ke home server (menglabs, 192.168.8.100)

Box: VM 101 di Proxmox 192.168.8.188 (host: i5-7500T, 4 core, AVX2). Guest
4 vCPU, `cpu: host`, `cpuunits: 50`, 7.8 GB RAM tanpa swap, Ubuntu 24.04,
Python 3.12, tanpa GPU. Elasticsearch di VM 100 = 192.168.8.104:9200.

`cpuunits: 50` (default 100) menahan bobot VM ini di bawah VM Elasticsearch:
host cuma punya 4 core dan annotator memakai semuanya, jadi tanpa itu query ES
ikut melambat saat backfill jalan.

```bash
ssh nasri@192.168.8.100
sudo apt-get install -y python3.12-venv          # ensurepip tidak ikut default
python3 -m venv ~/annotator-venv
~/annotator-venv/bin/pip install -r ~/annotator/requirements.txt
cp ~/annotator/.env.example ~/annotator/.env     # isi kredensial ES
sudo cp ~/annotator/annotator.service /etc/systemd/system/
sudo systemctl enable --now annotator
journalctl -u annotator -f
```

Angka terukur di box itu, dokumen asli dari index, 512 token + NER on,
`BATCH_SIZE=8`, `TORCH_THREADS=4`, `QUANTIZE=1`: **~0.9 s/dokumen**, RSS ~2.0 GB.
Batch 16 tidak lebih cepat dari 8.

Riwayat tuning, karena selisihnya besar dan gampang hilang:

| konfigurasi | s/dokumen |
|---|---|
| 2 vCPU, `cpu: x86-64-v2-AES` (tanpa AVX), fp32 | 5.5 |
| ...+ 256 token, NER off | 2.2 |
| 4 vCPU, `cpu: host` (AVX2), int8, 256 token, NER off | 0.30 |
| 4 vCPU, `cpu: host`, int8, 512 token, NER on | ~0.9 |

Yang paling berpengaruh bukan jumlah vCPU, tapi **CPU type**: `x86-64-v2-AES`
tidak punya AVX sama sekali, jadi torch jalan di kernel paling lambat dan
`QUANTIZE=1` mematikan proses dengan `Illegal instruction (core dumped)`
(fbgemm butuh AVX2). Dengan `cpu: host` keduanya hilang. `loader.py` tetap
memeriksa AVX2 dan mengabaikan `QUANTIZE=1` kalau tidak ada, supaya pindah ke
box lain tidak berujung crash.

## Konfigurasi

Semua lewat `.env` (lihat `.env.example`). Yang sering diubah:

| Variabel | Default | Keterangan |
|---|---|---|
| `ELASTICSEARCH_INDEX` | `online-news-*` | index pattern yang dibaca |
| `BATCH_SIZE` | `4` | dokumen per pass model |
| `IDLE_SLEEP` | `60` | detik tidur saat tidak ada kerjaan |
| `SORT_FIELD` | `scraped_at` | urutan terbaru → terlama |
| `NER_ENABLED` | `1` | `0` untuk hemat memori di box kecil |
| `TORCH_THREADS` | `2` | samakan dengan jumlah vCPU |
| `QUANTIZE` | `0` | `1` untuk int8 — **hanya di CPU ber-AVX2** |

## Struktur

```
annotator/
├── main.py            # pipeline: fetch → queue → annotate → write back
├── annotator.service  # systemd unit untuk home server
├── test_pipeline.py   # self-check untuk query & batch-failure logic
├── service/
│   ├── loader.py      # cached HF pipelines, truncation, threads, quantisation
│   ├── emotion.py
│   ├── sentiment.py
│   └── ner.py
└── bench_models.py    # bandingkan kandidat model (memori & latensi)
```

## Penggunaan programmatic

```python
from service.emotion import predict_emotion
from service.sentiment import analyze_sentiment
from service.ner import extract_entities

predict_emotion("Saya sangat bahagia", top_k=2)
analyze_sentiment("Produk ini bagus sekali")
extract_entities("Joko Widodo di Jakarta", min_score=0.8)
```
