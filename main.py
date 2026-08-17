"""Annotation pipeline.

Pulls un-annotated news from Elasticsearch newest-first, runs emotion,
sentiment and NER over them in batches, writes the result back into the
document under `annotate` - the field `media-monitoring-svc` aggregates on.

There is no HTTP surface: Elasticsearch itself is the work queue. A document
leaves the queue the moment its `annotate` field exists, so the pipeline is
resumable after a crash and needs no external broker. The in-process
`queue.Queue` only lets the ES fetch overlap with model inference.

    python main.py           # daemon: keep polling for new documents
    python main.py --once    # backfill: exit when nothing is left
"""
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timezone

from dotenv import load_dotenv
from elasticsearch import Elasticsearch, helpers

from service.emotion import predict_emotion_batch
from service.ner import ENABLED as NER_ON, extract_entities_batch
from service.sentiment import analyze_sentiment

load_dotenv()

INDEX = os.getenv("ELASTICSEARCH_INDEX", "online-news-*")
SORT_FIELD = os.getenv("SORT_FIELD", "scraped_at")
# 4, not 16: on the 2-vCPU target box a batch of 8 was no faster per document
# than a batch of 4 (~1.9 s/doc either way), and a smaller batch means less
# work repeated after a crash and a shorter shutdown wait.
BATCH = int(os.getenv("BATCH_SIZE", "4"))
IDLE_SLEEP = int(os.getenv("IDLE_SLEEP", "60"))
NER_MIN_SCORE = float(os.getenv("NER_MIN_SCORE", "0.8"))
# ponytail: cheap guard so the tokenizer does not chew through 100 KB bodies.
# The models cut at 512 *tokens* (~2000 chars of Indonesian) anyway, and the
# lede is at the front - past this the tokenizer just burns CPU for nothing.
MAX_CHARS = int(os.getenv("MAX_CHARS", "2500"))
# Which fields hold the text to annotate, in order. News articles carry
# title+body; a Threads post carries a single `text`. Everything else about
# the pipeline is identical, so this is a setting rather than a second script.
TEXT_FIELDS = [f.strip() for f in os.getenv("TEXT_FIELDS", "title,body").split(",") if f.strip()]

es = Elasticsearch(
    [os.getenv("ELASTICSEARCH_HOST", "http://localhost:9200")],
    http_auth=(
        os.getenv("ELASTICSEARCH_USERNAME", ""),
        os.getenv("ELASTICSEARCH_PASSWORD", ""),
    ),
    verify_certs=False,
    timeout=60,
)

stop = threading.Event()
work = queue.Queue(maxsize=2)  # batches fetched ahead of the models
_inflight = set()  # ids already queued but not yet written back
_inflight_lock = threading.Lock()


def log(msg: str):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


def build_query(skip_ids) -> dict:
    """Newest un-annotated documents first.

    Annotated documents drop out of this query, so the pipeline always reads
    page 0 - no cursor to keep, nothing to resume. `skip_ids` excludes the
    batches already in flight, which would otherwise come back again while
    the worker is still busy with them.
    """
    must_not = [{"exists": {"field": "annotate"}}]
    if skip_ids:
        must_not.append({"ids": {"values": list(skip_ids)}})
    return {
        "query": {"bool": {"must_not": must_not}},
        "sort": [{SORT_FIELD: {"order": "desc", "unmapped_type": "date"}}],
        "_source": TEXT_FIELDS,
        "size": BATCH,
    }


def build_text(source: dict) -> str:
    """Join the configured text fields, which is what the models should see."""
    parts = [(source.get(f) or "").strip() for f in TEXT_FIELDS]
    return "\n".join(p for p in parts if p).strip()[:MAX_CHARS]


def annotate_texts(texts):
    """Run all three models over a batch, one pass each."""
    emotions = predict_emotion_batch(texts)
    sentiments = analyze_sentiment(texts)
    entities = extract_entities_batch(texts, min_score=NER_MIN_SCORE)
    now = datetime.now(timezone.utc).isoformat()
    ann = [
        {"emotion": e, "sentiment": s, "entities": ent, "annotated_at": now}
        for e, s, ent in zip(emotions, sentiments, entities)
    ]
    if not NER_ON:
        # With NER off `entities` is empty for a real reason, not because the
        # article had none. The marker is what makes those documents findable
        # again later - without it they are indistinguishable from genuinely
        # entity-free articles and the gap is permanent.
        for a in ann:
            a["ner_skipped"] = True
    return ann


def annotate_batch(hits):
    """Annotation per hit id. Failures get an `error` marker.

    The marker matters: without it a document that always fails would be
    picked up forever, since only the presence of `annotate` retires it.
    """
    now = datetime.now(timezone.utc).isoformat()
    out = {}
    todo = []
    for h in hits:
        text = build_text(h["_source"])
        if text:
            todo.append((h["_id"], text))
        else:
            out[h["_id"]] = {"error": "empty document", "annotated_at": now}

    # Every document empty is a configuration error, not eight empty articles:
    # it means TEXT_FIELDS names fields this index does not have. Writing error
    # markers here would retire those documents permanently, so refuse instead.
    # This is not hypothetical - it silently poisoned 776 news documents once.
    if not todo and len(hits) >= 4:
        raise SystemExit(
            f"refusing to annotate: none of {TEXT_FIELDS} exist on any of "
            f"{len(hits)} documents in {INDEX}. Check TEXT_FIELDS."
        )

    if not todo:
        return out

    try:
        for (doc_id, _), ann in zip(todo, annotate_texts([t for _, t in todo])):
            out[doc_id] = ann
    except Exception as e:
        # One bad document must not poison the whole batch: retry alone so
        # only the offender gets the error marker.
        log(f"batch failed ({e}); retrying individually")
        for doc_id, text in todo:
            try:
                out[doc_id] = annotate_texts([text])[0]
            except Exception as inner:
                out[doc_id] = {"error": str(inner)[:200], "annotated_at": now}
    return out


def write_back(hits, annotations):
    actions = [
        {
            "_op_type": "update",
            "_index": h["_index"],  # concrete index, not the wildcard
            "_id": h["_id"],
            "doc": {"annotate": annotations[h["_id"]]},
        }
        for h in hits
        if h["_id"] in annotations
    ]
    # wait_for, not False: the ids leave `_inflight` as soon as this returns,
    # and until Elasticsearch refreshes (up to 1s) the documents still look
    # un-annotated. The producer fetches inside that window and hands the same
    # batch out again - measured at exactly 2x the work for 1x the progress.
    ok, errors = helpers.bulk(es, actions, raise_on_error=False, refresh="wait_for")
    if errors:
        log(f"{len(errors)} write(s) failed, first: {str(errors[0])[:200]}")
    return ok


def producer(once: bool):
    """Fetch batches ahead of the worker and hand them over the queue."""
    while not stop.is_set():
        try:
            with _inflight_lock:
                body = build_query(_inflight)
            hits = es.search(index=INDEX, body=body)["hits"]["hits"]
        except Exception as e:
            log(f"fetch failed: {e}")
            stop.wait(IDLE_SLEEP)
            continue

        if not hits:
            if once:
                work.put(None)
                return
            log("nothing to annotate, sleeping")
            stop.wait(IDLE_SLEEP)
            continue

        with _inflight_lock:
            _inflight.update(h["_id"] for h in hits)

        while not stop.is_set():
            try:
                work.put(hits, timeout=1)
                break
            except queue.Full:
                continue


def main():
    once = "--once" in sys.argv
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())

    log(f"index={INDEX} fields={','.join(TEXT_FIELDS)} batch={BATCH} "
        f"mode={'once' if once else 'daemon'}")
    threading.Thread(target=producer, args=(once,), daemon=True).start()

    total = 0
    while not stop.is_set():
        try:
            hits = work.get(timeout=1)
        except queue.Empty:
            continue
        if hits is None:  # producer drained the index (--once)
            break

        t0 = time.time()
        try:
            annotations = annotate_batch(hits)
            written = write_back(hits, annotations)
            total += written
            elapsed = time.time() - t0
            log(f"{written} docs in {elapsed:.1f}s ({elapsed / len(hits):.2f}s/doc), total {total}")
        finally:
            with _inflight_lock:
                _inflight.difference_update(h["_id"] for h in hits)

    log(f"stopped, {total} documents annotated")


if __name__ == "__main__":
    main()
