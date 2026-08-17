"""Self-check for the queue/query logic. Run: python test_pipeline.py

Does not touch Elasticsearch or load any model - it only covers the parts
that decide *which* documents get picked and *what text* they are scored on.
"""
import main


def test_query_skips_annotated_and_inflight():
    q = main.build_query({"a", "b"})
    must_not = q["query"]["bool"]["must_not"]
    assert {"exists": {"field": "annotate"}} in must_not
    assert sorted(must_not[1]["ids"]["values"]) == ["a", "b"]
    assert q["sort"][0][main.SORT_FIELD]["order"] == "desc"  # newest first

    # nothing in flight: no ids clause at all
    assert len(main.build_query(set())["query"]["bool"]["must_not"]) == 1


def test_text_building():
    assert main.build_text({"title": "Judul", "body": "Isi"}) == "Judul\nIsi"
    assert main.build_text({"title": "Judul"}) == "Judul"
    assert main.build_text({"body": None, "title": None}) == ""
    assert len(main.build_text({"body": "x" * 99999})) == main.MAX_CHARS


def test_text_fields_are_configurable():
    """A Threads post keeps its text in `text`, not title+body."""
    original = main.TEXT_FIELDS
    main.TEXT_FIELDS = ["text"]
    try:
        assert main.build_text({"text": "Isi post", "title": "diabaikan"}) == "Isi post"
        assert main.build_text({"title": "hanya judul"}) == ""
        assert main.build_query(set())["_source"] == ["text"]
    finally:
        main.TEXT_FIELDS = original


def test_empty_docs_get_error_marker_not_a_model_call():
    hits = [{"_id": "1", "_source": {"title": "", "body": ""}}]
    out = main.annotate_batch(hits)
    assert "error" in out["1"]  # would loop forever without this


def test_bad_doc_does_not_poison_the_batch(monkeypatch=None):
    real = main.annotate_texts
    calls = []

    def fake(texts):
        calls.append(list(texts))
        if len(texts) > 1:
            raise RuntimeError("batch boom")
        if texts[0] == "bad":
            raise RuntimeError("doc boom")
        return [{"emotion": {}, "sentiment": {}, "entities": [], "annotated_at": "t"}]

    main.annotate_texts = fake
    try:
        out = main.annotate_batch([
            {"_id": "good", "_source": {"title": "ok"}},
            {"_id": "bad", "_source": {"title": "bad"}},
        ])
    finally:
        main.annotate_texts = real

    assert "error" not in out["good"], "healthy doc must still be annotated"
    assert out["bad"]["error"] == "doc boom"
    assert calls[0] == ["ok", "bad"]  # batched first, then retried alone


def test_wrong_text_fields_refuses_instead_of_marking_everything_bad():
    """Wrong TEXT_FIELDS must stop the run, not retire the whole index."""
    original = main.TEXT_FIELDS
    main.TEXT_FIELDS = ["nonexistent_field"]
    hits = [{"_id": str(i), "_source": {"title": "ada isinya", "body": "juga"}} for i in range(8)]
    try:
        try:
            main.annotate_batch(hits)
        except SystemExit as e:
            assert "TEXT_FIELDS" in str(e), e
        else:
            raise AssertionError("should have refused")
    finally:
        main.TEXT_FIELDS = original


def test_single_genuinely_empty_document_still_gets_a_marker():
    """One empty article is normal and must still be retired."""
    out = main.annotate_batch([{"_id": "1", "_source": {"title": "", "body": ""}}])
    assert "error" in out["1"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
