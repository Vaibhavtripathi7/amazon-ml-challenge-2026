import io

from PIL import Image

from amlc.images import download_all


def png_bytes(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), "red").save(buf, format="PNG")
    return buf.getvalue()


def test_downloads_resize_to_max_side(tmp_path):
    fetch = lambda url: png_bytes(2000, 1000)
    res = download_all({1: "http://x/a.jpg"}, tmp_path, fetch=fetch, max_side=512)
    im = Image.open(tmp_path / "1.jpg")
    assert max(im.size) == 512 and im.size == (512, 256)
    assert res.status.tolist() == ["ok"]


def test_existing_files_are_skipped_so_reruns_resume(tmp_path):
    calls = []
    fetch = lambda url: calls.append(url) or png_bytes(10, 10)
    download_all({1: "u1", 2: "u2"}, tmp_path, fetch=fetch)
    calls.clear()
    res = download_all({1: "u1", 2: "u2", 3: "u3"}, tmp_path, fetch=fetch)
    assert calls == ["u3"]
    assert sorted(res.status) == ["ok", "skipped", "skipped"]


def test_failures_are_retried_then_recorded_not_raised(tmp_path):
    attempts = []

    def fetch(url):
        attempts.append(url)
        raise OSError("throttled")

    res = download_all({7: "bad"}, tmp_path, fetch=fetch, retries=3, backoff=0)
    assert len(attempts) == 3
    assert res.status.tolist() == ["failed"]
    assert "throttled" in res.error.iloc[0]
    assert not (tmp_path / "7.jpg").exists()
