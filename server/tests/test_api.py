"""HTTP contracts using real image parsing and a replaceable model boundary."""
import asyncio
from contextlib import asynccontextmanager
from importlib.util import find_spec
from io import BytesIO
import threading

import httpx
from PIL import Image, ImageDraw
import pytest

ROUTE = "/v1/tamil-symbols/recognize"
TOKEN = "test-service-credential-" * 3
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def png_bytes():
    image = Image.new("RGB", (320, 320), "white")
    ImageDraw.Draw(image).line([(70, 230), (150, 70), (230, 230)], fill="black", width=12)
    stream = BytesIO()
    image.save(stream, "PNG")
    return stream.getvalue()


def recognized(tensor):
    assert tensor.shape == (1, 3, 224, 224)
    return {"status": "recognized", "character": "அ", "model_version": "test-model"}


@asynccontextmanager
async def client_for(app):
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            yield client


def service_app(tmp_path, **kwargs):
    from recognition.service import create_app
    credential = tmp_path / "service-token"
    credential.write_text(TOKEN)
    return create_app(token_file=credential, **kwargs)


def test_service_requires_auth_and_real_model_for_readiness(tmp_path):
    assert find_spec("recognition.service") is not None, "HTTP service must be implemented"
    async def scenario():
        async with client_for(service_app(tmp_path)) as client:
            assert (await client.get("/health/live")).status_code == 401
            assert (await client.get("/health/live", headers=AUTH)).status_code == 200
            assert (await client.get("/health/ready", headers=AUTH)).status_code == 503
            response = await client.post(ROUTE, headers=AUTH)
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "not_ready"
    asyncio.run(scenario())


def test_missing_credential_configuration_fails_closed():
    from recognition.service import create_app
    async def scenario():
        async with client_for(create_app(predictor=recognized)) as client:
            assert (await client.get("/health/ready", headers=AUTH)).status_code == 503
    asyncio.run(scenario())


def test_recognition_returns_contract_without_writing_uploads(tmp_path, monkeypatch):
    import tempfile
    def forbidden(*args, **kwargs):
        pytest.fail("HTTP multipart upload must stay in memory")
    async def scenario():
        app = service_app(tmp_path, predictor=recognized)
        async with client_for(app) as client:
            monkeypatch.setattr(tempfile, "TemporaryFile", forbidden)
            monkeypatch.setattr(tempfile, "NamedTemporaryFile", forbidden)
            monkeypatch.setattr(tempfile, "SpooledTemporaryFile", forbidden)
            response = await client.post(ROUTE, headers=AUTH, files={"image": ("../../private.png", png_bytes(), "image/png")})
            assert response.status_code == 200, response.text
            result = response.json()
            assert result["character"] == "அ"
            assert result["status"] == "recognized"
            assert result["model_version"] == "test-model"
            assert result["request_id"]
            assert result["processing_ms"] >= 0
            assert set(result) == {"status", "character", "model_version", "request_id", "processing_ms"}
            assert len(list(tmp_path.iterdir())) == 1  # Only the explicitly provisioned credential.
    asyncio.run(scenario())


@pytest.mark.parametrize("files,expected", [
    ([("wrong", ("x.png", b"x", "image/png"))], 400),
    ([("image", ("x.png", b"x", "image/png")), ("image", ("y.png", b"y", "image/png"))], 400),
    ([("image", ("x.txt", b"text", "text/plain"))], 415),
    ([("image", ("x.png", b"not png", "image/png"))], 422),
    ([("image", ("x.png", b"x" * (1024 * 1024 + 1), "image/png"))], 413),
])
def test_invalid_multipart_is_rejected(tmp_path, files, expected):
    async def scenario():
        async with client_for(service_app(tmp_path, predictor=recognized)) as client:
            response = await client.post(ROUTE, headers=AUTH, files=files)
            assert response.status_code == expected, response.text
            assert set(response.json()) == {"request_id", "error"}
    asyncio.run(scenario())


def test_rejects_malformed_missing_extra_and_oversized_streamed_body(tmp_path):
    async def scenario():
        async with client_for(service_app(tmp_path, predictor=recognized)) as client:
            response = await client.post(ROUTE, headers={**AUTH, "Content-Type": "multipart/form-data; boundary=nope"}, content=b"malformed")
            assert response.status_code == 400
            assert (await client.post(ROUTE, headers=AUTH)).status_code == 400
            response = await client.post(ROUTE, headers=AUTH, files={"image": ("a.png", png_bytes(), "image/png")}, data={"extra": "field"})
            assert response.status_code == 400
            async def oversized():
                yield b"x" * (1024 * 1024)
                yield b"x" * (64 * 1024 + 1)
            assert (await client.post(ROUTE, headers=AUTH, content=oversized())).status_code == 413
    asyncio.run(scenario())


def test_busy_and_cancelled_requests_hold_slots_until_worker_finishes(tmp_path):
    async def scenario():
        entered = asyncio.Event()
        finished = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def blocked(tensor):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5), "test must release worker"
            loop.call_soon_threadsafe(finished.set)
            return recognized(tensor)
        app = service_app(tmp_path, predictor=blocked, capacity=1, workers=1)
        async with client_for(app) as client:
            first = asyncio.create_task(client.post(ROUTE, headers=AUTH, files={"image": ("a.png", png_bytes(), "image/png")}))
            try:
                await asyncio.wait_for(entered.wait(), 5)
                # A live endpoint must stay responsive while inference is blocked.
                assert (await client.get("/health/live", headers=AUTH)).status_code == 200
                first.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await first
                response = await client.post(ROUTE, headers=AUTH, files={"image": ("b.png", png_bytes(), "image/png")})
                assert response.status_code == 429
                assert response.headers["Retry-After"] == "1"
            finally:
                release.set()
                await asyncio.wait_for(finished.wait(), 5)
    asyncio.run(scenario())


def test_model_errors_do_not_expose_exception_details(tmp_path):
    def broken(tensor):
        raise RuntimeError("sensitive internal path and credential")
    async def scenario():
        async with client_for(service_app(tmp_path, predictor=broken)) as client:
            response = await client.post(ROUTE, headers=AUTH, files={"image": ("a.png", png_bytes(), "image/png")})
            assert response.status_code == 503
            assert "sensitive" not in response.text
    asyncio.run(scenario())


def test_queued_cancelled_job_still_counts_toward_capacity(tmp_path):
    async def scenario():
        entered = asyncio.Event()
        queued_body_received = asyncio.Event()
        release = threading.Event()
        loop = asyncio.get_running_loop()
        def blocked(tensor):
            loop.call_soon_threadsafe(entered.set)
            assert release.wait(5)
            return recognized(tensor)
        body = b'--test\r\nContent-Disposition: form-data; name="image"; filename="a.png"\r\nContent-Type: image/png\r\n\r\n' + png_bytes() + b'\r\n--test--\r\n'
        async def queued_upload():
            yield body
            queued_body_received.set()
        async with client_for(service_app(tmp_path, predictor=blocked, capacity=2, workers=1)) as client:
            first = asyncio.create_task(client.post(ROUTE, headers=AUTH, files={"image": ("a.png", png_bytes(), "image/png")}))
            second = None
            try:
                await asyncio.wait_for(entered.wait(), 5)
                second = asyncio.create_task(client.post(ROUTE, headers={**AUTH, "Content-Type": "multipart/form-data; boundary=test"}, content=queued_upload()))
                await asyncio.wait_for(queued_body_received.wait(), 5)
                second.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await second
                response = await client.post(ROUTE, headers=AUTH, files={"image": ("a.png", png_bytes(), "image/png")})
                assert response.status_code == 429
            finally:
                release.set()
                assert (await asyncio.wait_for(first, 5)).status_code == 200
    asyncio.run(scenario())


def test_two_workers_run_concurrently_and_logs_contain_only_operational_metadata(tmp_path, caplog):
    async def scenario():
        both_entered = asyncio.Event()
        release = threading.Event()
        lock = threading.Lock()
        arrivals = 0
        loop = asyncio.get_running_loop()
        def blocked(tensor):
            nonlocal arrivals
            with lock:
                arrivals += 1
                if arrivals == 2:
                    loop.call_soon_threadsafe(both_entered.set)
            assert release.wait(5)
            return recognized(tensor)
        caplog.set_level("INFO", logger="recognition.requests")
        async with client_for(service_app(tmp_path, predictor=blocked, capacity=2, workers=2)) as client:
            jobs = [asyncio.create_task(client.post(ROUTE, headers=AUTH, files={"image": ("private-filename.png", png_bytes(), "image/png")})) for _ in range(2)]
            try:
                await asyncio.wait_for(both_entered.wait(), 5)
                assert (await client.post(ROUTE, headers=AUTH)).status_code == 429
            finally:
                release.set()
                results = await asyncio.wait_for(asyncio.gather(*jobs), 5)
            assert all(result.status_code == 200 for result in results)
        import json
        logs = [json.loads(record.message) for record in caplog.records if record.name == "recognition.requests"]
        assert len(logs) == 3
        assert all(set(row) == {"request_id", "status", "code", "model_version", "processing_ms"} for row in logs)
        assert [row['model_version'] for row in logs].count('test-model') == 2
        assert TOKEN not in caplog.text and "private-filename" not in caplog.text
    asyncio.run(scenario())


def test_invalid_release_never_becomes_ready(tmp_path):
    async def scenario():
        async with client_for(service_app(tmp_path, release_dir=tmp_path / "absent-release")) as client:
            assert (await client.get("/health/ready", headers=AUTH)).status_code == 503
    asyncio.run(scenario())
