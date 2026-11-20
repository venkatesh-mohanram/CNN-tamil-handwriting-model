"""Authenticated, bounded in-memory PNG recognition over HTTP."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
import hmac
import json
import logging
import os
from pathlib import Path
import threading
import time
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from python_multipart import MultipartParser
from python_multipart.exceptions import MultipartParseError
from python_multipart.multipart import parse_options_header
from starlette.requests import ClientDisconnect

PNG_LIMIT = 1024 * 1024
BODY_LIMIT = PNG_LIMIT + 64 * 1024
SYMBOLS = frozenset("அஆஇஈஉஊஎஏஐஒஓஔஃ")
LOGGER = logging.getLogger("recognition.requests")


class RequestRejection(Exception):
    """A deliberately public HTTP rejection, without infrastructure details."""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code


def create_app(
    release_dir: Path | None = None,
    token_file: Path | None = None,
    predictor=None,
    capacity: int = 20,
    workers: int = 2,
) -> FastAPI:
    """Create a service; missing credentials or a validated model fail closed.

    Capacity includes uploading, queued, and running requests. Cancellation never
    releases a submitted job's admission permit before the CPU worker completes.
    """
    if not isinstance(capacity, int) or not isinstance(workers, int) or not 1 <= workers <= capacity <= 20:
        raise ValueError("Require 1 <= workers <= capacity <= 20.")
    executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tamil-recognition")
    admission = threading.BoundedSemaphore(capacity)
    active_predictor = predictor
    credential = None

    @asynccontextmanager
    async def lifespan(application):
        nonlocal active_predictor, credential
        credential = await asyncio.to_thread(_read_credential, token_file)
        if active_predictor is None and release_dir is not None:
            try:
                from .model import load_model as load_release
                active_predictor = await asyncio.to_thread(load_release, release_dir)
            except (ImportError, OSError, ValueError):
                # Readiness exposes absence, while loader paths/details stay private.
                active_predictor = None
                LOGGER.error(json.dumps({"event": "model_unavailable"}))
        try:
            yield
        finally:
            await asyncio.to_thread(executor.shutdown, wait=True, cancel_futures=False)

    application = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    def authenticate(request):
        if credential is None:
            raise RequestRejection(503, "not_ready", "Service credentials are not configured.")
        supplied = request.headers.get("authorization", "")
        if not supplied.startswith("Bearer ") or not hmac.compare_digest(supplied[7:].encode("utf-8"), credential):
            raise RequestRejection(401, "unauthorized", "A valid bearer credential is required.")

    async def health(request: Request, ready: bool):
        request_id = uuid4().hex
        try:
            authenticate(request)
            if ready and not callable(active_predictor):
                raise RequestRejection(503, "not_ready", "A validated model is not loaded.")
            return JSONResponse({"status": "ready" if ready else "live"})
        except RequestRejection as exc:
            return _error(request_id, exc)

    @application.get("/health/live")
    async def live(request: Request):
        return await health(request, False)

    @application.get("/health/ready")
    async def ready(request: Request):
        return await health(request, True)

    @application.post("/v1/tamil-symbols/recognize")
    async def recognize(request: Request):
        started = time.perf_counter()
        request_id = uuid4().hex
        acquired = False
        submitted = False
        response_status = 500
        code = "interrupted"
        model_version = None
        try:
            authenticate(request)
            if not callable(active_predictor):
                raise RequestRejection(503, "not_ready", "A validated model is not loaded.")
            acquired = admission.acquire(blocking=False)
            if not acquired:
                raise RequestRejection(429, "busy", "Recognition capacity is full. Retry shortly.")
            body = await _read_body(request)
            future = executor.submit(_recognize_body, body, request.headers.get("content-type", ""), active_predictor)
            # The worker owns this permit after submission, including on disconnect.
            future.add_done_callback(lambda completed: admission.release())
            submitted = True
            outcome = await asyncio.shield(asyncio.wrap_future(future))
            if isinstance(outcome, RequestRejection):
                raise outcome
            response_status, code = 200, outcome["status"]
            model_version = outcome["model_version"]
            return JSONResponse({**outcome, "request_id": request_id, "processing_ms": (time.perf_counter() - started) * 1000})
        except RequestRejection as exc:
            response_status, code = exc.status, exc.code
            return _error(request_id, exc)
        finally:
            if acquired and not submitted:
                admission.release()
            LOGGER.info(json.dumps({"request_id": request_id, "status": response_status, "code": code,
                                   "model_version": model_version,
                                   "processing_ms": round((time.perf_counter() - started) * 1000, 3)}))

    return application


def _read_credential(token_file):
    if token_file is None:
        return None
    try:
        with Path(token_file).open("rb") as stream:
            value = stream.read(4097).strip()
        if not 32 <= len(value) <= 4096 or any(byte < 33 or byte > 126 for byte in value):
            return None
        return value
    except OSError:
        return None


async def _read_body(request):
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            length = int(declared)
        except ValueError as exc:
            raise RequestRejection(400, "invalid_multipart", "Invalid request length.") from exc
        if length < 0:
            raise RequestRejection(400, "invalid_multipart", "Invalid request length.")
        if length > BODY_LIMIT:
            raise RequestRejection(413, "image_too_large", "Request exceeds the upload size limit.")
    chunks = bytearray()
    try:
        async with asyncio.timeout(10):
            async for chunk in request.stream():
                if len(chunks) + len(chunk) > BODY_LIMIT:
                    raise RequestRejection(413, "image_too_large", "Request exceeds the upload size limit.")
                chunks.extend(chunk)
    except (TimeoutError, ClientDisconnect) as exc:
        raise RequestRejection(400, "invalid_multipart", "Upload did not complete.") from exc
    return bytes(chunks)


def _recognize_body(body, content_type, predictor):
    """Translate worker failures to safe outcomes, including abandoned requests."""
    try:
        from .images import ImageInputError, preprocess_png
        payload = _multipart_png(body, content_type)
        try:
            tensor = preprocess_png(payload)
        except ImageInputError as exc:
            return RequestRejection(exc.status, exc.code, str(exc))
        result = predictor(tensor)
        if not isinstance(result, dict) or result.get("status") not in {"recognized", "unrecognized"}:
            raise ValueError("Invalid predictor outcome")
        if result["status"] == "recognized" and result.get("character") not in SYMBOLS:
            raise ValueError("Invalid recognized symbol")
        if result["status"] == "unrecognized" and result.get("character") is not None:
            raise ValueError("Unrecognized output contains a symbol")
        if not isinstance(result.get("model_version"), str) or not result["model_version"]:
            raise ValueError("Missing model version")
        return {key: result[key] for key in ("status", "character", "model_version")}
    except RequestRejection as exc:
        return exc
    except Exception:
        # This is the external model/codec boundary: no exception details reach
        # responses or logs, and callers receive an explicit retryable failure.
        return RequestRejection(503, "inference_unavailable", "Recognition is temporarily unavailable.")


def _multipart_png(body, content_type):
    """Parse one PNG part with low-level callbacks; never spool uploads to disk."""
    media_type, options = parse_options_header(content_type)
    boundary = options.get(b"boundary")
    if media_type.lower() != b"multipart/form-data" or not boundary or len(boundary) > 200:
        raise RequestRejection(400, "invalid_multipart", "Send exactly one multipart image file.")
    headers = {}
    header_name, header_value, payload = bytearray(), bytearray(), bytearray()
    parts = 0
    ended = False

    def begin_part():
        nonlocal parts
        parts += 1
        if parts != 1:
            raise RequestRejection(400, "invalid_multipart", "Send exactly one image part.")

    def field(chunk, start, end):
        header_name.extend(chunk[start:end])

    def value(chunk, start, end):
        header_value.extend(chunk[start:end])

    def end_header():
        key = bytes(header_name).lower()
        if key in headers:
            raise RequestRejection(400, "invalid_multipart", "Duplicate multipart header.")
        headers[key] = bytes(header_value)
        header_name.clear()
        header_value.clear()

    def end_headers():
        disposition, attributes = parse_options_header(headers.get(b"content-disposition", b""))
        if disposition != b"form-data" or attributes.get(b"name") != b"image" or b"filename" not in attributes:
            raise RequestRejection(400, "invalid_multipart", "Expected one image file part.")
        part_type, _ = parse_options_header(headers.get(b"content-type", b""))
        if part_type.lower() != b"image/png":
            raise RequestRejection(415, "unsupported_media_type", "Only PNG images are supported.")
        if b"content-transfer-encoding" in headers:
            raise RequestRejection(400, "invalid_multipart", "Encoded multipart parts are unsupported.")

    def image_chunk(chunk, start, end):
        if len(payload) + end - start > PNG_LIMIT:
            raise RequestRejection(413, "image_too_large", "PNG exceeds 1 MiB.")
        payload.extend(chunk[start:end])

    def end():
        nonlocal ended
        ended = True

    parser = MultipartParser(boundary, {"on_part_begin": begin_part, "on_header_field": field, "on_header_value": value, "on_header_end": end_header, "on_headers_finished": end_headers, "on_part_data": image_chunk, "on_end": end})
    try:
        parser.write(body)
        parser.finalize()
    except (MultipartParseError, ValueError) as exc:
        raise RequestRejection(400, "invalid_multipart", "Malformed multipart upload.") from exc
    finally:
        parser.close()
    if not ended or parts != 1 or not payload:
        raise RequestRejection(400, "invalid_multipart", "Incomplete or empty image upload.")
    return bytes(payload)


def _error(request_id, rejection):
    headers = {"Retry-After": "1"} if rejection.status == 429 else {}
    if rejection.status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse({"request_id": request_id, "error": {"code": rejection.code, "message": str(rejection)}}, status_code=rejection.status, headers=headers)


