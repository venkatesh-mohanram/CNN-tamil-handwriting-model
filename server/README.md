# Portable experimental Tamil recognition API

This folder contains a Python HTTP service and an ONNX model for **அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ ஃ**. Copy this folder to another machine to use it. It does not import the original training project, download model weights, or require PyTorch.

**The bundled model is an experimental candidate, not an accuracy-qualified release.** Predictions and rejection can be wrong. The original evaluation, provenance, rejection thresholds and source release are included unchanged in `model/`. Starting this service does not change the model's qualification or the original project's production gate. Successful readiness means the candidate loaded and executed; it does not certify recognition quality.

## Install and start

Use Python 3.11–3.13 on a supported CPU platform. From this folder:

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock.txt
```

On Windows use `.venv\Scripts\python.exe` in place of `.venv/bin/python`. The dependency lock includes the complete CPU runtime dependency set. Creating a virtual environment and installing dependencies need to be done once per machine.

Prefer the handoff's launcher, which creates a private temporary backend credential and starts the client and API together. To start only the API, first create a local credential file:

```sh
.venv/bin/python -c "import os,secrets; f=os.open('service-token.txt',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600); os.write(f,secrets.token_urlsafe(48).encode()); os.close(f)"
.venv/bin/python app.py --token-file service-token.txt --port 8000 --cpu-threads 1
```

Keep `service-token.txt` private and outside any public static-file directory. Never place its contents in browser JavaScript, URLs, source control or command-line arguments. If the file already exists, the credential creation command refuses to overwrite it.

The CLI binds **127.0.0.1 only**. `--model-dir PATH` selects a trusted copy of this experimental model package. `--cpu-threads` controls ONNX's per-session intra-operation CPU threads, with one thread by default. The HTTP service runs two CPU workers with a total admission limit of 20 requests. These conservative limits include uploading, waiting and running work. Benchmark before changing them. There is no permissive CORS configuration; a browser client should call its own same-origin host, which keeps the service token on the backend.

The factory is also available for a host launcher:

```python
from pathlib import Path
from app import create_app

application = create_app(token_file=Path("/private/location/service-token.txt"))
```

## HTTP contract

All three routes require `Authorization: Bearer <credential>`:

- `GET /health/live`: `200` when the HTTP application is live.
- `GET /health/ready`: `200` only after model integrity, metadata, ONNX signatures and a real execution pass. Missing credentials or invalid model assets produce `503`.
- `POST /v1/tamil-symbols/recognize`: multipart form data with **exactly one file part named `image`**, `Content-Type: image/png`. Maximum PNG file size is **1 MiB**, maximum dimensions are **1024×1024**, and the overall multipart body is limited to 1 MiB plus 64 KiB. Use one dark handwritten symbol on a light background. Blank, multi-frame and invalid images are rejected.

A successful recognition returns:

```json
{
  "request_id": "generated-request-id",
  "status": "recognized",
  "character": "அ",
  "model_version": "tamil-symbols-0.1.0",
  "processing_ms": 12.5
}
```

This is an illustrative response shape, not a recorded accuracy result. A rejected prediction returns HTTP `200`, `"status": "unrecognized"` and `"character": null`. Model confidence is not exposed. `processing_ms` covers the handler's upload reading, queueing, preprocessing and inference; client/network time before the handler is excluded.

Errors have the form `{"request_id":"...","error":{"code":"...","message":"..."}}`:

| HTTP status | Meaning |
|---|---|
| 400 | Malformed, missing, empty or extra multipart parts; incomplete upload |
| 401 | Missing or incorrect bearer credential |
| 413 | File or request-body size limit exceeded |
| 415 | Non-PNG image or part media type |
| 422 | Undecodable image, unsupported dimensions, blank canvas or invalid ink |
| 429 | All admission slots occupied; `Retry-After: 1` |
| 503 | Missing credential configuration, invalid/unavailable model, or inference failure |

Cancellation does not create extra CPU capacity: an admitted inference retains its slot until its worker finishes. Uploads remain in bounded memory; filenames are ignored and image files are not written to disk. Operational logs contain request IDs, status/error codes, model version and timing, never images, credentials or client IPs. Uvicorn access logging is disabled by the CLI.

## Python request example

This uses only Python's standard library. Replace `drawing.png` with a PNG you may share with your local application. It sends a real HTTP request to the running API and prints its actual result.

```python
import json
from pathlib import Path
from urllib.request import Request, urlopen
from uuid import uuid4

token = Path("service-token.txt").read_text().strip()
image = Path("drawing.png").read_bytes()
boundary = "tamil-" + uuid4().hex
body = (
    f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="drawing.png"\r\n'
    'Content-Type: image/png\r\n\r\n'
).encode() + image + f"\r\n--{boundary}--\r\n".encode()
request = Request(
    "http://127.0.0.1:8000/v1/tamil-symbols/recognize",
    data=body,
    headers={
        "Authorization": "Bearer " + token,
        "Content-Type": "multipart/form-data; boundary=" + boundary,
    },
    method="POST",
)
with urlopen(request, timeout=15) as response:
    print(json.load(response))
```

## Model integrity and provenance

`model/manifest.json` declares exactly seven included files with SHA-256 and byte counts. Loading rejects missing, changed, oversized, symlinked or unexpected assets; label and preprocessing maps must match the implementation. The included `source-release.json` binds the copied subset to the original candidate's hashes. Source permission, finite 13×576 shape prototypes, fixed thresholds and actual ONNX input/output execution are also checked. Checksums detect changes relative to the trusted package; they are not a digital signature or a substitute for obtaining this folder from a trusted source.

The recognition decision matches the original model: stable softmax over 13 logits, followed by the saved probability threshold and similarity to the **predicted class's** prototype. The saved thresholds are not modified to make demonstrations appear better.

Dataset: Sasikaladevi N and colleagues, *tamil-handwritten-character-carpus*, Mendeley version 1, DOI [10.17632/6zcpgchvmx.1](https://data.mendeley.com/datasets/6zcpgchvmx/1), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The original attribution, modifications and evaluation limitations are preserved in `model/provenance.json` and `model/evaluation.json`. No writer-disjoint, child-handwriting or browser-canvas accuracy qualification is asserted.

## Tests

Install pytest and httpx in a development environment, then run `python -m pytest tests` from this folder. Tests include real bundled-candidate ONNX execution, corruption/permission/metadata rejection, authentication, malformed inputs, bounded concurrency, cancellation, safe logs and in-memory upload handling. Model-boundary doubles are used for HTTP concurrency tests, not as accuracy evidence.

The service makes no universal latency claim. Measure the complete request path with representative drawings on the deployment CPU.
