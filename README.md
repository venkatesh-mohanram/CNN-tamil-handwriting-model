# Tamil handwriting recognizer — portable local handoff

Start this folder on another macOS or Linux computer, open the printed browser URL, draw one character, and select Recognize. The client calls the bundled Python server; inference uses the included ONNX model on CPU. No GPU, Node.js, PyTorch, hosted model account, or original workshop folder is required.

Supported symbols: **அ ஆ இ ஈ உ ஊ எ ஏ ஐ ஒ ஓ ஔ ஃ**. One canvas-drawn symbol per request. This is a **local manual-verification package**, not a qualified production release: the candidate reached 95.38% overall held-out accuracy but failed the required per-symbol scores for எ and ஒ, and accepts most unsupported Tamil inputs. The original `qualified: false` metadata is preserved. See [model and training details](MODEL_TECHNICAL_DETAILS.md) and [measured evaluation](server/docs/model-evaluation.md).

## Start

Install Python **3.11–3.13** (3.12 recommended). Internet access is needed on the first run to install the pinned CPU dependencies; subsequent runs can reuse the local environment.

From the extracted folder, on macOS/Linux:

```sh
./start.sh
```

The launcher creates `.venv`, installs [server/requirements.lock.txt](server/requirements.lock.txt), starts both processes, and prints their URLs. Open **http://127.0.0.1:8080/**. The private inference service listens on **127.0.0.1:8000**. Wait for startup/readiness before drawing. **Ctrl+C** stops both processes.

If an archive tool removed executable permission:

```sh
chmod +x start.sh
./start.sh
```

To use different ports or an existing supported Python:

```sh
./start.sh --server-port 8100 --client-port 8180
TAMIL_PYTHON=python3.12 ./start.sh
```

Additional options:

```sh
./start.sh --setup-only
./start.sh --no-install
./start.sh --help
```

`--no-install` uses the current bootstrap Python and skips environment creation/dependency installation; that interpreter must already have the pinned dependencies. Keep the complete folder together when moving it. The model is included; startup does not download a foundation model or contact a model registry.

## Verify manually

1. Open the client URL printed at startup.
2. Draw one supported symbol using mouse, touch or stylus; disconnected dots/strokes belong to the same drawing.
3. Select Recognize once and inspect the prediction or “not recognized.”
4. Use Undo to remove the last stroke or Clear to start again.
5. Try several different styles, including எ, ஒ and unsupported marks, to understand the known weaknesses.

Predictions identify symbols; they do not grade handwriting or assess stroke order. English letters, words, photographs and arbitrary characters are outside the 13-class scope and may still be misclassified as a supported symbol.

## Folder layout

- `server/`: Python inference service, full ONNX candidate under `model/`, dependency lock, evaluation/training/source metadata under `docs/`, and upstream notices under `licenses/`.
- `client/`: local drawing interface and host proxy. The browser uses the same-origin `/api/...` adapter; the proxy calls the authenticated private API.
- `start.sh`, `start.py`: coordinated local environment setup, startup and shutdown.
- `MODEL_TECHNICAL_DETAILS.md`: architecture, preprocessing, training, dataset, rejection method and measured limitations.
- `ATTRIBUTION.md`: dataset creators, DOI, license and project modifications.

The inference boundary is `POST /v1/tamil-symbols/recognize`, accepting one multipart PNG field named `image` (≤1 MiB, dimensions ≤1024×1024). A recognized result includes `request_id`, `status`, `character`, `model_version`, and `processing_ms`; an unrecognized result has `character: null`. Backend credentials remain in temporary server-side storage and never appear in browser JavaScript. The launcher generates the local credential automatically.

Submitted images and strokes are processed in memory, not saved for training or logged as request bodies. The client sends the drawing to the local server when you choose Recognize; this differs from a browser-only inference demo. No application traffic is sent to external inference services.

## Troubleshooting and limits

- A port-in-use error: stop the conflicting application or select two unused ports.
- An unsupported-Python error: select Python 3.11–3.13 with `TAMIL_PYTHON`; Python 3.12 is the recommended baseline.
- Dependency installation failure: verify internet/certificate access to the package source and that wheels exist for your OS/CPU. No TLS bypass is required or recommended. macOS and Linux CPU environments are the intended targets; other platforms are unverified.
- Blank/oversized/invalid images are rejected; use a single dark symbol on the white canvas.
- “Not recognized” is a valid decision; a confident-looking prediction can also be wrong. Softmax values are not calibrated confidence and are not exposed as such.

The package does not claim Ubuntu 20-user latency/load qualification, children's canvas accuracy, or complete touch/stylus/accessibility validation. Keep it on loopback for this local handoff. Production deployment requires improved per-symbol/rejection results, independent evaluation and the outstanding hardware/integration gates. The original production loader's fail-closed qualification rule is not weakened by this explicitly experimental local package.

## Handoff verification

The isolated CPU environment was installed and the one-command launcher started both services on this Mac with Python 3.12. All **43 Python tests passed**. The actual canvas-to-proxy-to-ONNX flow returned **ஃ** for a three-dot test drawing; that individual request reported **256.2 ms** of server processing. This is an integration check, not a concurrent-load or accuracy qualification. A [development screenshot](server/docs/handoff-preview.jpg) contains generated test dots rather than a retained application submission.
