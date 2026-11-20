# Portable canvas client

From the handoff root, run `./start.sh` and open the printed local URL (default `http://127.0.0.1:8080/`). The launcher starts the API and this canvas host. No Node build, original workshop folder, CDN, or model download is needed.

The page supports mouse, touch, and stylus, Clear, Undo, and explicit Recognize. It displays a prediction or “not recognized,” model version, and server processing time. The model is experimental; predictions are not correctness feedback. Drawings stay in memory and are not added to training.

## Integration

`component/index.js` registers the reusable `<tamil-symbol-input>` Web Component; serve it together with `component/recognition.js`. Assign its `recognize(image: Blob, signal: AbortSignal)` property an async function returning the API response. See [app.mjs](app.mjs) for a complete adapter. Events are `recognition-started`, `recognition-result` (`detail` contains the validated response), and `recognition-error` (`detail.message` is safe text). A pending request disables duplicate submission; disconnect and the ten-second UI timeout signal cancellation.

The browser calls same-origin `POST /api/tamil-symbols/recognize`, sending multipart `image` PNG. [serve.py](serve.py) forwards to the private loopback API and adds its token on the backend. No bearer token appears in HTML or JavaScript, and no CORS permission is needed. The private API contract and direct nonbrowser examples are in [server/README.md](../server/README.md).

For manual operation with dependencies installed as described in the root README:

```sh
python client/serve.py --token-file /private/path/service-token --api-port 8000 --port 8080
```

Aliases `--server-port` and `--client-port` are supported. Both listeners must use different ports. The client host binds only `127.0.0.1`, verifies loopback Host and same-origin Origin headers, limits uploads to 1 MiB plus multipart overhead, bounds responses, and propagates cancellation. It serves only the page, browser script, and component assets, never the token or Python source. The token must be outside this handoff folder. Access logs, request-body logging, uploads to disk, browser storage, and automatic stroke submission are absent.

This local adapter has no user-account system. A deployed web/mobile application must provide its own authenticated backend and public-network controls. Native applications can reuse the [API](../server/README.md) with their own drawing surface or embed the component in a WebView. See [model technical details](../MODEL_TECHNICAL_DETAILS.md) for model/data limitations.

Tests: `python -m pytest client/tests`. Tests use explicit transport fixtures, not simulated predictions in the application.
