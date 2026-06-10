# -*- coding: utf-8 -*-
"""PROMPT 19: HTTP transport for the Spin Wheel Predictor REST API.

Two transports share ONE routing core (`api.service.ApiService`):

* FastAPI + uvicorn when installed -> interactive Swagger docs at /docs,
  pydantic request schemas, the "nice" experience.
* A stdlib `http.server` fallback otherwise -> zero extra dependencies, so the
  API ALWAYS works even on a bare Python install.

`serve()` prefers FastAPI and transparently falls back. Tests drive the stdlib
server directly (see `make_stdlib_server`).
"""
import json as _json


def make_service(vm=None):
    """Build an ApiService backed by the real Tracker/ViewModel."""
    from api.service import ApiService, RealBackend
    return ApiService(RealBackend(vm=vm))


# ---------------------------------------------------------------------- #
# FastAPI transport (optional)
# ---------------------------------------------------------------------- #
def create_fastapi_app(service):
    """Build a FastAPI app delegating every route to the shared service core.

    Imported lazily so this module imports fine without FastAPI installed.
    """
    from typing import Any, List, Optional
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel

    class PredictReq(BaseModel):
        engine: Optional[str] = None
        history_length: Optional[int] = None

    class BetItem(BaseModel):
        number: int
        token_bet: float = 0
        confidence: Optional[float] = None
        ev_per_token: Optional[float] = None
        is_positive_ev: Optional[bool] = None
        support: Optional[Any] = None

    class RecordReq(BaseModel):
        actual_number: int
        predicted_number: Optional[int] = None
        profit_change: float
        bets: Optional[List[BetItem]] = None
        engine_used: Optional[str] = None
        mode: Optional[str] = None

    app = FastAPI(
        title="Spin Wheel Predictor API",
        version=str(service.backend.version()),
        description=(
            "REST access to the Spin Wheel Predictor. NOTE: predictions are "
            "probabilistic estimates on a near-random process, NOT guaranteed "
            "outcomes."
        ),
    )

    def _resp(result):
        status, payload = result
        return JSONResponse(status_code=status, content=payload)

    @app.get("/")
    def root():
        return _resp(service.handle("GET", "/"))

    @app.get("/health")
    def health():
        return _resp(service.handle("GET", "/health"))

    @app.get("/stats")
    def stats():
        return _resp(service.handle("GET", "/stats"))

    @app.get("/history")
    def history(limit: int = 50):
        return _resp(service.handle("GET", "/history", query={"limit": limit}))

    @app.post("/predict")
    def predict(req: PredictReq):
        return _resp(service.handle("POST", "/predict", body=req.model_dump()))

    @app.post("/record")
    def record(req: RecordReq):
        return _resp(service.handle("POST", "/record", body=req.model_dump()))

    return app


# ---------------------------------------------------------------------- #
# stdlib transport (always available)
# ---------------------------------------------------------------------- #
def make_stdlib_server(service, host="127.0.0.1", port=8000):
    """Return a ThreadingHTTPServer routing JSON requests through `service`.

    Pass port=0 to let the OS pick a free port (handy for tests); read the real
    port from `server.server_address[1]`.
    """
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status, payload):
            body = _json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_body(self):
            try:
                length = int(self.headers.get("Content-Length", 0) or 0)
            except (TypeError, ValueError):
                length = 0
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            if not raw:
                return {}
            try:
                return _json.loads(raw.decode("utf-8"))
            except Exception:
                return None  # signals malformed JSON

        def do_GET(self):
            self._send(*service.handle("GET", self.path))

        def do_POST(self):
            body = self._read_body()
            if body is None:
                self._send(400, {"error": "invalid JSON body"})
                return
            self._send(*service.handle("POST", self.path, body=body))

        def log_message(self, *args):  # keep the console quiet
            pass

    return ThreadingHTTPServer((host, port), Handler)


def serve(host="127.0.0.1", port=8000, prefer_fastapi=True, vm=None):
    """Run the API server. Prefers FastAPI+uvicorn, falls back to stdlib."""
    service = make_service(vm=vm)
    if prefer_fastapi:
        try:
            import uvicorn  # noqa: F401
            app = create_fastapi_app(service)
            print(f"[API] FastAPI live at http://{host}:{port}  (Swagger docs: /docs)")
            import uvicorn as _uvicorn
            _uvicorn.run(app, host=host, port=port)
            return
        except ImportError:
            print("[API] FastAPI/uvicorn not installed -> using stdlib http.server fallback.")
            print("      (install with: pip install -r requirements-api.txt)")
    httpd = make_stdlib_server(service, host=host, port=port)
    print(f"[API] stdlib http.server live at http://{host}:{port}")
    print("      Endpoints: GET /health /stats /history  |  POST /predict /record")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[API] shutting down...")
        httpd.shutdown()
