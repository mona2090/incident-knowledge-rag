from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import Field
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response
from src.observability.runtime import ObservedRuntime, public_result
from src.retrieval.filters import SearchScope


class AskRequest(SearchScope):
    query: str = Field(min_length=1, max_length=1000)


def create_app(runtime=None):
    @asynccontextmanager
    async def lifespan(app):
        app.state.runtime = runtime if runtime is not None else ObservedRuntime()
        yield

    app = FastAPI(title="Incident Knowledge Assistant", lifespan=lifespan)

    @app.get("/healthz")
    def health():
        return {"status": "ok"}

    @app.get("/metrics")
    def metrics():
        return Response(content=generate_latest(app.state.runtime.telemetry.registry),
                        headers={"Content-Type": CONTENT_TYPE_LATEST})

    @app.post("/ask")
    def ask(request: AskRequest):
        request_id = str(uuid4())
        try:
            state = app.state.runtime.invoke(request.model_dump(), request_id=request_id)
        except Exception as exc:
            raise HTTPException(status_code=502, detail={"request_id": request_id,
                "error": type(exc).__name__, "message": "RAG request failed; inspect the matching trace."}) from None
        result = public_result(state)
        if result["status"] == "error":
            raise HTTPException(status_code=502, detail=result)
        return result

    return app


app = create_app()
