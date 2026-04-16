from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from web.routes import router


BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(title="Document Extraction Web UI")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
app.include_router(router)


@app.get("/", response_class=HTMLResponse)
def index():
    return (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")


@app.get("/favicon.ico")
def favicon():
    raise HTTPException(status_code=404, detail="No favicon")
