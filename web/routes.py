from __future__ import annotations

from pathlib import Path
import subprocess
import traceback

import psutil
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.layout import LAYOUT_ENGINE_CHOICES
from services.ocr import OCR_ENGINE_CHOICES
from services.service import (
    AVAILABLE_LAYOUT_ENGINES,
    AVAILABLE_OCR_ENGINES,
    GPU_CHOICES,
    MODEL_CHOICES,
    QUANT_CHOICES,
    _model_cache,
    clear_extraction_cache,
    encode_gallery_data_urls,
    extraction_cache_key,
    get_cache_stats,
    get_preview_from_path,
    get_preview_from_upload,
    load_extraction_cache,
    run_batch_from_uploads,
    run_extraction_from_path,
    run_extraction_from_upload,
    save_extraction_cache,
)


router = APIRouter()


@router.get("/api/options")
def get_options():
    return {
        "models": MODEL_CHOICES,
        "quantizations": QUANT_CHOICES,
        "gpus": GPU_CHOICES,
        "defaults": {
            "model": "Qwen2B",
            "quantization": "none",
            "gpu": "auto",
            "max_tokens": 2048,
            "ocr_engine": "paddleocr",
            "layout_engine": "doclayout_yolo",
        },
        "layout_engines": LAYOUT_ENGINE_CHOICES,
        "available_layout_engines": AVAILABLE_LAYOUT_ENGINES,
        "ocr_engines": OCR_ENGINE_CHOICES,
        "available_ocr_engines": AVAILABLE_OCR_ENGINES,
    }


@router.post("/api/extract")
async def extract_document(
    model_name: str = Form(...),
    max_tokens: int = Form(2048),
    quantization: str = Form("none"),
    gpu: str = Form("auto"),
    ocr_engine: str = Form("paddleocr"),
    layout_engine: str = Form("doclayout_yolo"),
    folder_path: str = Form(""),
    page_number: int | None = Form(default=None),
    file: UploadFile | None = File(default=None),
):
    file_bytes: bytes | None = None
    cache_key: str | None = None

    try:
        if file is not None:
            file_bytes = await file.read()
            cache_key = extraction_cache_key(
                file_bytes, page_number, model_name, quantization, ocr_engine, layout_engine
            )
            cached = load_extraction_cache(cache_key)
            if cached is not None:
                return cached

            result = run_extraction_from_upload(
                upload_bytes=file_bytes,
                filename=file.filename or "upload.pdf",
                model_name=model_name,
                max_tokens=max_tokens,
                quantization=quantization,
                gpu=gpu,
                ocr_engine=ocr_engine,
                layout_engine=layout_engine,
                page_number=page_number,
            )
        elif folder_path.strip():
            result = run_extraction_from_path(
                input_path=Path(folder_path.strip()),
                model_name=model_name,
                max_tokens=max_tokens,
                quantization=quantization,
                gpu=gpu,
                ocr_engine=ocr_engine,
                layout_engine=layout_engine,
                page_number=page_number,
            )
        else:
            raise HTTPException(status_code=400, detail="請上傳檔案或輸入資料夾路徑。")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        ) from exc

    gallery = encode_gallery_data_urls(result["gallery_items"])
    response = {
        "meta": {
            "model_name": result["model_name"],
            "model_id": result["model_id"],
            "device": result["device"],
            "quantization": result["quantization"],
            "elapsed_seconds": result["elapsed_seconds"],
            "cache_miss": result["cache_miss"],
            "json_valid": result["json_error"] is None,
            "total_pages": result.get("total_pages", 1),
            "current_page": result.get("current_page"),
            "layout_engine": result["layout_engine"],
            "layout_error": result["layout_error"],
            "layout_region_count": len(result["layout_regions"]),
            "ocr_engine": result["ocr_engine"],
            "ocr_error": result["ocr_error"],
            "ocr_block_count": len(result["ocr_blocks"]),
            "bbox_annotation_count": len(result["bbox_annotations"]),
        },
        "input_labels": result["input_labels"],
        "gallery": gallery,
        "raw_response": result["raw_response"],
        "parsed_json": result["parsed_json"],
        "json_error": result["json_error"],
        "layout_regions": result["layout_regions"],
        "parsed_layout": result["parsed_layout"],
        "document_context": result["document_context"],
        "ocr_blocks": result["ocr_blocks"],
        "bbox_annotations": result["bbox_annotations"],
    }

    if cache_key is not None:
        save_extraction_cache(cache_key, response)

    return response


@router.get("/api/preview")
def preview_document(path: str):
    try:
        if not path.strip():
            raise HTTPException(status_code=400, detail="請提供路徑。")
        return get_preview_from_path(Path(path.strip()))
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        ) from exc


@router.post("/api/preview")
async def preview_document_upload(
    file: UploadFile = File(...),
    page_number: int | None = Form(default=None),
):
    try:
        return get_preview_from_upload(
            upload_bytes=await file.read(),
            filename=file.filename or "upload.pdf",
            page_number=page_number,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        ) from exc


@router.post("/api/batch")
async def batch_extract(
    model_name: str = Form(...),
    max_tokens: int = Form(2048),
    quantization: str = Form("none"),
    gpu: str = Form("auto"),
    output_dir: str = Form("./output"),
    files: list[UploadFile] = File(default=[]),
):
    try:
        if not files:
            raise HTTPException(status_code=400, detail="請選擇至少一個 PDF 檔案。")

        uploads = [(file.filename or "upload.pdf", await file.read()) for file in files]
        result = run_batch_from_uploads(
            uploads=uploads,
            model_name=model_name,
            max_tokens=max_tokens,
            quantization=quantization,
            gpu=gpu,
            output_dir=output_dir,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="".join(traceback.format_exception_only(type(exc), exc)).strip(),
        ) from exc

    return result


@router.get("/api/model_status")
def model_status():
    if _model_cache:
        model_id, quantization, gpu = next(iter(_model_cache.keys()))
        return {"loaded": True, "model_id": model_id, "quantization": quantization, "gpu": gpu}
    return {"loaded": False, "model_id": None, "quantization": None, "gpu": None}


@router.get("/api/cache/stats")
def cache_stats():
    return get_cache_stats()


@router.delete("/api/cache")
def cache_clear():
    count = clear_extraction_cache()
    return {"cleared": count}


@router.get("/api/system_stats")
def system_stats():
    cpu_pct = psutil.cpu_percent(interval=0.1)
    ram = psutil.virtual_memory()
    gpus = []
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            text=True, timeout=3,
        )
        for line in out.strip().splitlines():
            idx, util, mem_used, mem_total = [x.strip() for x in line.split(",")]
            gpus.append({
                "index": int(idx),
                "util_pct": int(util),
                "mem_used_mb": int(mem_used),
                "mem_total_mb": int(mem_total),
            })
    except Exception:
        pass
    return {
        "cpu_pct": cpu_pct,
        "ram_used_gb": round(ram.used / 1e9, 1),
        "ram_total_gb": round(ram.total / 1e9, 1),
        "gpus": gpus,
    }


@router.get("/health")
def health_check():
    return {"status": "ok"}
