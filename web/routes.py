from __future__ import annotations

from pathlib import Path
import traceback

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from services.service import (
    GPU_CHOICES,
    MODEL_CHOICES,
    QUANT_CHOICES,
    encode_gallery_data_urls,
    get_preview_from_path,
    run_batch_from_uploads,
    run_extraction_from_path,
    run_extraction_from_upload,
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
            "max_tokens": 300,
        },
    }


@router.post("/api/extract")
async def extract_document(
    model_name: str = Form(...),
    max_tokens: int = Form(300),
    quantization: str = Form("none"),
    gpu: str = Form("auto"),
    folder_path: str = Form(""),
    file: UploadFile | None = File(default=None),
):
    try:
        if file is not None:
            result = run_extraction_from_upload(
                upload_bytes=await file.read(),
                filename=file.filename or "upload.pdf",
                model_name=model_name,
                max_tokens=max_tokens,
                quantization=quantization,
                gpu=gpu,
            )
        elif folder_path.strip():
            result = run_extraction_from_path(
                input_path=Path(folder_path.strip()),
                model_name=model_name,
                max_tokens=max_tokens,
                quantization=quantization,
                gpu=gpu,
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
    return {
        "meta": {
            "model_name": result["model_name"],
            "model_id": result["model_id"],
            "device": result["device"],
            "quantization": result["quantization"],
            "elapsed_seconds": result["elapsed_seconds"],
            "cache_miss": result["cache_miss"],
            "json_valid": result["json_error"] is None,
        },
        "input_labels": result["input_labels"],
        "gallery": gallery,
        "raw_response": result["raw_response"],
        "parsed_json": result["parsed_json"],
        "json_error": result["json_error"],
    }


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


@router.post("/api/batch")
async def batch_extract(
    model_name: str = Form(...),
    max_tokens: int = Form(300),
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


@router.get("/health")
def health_check():
    return {"status": "ok"}
