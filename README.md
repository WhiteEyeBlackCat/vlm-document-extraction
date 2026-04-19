# VLM Document Extraction

Automatically extract structured data from PDFs and images using Vision Language Models (VLMs). Supports both CLI and Web UI.

## Demo Preview

### Web UI Overview

![Web UI Overview](https://github.com/user-attachments/assets/de7d1864-c581-44b5-acd7-19e48273f824)

### Extraction Settings

![Batch Processing Workflow](https://github.com/user-attachments/assets/0cac7050-5420-46c7-8757-31da6d0851b4)

### Extraction Result Example

![Extraction Result Example](https://github.com/user-attachments/assets/fb371a82-b880-4907-af96-9aaf2ac305b1)

## Workflow

```text
Upload PDF/Image
    ↓
Preview / Page Conversion
    ↓
Layout Detection + OCR Assistance
    ↓
VLM Inference
    ↓
Structured JSON Output
    ↓
Export / Batch Down
```


## Features

- Accepts PDF, JPG, PNG, and TIFF input
- Multi-page PDF extraction with per-page results
- Layout detection (DocLayout-YOLO) + OCR (PaddleOCR) to assist VLM inference
- Bounding box annotation with visual overlay
- Disk cache for extraction results — identical file/params skips GPU entirely
- Web UI: batch upload, real-time progress, workspace preview, selective ZIP download

## Supported Models

| Name | HuggingFace ID |
|------|----------------|
| `Qwen2B` | `Qwen/Qwen3-VL-2B-Instruct` |
| `Qwen8B` | `Qwen/Qwen3-VL-8B-Instruct` |
| `llama` | `meta-llama/Llama-3.2-11B-Vision-Instruct` |

> Llama does not support bitsandbytes quantization. Use `--quantization none` with Llama models.

## Requirements

- Python 3.10+
- CUDA GPU (≥16 GB VRAM recommended; 4-bit quantization reduces to ≈8 GB)
- `pdfinfo` / `pdftoppm` (Poppler, for PDF-to-image conversion)
- `nvidia-smi` (for GPU monitoring in the Dashboard)

```bash
# Recommended: create a conda environment named CV
conda create -n CV python=3.10
conda activate CV
pip install torch transformers bitsandbytes pillow fastapi uvicorn psutil
pip install paddleocr        # optional: OCR engine
pip install doclayout-yolo   # optional: layout detection
```

## CLI Usage

```bash
python main.py \
  --model Qwen2B \
  --file path/to/document.pdf \
  --max_tokens 2048 \
  --quantization 4bit \
  --gpu 0
```

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | `llama` / `Qwen2B` / `Qwen8B` | required |
| `--file` | Path to a PDF or a folder containing JPG images | required |
| `--max_tokens` | Maximum tokens to generate | `300` |
| `--quantization` | `none` / `8bit` / `4bit` | `none` |
| `--gpu` | `auto` or a specific GPU index (`0`, `1`, ...) | `auto` |
| `--seed` | Random seed for deterministic generation | `42` |

## Web UI

```bash
# Start server (default: 127.0.0.1:8001)
bash run_web.sh

# Custom host/port
HOST=0.0.0.0 PORT=8080 bash run_web.sh
```

Open `http://127.0.0.1:8001` in your browser.

### Pages

| Page | Description |
|------|-------------|
| Dashboard | Session stats, live GPU/CPU status, cache management |
| Projects (Ingest) | Drag-and-drop upload, model configuration, batch submission |
| Documents (Batch) | Batch progress table, job retry, selective download |
| Extraction (Workspace) | Document preview, JSON output, bounding box overlay |

### Download Format

Select documents in the Batch page and click **Download Selected** to generate a ZIP. Each document gets its own folder:

```
batch_results.zip
├── document_a/
│   ├── result.json            # Structured extraction output
│   ├── bbox_annotations.json  # Bounding box annotations
│   └── metadata.json          # Model, timing, and config metadata
└── document_b/
    └── ...
```

## API Reference

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/options` | Available models, quantizations, and GPUs |
| `POST` | `/api/extract` | Extract a single page (file upload or folder path) |
| `POST` | `/api/preview` | Convert PDF to preview images (no model inference) |
| `GET` | `/api/model_status` | Currently loaded model info |
| `GET` | `/api/system_stats` | CPU, RAM, and per-GPU utilization |
| `GET` | `/api/cache/stats` | Cache entry count and total size |
| `DELETE` | `/api/cache` | Clear all cached results |
| `POST` | `/api/batch` | Server-side multi-file batch extraction |

## Project Structure

```
├── main.py               # CLI entry point
├── model.py              # Model loading and inference
├── prompt.py             # Prompt construction
├── parse_args.py         # CLI argument parsing
├── run_web.sh            # Web server startup script
├── services/
│   ├── service.py        # Extraction pipeline, model cache, extraction cache
│   ├── layout.py         # DocLayout-YOLO layout detection
│   ├── ocr.py            # PaddleOCR integration
│   ├── annotate.py       # Bounding box annotation
│   └── parse.py          # OCR block normalization, document context building
├── web/
│   ├── app.py            # FastAPI app with lifespan
│   ├── routes.py         # API endpoints
│   └── frontend/         # Static frontend (HTML / CSS / JS)
├── models/               # Local model weights (e.g. doclayout_yolo.pt)
├── json/                 # JSON schema definitions
└── output/
    └── cache/            # Extraction cache (auto-cleared on server shutdown)
```

## Extraction Cache

Results are cached to `output/cache/` using a key derived from `SHA256(file)[:16] + page + model + quantization + ocr_engine + layout_engine`.

- **Cache hit** — returns immediately, no GPU usage
- **Cache miss** — runs inference, saves result to disk
- **Auto-clear** — cache is wiped when the server shuts down via FastAPI lifespan
- **Manual clear** — use the Clear button on the Dashboard, or `DELETE /api/cache`
