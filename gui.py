from io import BytesIO
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Thread
import time
import traceback

import gradio as gr
import torch
from PIL import Image
from transformers import TextIteratorStreamer

from model import MODEL_ID_MAP, model_function, resolve_model_id

_model_cache: dict = {}
MODEL_CHOICES = list(MODEL_ID_MAP.keys())
QUANT_CHOICES = ["none", "8bit", "4bit"]


def _get_model(model_id: str, quantization: str):
    key = (model_id, quantization)
    if key not in _model_cache:
        runner = model_function(model_id)
        model, processor = runner.build_model(quantization=quantization)
        _model_cache.clear()
        _model_cache[key] = (model, processor)
    return _model_cache[key]


def _build_gallery(image_paths, preview_images):
    if preview_images:
        return [
            (Image.open(BytesIO(img_bytes)), f"page-{i + 1}")
            for i, img_bytes in enumerate(preview_images)
        ]
    return [
        (Image.open(path).convert("RGB"), Path(str(path)).name)
        for path in image_paths
    ]


def _resolve_path(uploaded_file, folder_path: str):
    if uploaded_file is not None:
        return Path(uploaded_file)
    if folder_path and folder_path.strip():
        return Path(folder_path.strip())
    return None


def run_extraction(model_name, uploaded_file, folder_path, max_tokens, seed, quantization):
    input_path = _resolve_path(uploaded_file, folder_path)
    if input_path is None:
        yield [], "請選擇檔案或輸入資料夾路徑。", gr.update(visible=False), gr.update()
        return

    t_start = time.time()
    gallery_items = []

    try:
        runner = model_function(resolve_model_id(model_name))

        # Step 1: load images immediately
        image_paths, images, preview_images = runner.load_images(input_path)
        gallery_items = _build_gallery(image_paths, preview_images)
        yield gallery_items, "⏳ 圖片載入完成，推論中...", gr.update(visible=False)

        # Step 2: model loading status
        model_id = resolve_model_id(model_name)
        if (model_id, quantization) not in _model_cache:
            yield gallery_items, "⌛ 首次使用，載入模型中（約 30 秒）...", gr.update(visible=False)

        runner.set_seed(seed)
        model, processor = _get_model(model_id, quantization)
        messages = runner.build_messages(len(images))
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(images=images, text=input_text, return_tensors="pt")
        inputs = {
            k: v.contiguous().to(model.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs, max_new_tokens=int(max_tokens), do_sample=False, num_beams=1, streamer=streamer,
        )
        token_queue: Queue = Queue()

        def produce_tokens():
            for token in streamer:
                token_queue.put(token)
            token_queue.put(None)

        thread = Thread(target=model.generate, kwargs=generate_kwargs)
        producer = Thread(target=produce_tokens)
        thread.start()
        producer.start()

        partial = ""
        done = False
        while not done:
            try:
                token = token_queue.get(timeout=0.1)
                if token is None:
                    done = True
                else:
                    partial += token
            except Empty:
                pass

            elapsed = time.time() - t_start
            header = f"⏱ {elapsed:.1f}s  |  {model_name}  |  {model.device}  |  quant: {quantization}\n\n"
            yield gallery_items, header + partial, gr.update(visible=False)

        thread.join()
        producer.join()

        # JSON validation
        elapsed = time.time() - t_start
        header = f"⏱ {elapsed:.1f}s  |  {model_name}  |  {model.device}  |  quant: {quantization}\n\n"
        try:
            parsed = json.loads(partial.strip())
            formatted = json.dumps(parsed, ensure_ascii=False, indent=2)
            save_path = Path("/tmp/extraction_last.json")
            save_path.write_text(formatted, encoding="utf-8")
            yield gallery_items, header + formatted, gr.update(visible=True, value=str(save_path)), gr.update(value=None)
        except json.JSONDecodeError as e:
            yield gallery_items, header + partial + f"\n\n⚠️ JSON 解析失敗: {e}", gr.update(visible=False), gr.update(value=None)

    except Exception as exc:
        error_msg = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        yield gallery_items, f"錯誤：\n{error_msg}", gr.update(visible=False), gr.update(value=None)


def run_batch(model_name, batch_files, max_tokens, seed, quantization, output_dir):
    if not batch_files:
        yield "請選擇至少一個 PDF 檔案。"
        return

    out_dir = Path(output_dir.strip()) if output_dir and output_dir.strip() else Path("./output")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_id = resolve_model_id(model_name)
    if (model_id, quantization) not in _model_cache:
        yield "⌛ 載入模型中...\n"

    runner = model_function(model_id)
    runner.set_seed(int(seed))
    model, processor = _get_model(model_id, quantization)

    results = []
    total = len(batch_files)
    t_start = time.time()

    for i, file_path in enumerate(batch_files):
        path = Path(file_path)
        yield "\n".join(results) + ("\n" if results else "") + f"[{i+1}/{total}] 處理中: {path.name}..."

        try:
            _, images, _ = runner.load_images(path)
            messages = runner.build_messages(len(images))
            input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
            inputs = processor(images=images, text=input_text, return_tensors="pt")
            inputs = {
                k: v.contiguous().to(model.device) if torch.is_tensor(v) else v
                for k, v in inputs.items()
            }
            output = model.generate(**inputs, max_new_tokens=int(max_tokens), do_sample=False, num_beams=1)
            prompt_length = inputs["input_ids"].shape[-1]
            response = processor.decode(output[0][prompt_length:], skip_special_tokens=True).strip()

            out_path = out_dir / (path.stem + ".json")
            try:
                parsed = json.loads(response.strip())
                out_path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
                results.append(f"✅ {path.name} → {out_path.name}")
            except json.JSONDecodeError:
                out_path.write_text(response, encoding="utf-8")
                results.append(f"⚠️ {path.name} → JSON 解析失敗，原始輸出已儲存")

        except Exception as exc:
            results.append(f"❌ {path.name} → {exc}")

    elapsed = time.time() - t_start
    yield "\n".join(results) + f"\n\n✅ 完成！共 {total} 個檔案，耗時 {elapsed:.1f}s\n結果儲存於 {out_dir.absolute()}"


CSS = """
.gradio-container { max-width: 100% !important; padding: 12px 16px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Document Extraction Viewer", theme=gr.themes.Soft(), css=CSS) as demo:

    with gr.Tabs():

        # ── 單檔萃取 ──────────────────────────────────────────────────
        with gr.Tab("單檔萃取"):
            with gr.Row():
                model_dd = gr.Dropdown(choices=MODEL_CHOICES, value="Qwen2B", label="Model")
                quant_dd = gr.Dropdown(choices=QUANT_CHOICES, value="none", label="Quantization")
                max_tokens_num = gr.Number(value=800, label="Max Tokens", precision=0)
                seed_num = gr.Number(value=42, label="Seed", precision=0)

            with gr.Row():
                file_picker = gr.File(
                    label="選擇 PDF 或圖片",
                    file_types=[".pdf", ".jpg", ".jpeg", ".png"],
                )
                folder_path = gr.Textbox(
                    placeholder="或輸入圖片資料夾路徑 data/jpg/",
                    label="圖片資料夾路徑（選填）",
                )

            run_btn = gr.Button("▶  Run", variant="primary")

            with gr.Row():
                gallery = gr.Gallery(label="Input Document", columns=1, height=700, object_fit="contain")
                with gr.Column():
                    output_box = gr.Textbox(label="Extracted Result", lines=28, max_lines=60)
                    download_btn = gr.DownloadButton("💾  Save JSON", visible=False)

            run_btn.click(
                fn=run_extraction,
                inputs=[model_dd, file_picker, folder_path, max_tokens_num, seed_num, quant_dd],
                outputs=[gallery, output_box, download_btn, file_picker],
            )

        # ── 批次處理 ──────────────────────────────────────────────────
        with gr.Tab("批次處理"):
            gr.Markdown("選擇多個 PDF，結果 JSON 儲存到指定輸出目錄。")
            with gr.Row():
                model_dd_b = gr.Dropdown(choices=MODEL_CHOICES, value="Qwen2B", label="Model")
                quant_dd_b = gr.Dropdown(choices=QUANT_CHOICES, value="none", label="Quantization")
                max_tokens_b = gr.Number(value=800, label="Max Tokens", precision=0)
                seed_b = gr.Number(value=42, label="Seed", precision=0)
                output_dir_b = gr.Textbox(value="./output", label="輸出目錄")

            batch_files = gr.File(
                label="選擇多個 PDF",
                file_types=[".pdf"],
                file_count="multiple",
            )
            batch_run_btn = gr.Button("▶  開始批次處理", variant="primary")
            batch_output = gr.Textbox(label="批次進度與結果", lines=20)

            batch_run_btn.click(
                fn=run_batch,
                inputs=[model_dd_b, batch_files, max_tokens_b, seed_b, quant_dd_b, output_dir_b],
                outputs=[batch_output],
            )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8000, share=False)
