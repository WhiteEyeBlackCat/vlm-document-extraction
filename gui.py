from io import BytesIO
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


# Cache: avoid reloading model weights on every request
_model_cache: dict = {}  # key: (model_id, quantization) → (model, processor)


def _get_model(model_id: str, quantization: str):
    key = (model_id, quantization)
    if key not in _model_cache:
        runner = model_function(model_id)
        model, processor = runner.build_model(quantization=quantization)
        _model_cache.clear()          # release previous model from GPU memory
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


def run_extraction(model_name: str, input_path: str, max_tokens: int, seed: int, quantization: str):
    if not input_path or not input_path.strip():
        yield [], "請輸入 PDF 路徑或圖片資料夾路徑。"
        return

    t_start = time.time()

    try:
        runner = model_function(resolve_model_id(model_name))

        # Step 1: load images and show them immediately
        image_paths, images, preview_images = runner.load_images(Path(input_path.strip()))
        gallery_items = _build_gallery(image_paths, preview_images)
        yield gallery_items, "⏳ 圖片載入完成，推論中..."

        # Step 2: run inference (model loaded once and cached)
        runner.set_seed(seed)
        model, processor = _get_model(resolve_model_id(model_name), quantization)
        messages = runner.build_messages(len(images))
        input_text = processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = processor(images=images, text=input_text, return_tensors="pt")
        inputs = {
            k: v.contiguous().to(model.device) if torch.is_tensor(v) else v
            for k, v in inputs.items()
        }

        streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
        generate_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            num_beams=1,
            streamer=streamer,
        )
        token_queue: Queue = Queue()

        def produce_tokens():
            for token in streamer:
                token_queue.put(token)
            token_queue.put(None)  # sentinel: generation done

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
                pass  # no new token yet, just refresh the timer

            elapsed = time.time() - t_start
            header = f"⏱ {elapsed:.1f}s  |  {model_name}  |  {model.device}  |  quant: {quantization}\n\n"
            yield gallery_items, header + partial

        thread.join()
        producer.join()

    except Exception as exc:
        error_msg = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        yield gallery_items if "gallery_items" in locals() else [], f"錯誤：\n{error_msg}"
        return


CSS = """
.gradio-container { max-width: 100% !important; padding: 12px 16px !important; }
footer { display: none !important; }
"""

with gr.Blocks(title="Document Extraction Viewer", theme=gr.themes.Soft(), css=CSS) as demo:
    gr.Markdown("# Document Extraction Viewer")
    gr.Markdown("輸入 PDF 或圖片資料夾路徑，選擇模型後按 **Run** 執行萃取。")

    with gr.Row():
        model_dd = gr.Dropdown(
            choices=list(MODEL_ID_MAP.keys()),
            value="Qwen2B",
            label="Model",
        )
        quant_dd = gr.Dropdown(
            choices=["none", "8bit", "4bit"],
            value="none",
            label="Quantization",
        )
        max_tokens_num = gr.Number(value=300, label="Max Tokens", precision=0)
        seed_num = gr.Number(value=42, label="Seed", precision=0)

    input_path_box = gr.Textbox(
        placeholder="data/example.pdf  或  data/jpg/",
        label="PDF 或圖片資料夾路徑",
    )

    run_btn = gr.Button("Run", variant="primary")

    with gr.Row():
        gallery = gr.Gallery(label="Input Document", columns=1, height=700, object_fit="contain")
        output_box = gr.Textbox(label="Extracted Result", lines=30, max_lines=60)

    run_btn.click(
        fn=run_extraction,
        inputs=[model_dd, input_path_box, max_tokens_num, seed_num, quant_dd],
        outputs=[gallery, output_box],
    )


if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=8000, share=False)
