from html import escape
from http.server import BaseHTTPRequestHandler, HTTPServer
import base64
from io import BytesIO
from pathlib import Path
import traceback
from urllib.parse import parse_qs, quote, unquote, urlparse

from PIL import Image

from model import MODEL_ID_MAP, model_function, resolve_model_id


HOST = "127.0.0.1"
PORT = 8000
QUANTIZATION_OPTIONS = ["none", "8bit", "4bit"]


def render_page(
    model_name="Qwen2B",
    input_value="",
    max_tokens="300",
    seed="42",
    quantization="none",
    result="",
    error="",
    image_paths=None,
    preview_images=None,
):
    image_paths = image_paths or []
    preview_images = preview_images or []

    option_html = []
    for name in MODEL_ID_MAP:
        selected = " selected" if name == model_name else ""
        option_html.append(f'<option value="{name}"{selected}>{name}</option>')

    quantization_html = []
    for name in QUANTIZATION_OPTIONS:
        selected = " selected" if name == quantization else ""
        quantization_html.append(f'<option value="{name}"{selected}>{name}</option>')

    image_html = []
    for index, image_path in enumerate(image_paths):
        if preview_images:
            encoded_bytes = base64.b64encode(preview_images[index]).decode("ascii")
            image_src = f"data:image/jpeg;base64,{encoded_bytes}"
        else:
            encoded_path = quote(str(image_path))
            image_src = f"/image?path={encoded_path}"

        image_html.append(
            f"""
            <div class="image-card">
              <div class="image-name">{escape(Path(str(image_path)).name)}</div>
              <img src="{image_src}" alt="{escape(Path(str(image_path)).name)}" />
            </div>
            """
        )

    status_html = ""
    if error:
        status_html = f'<div class="status error">{escape(error)}</div>'
    elif result:
        status_html = '<div class="status success">Extraction completed.</div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Document Extraction Viewer</title>
  <style>
    :root {{
      --bg: #f5f1e8;
      --panel: #fffdf8;
      --ink: #1f2933;
      --muted: #5b6570;
      --accent: #a64b2a;
      --accent-soft: #f2d7c8;
      --border: #d7c7b6;
      --ok: #2d6a4f;
      --err: #9b2226;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, #f3e2cf 0, transparent 28%),
        linear-gradient(180deg, #f7f2ea 0%, var(--bg) 100%);
    }}
    .page {{
      max-width: 1600px;
      margin: 0 auto;
      padding: 24px;
    }}
    .hero {{
      margin-bottom: 18px;
    }}
    .hero h1 {{
      margin: 0 0 8px;
      font-size: 32px;
    }}
    .hero p {{
      margin: 0;
      color: var(--muted);
    }}
    .controls {{
      display: grid;
      grid-template-columns: 160px 1fr 130px 110px 130px 140px;
      gap: 12px;
      align-items: end;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 16px;
      margin-bottom: 18px;
      box-shadow: 0 10px 30px rgba(53, 40, 28, 0.06);
    }}
    label {{
      display: block;
      font-size: 13px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    input, select, button {{
      width: 100%;
      border-radius: 10px;
      border: 1px solid var(--border);
      padding: 10px 12px;
      font-size: 14px;
      background: white;
    }}
    button {{
      background: var(--accent);
      color: white;
      border: none;
      cursor: pointer;
      font-weight: 600;
    }}
    button:hover {{
      filter: brightness(1.05);
    }}
    .status {{
      margin-bottom: 18px;
      padding: 12px 14px;
      border-radius: 12px;
      font-weight: 600;
    }}
    .status.success {{
      background: #e7f5ec;
      color: var(--ok);
      border: 1px solid #b7dec7;
    }}
    .status.error {{
      background: #fdecec;
      color: var(--err);
      border: 1px solid #efc2c4;
    }}
    .content {{
      display: grid;
      grid-template-columns: minmax(0, 1.4fr) minmax(360px, 0.9fr);
      gap: 18px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px;
      min-height: 640px;
      box-shadow: 0 10px 30px rgba(53, 40, 28, 0.06);
    }}
    .panel h2 {{
      margin: 0 0 12px;
      font-size: 18px;
    }}
    .image-grid {{
      display: grid;
      gap: 14px;
    }}
    .image-card {{
      border: 1px solid var(--border);
      border-radius: 14px;
      overflow: hidden;
      background: #fff;
    }}
    .image-name {{
      padding: 10px 12px;
      background: var(--accent-soft);
      color: #6e2e14;
      font-weight: 600;
      border-bottom: 1px solid var(--border);
    }}
    .image-card img {{
      display: block;
      width: 100%;
      height: auto;
      background: #faf6ef;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 13px;
      line-height: 1.5;
      font-family: "Courier New", monospace;
      min-height: 560px;
    }}
    .empty {{
      color: var(--muted);
      padding-top: 12px;
    }}
    @media (max-width: 960px) {{
      .controls {{
        grid-template-columns: 1fr;
      }}
      .content {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <h1>Document Extraction Viewer</h1>
      <p>Preview your document images on the left and inspect the extracted JSON on the right.</p>
    </div>

    <form method="post" class="controls">
      <div>
        <label for="model">Model</label>
        <select id="model" name="model">
          {''.join(option_html)}
        </select>
      </div>
      <div>
        <label for="input_path">PDF Or Image Folder</label>
        <input id="input_path" name="input_path" value="{escape(input_value)}" placeholder="data/example.pdf or data/jpg/..." required />
      </div>
      <div>
        <label for="max_tokens">Max Tokens</label>
        <input id="max_tokens" name="max_tokens" value="{escape(max_tokens)}" required />
      </div>
      <div>
        <label for="seed">Seed</label>
        <input id="seed" name="seed" value="{escape(seed)}" required />
      </div>
      <div>
        <label for="quantization">Quantization</label>
        <select id="quantization" name="quantization">
          {''.join(quantization_html)}
        </select>
      </div>
      <div>
        <button type="submit">Run Extraction</button>
      </div>
    </form>

    {status_html}

    <div class="content">
      <section class="panel">
        <h2>Input Document</h2>
        <div class="image-grid">
          {''.join(image_html) if image_html else '<div class="empty">Submit a PDF or image folder to preview document pages here.</div>'}
        </div>
      </section>

      <section class="panel">
        <h2>Extracted Result</h2>
        <pre>{escape(result) if result else 'Run extraction to see the output here.'}</pre>
      </section>
    </div>
  </div>
</body>
</html>
"""


class ExtractionHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_html(render_page())
            return

        if parsed.path == "/image":
            params = parse_qs(parsed.query)
            raw_path = params.get("path", [""])[0]
            self._serve_image(Path(unquote(raw_path)))
            return

        self.send_error(404, "Not Found")

    def do_POST(self):
        if self.path != "/":
            self.send_error(404, "Not Found")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(content_length).decode("utf-8")
        form = parse_qs(body)

        model_name = form.get("model", ["llama"])[0]
        input_value = form.get("input_path", [""])[0].strip()
        max_tokens = form.get("max_tokens", ["300"])[0].strip()
        seed = form.get("seed", ["42"])[0].strip()
        quantization = form.get("quantization", ["none"])[0].strip()

        image_paths = []
        preview_images = []
        result_text = ""
        error_text = ""

        try:
            runner = model_function(resolve_model_id(model_name))
            input_path = Path(input_value)
            result = runner.run_inference(
                input_path,
                int(max_tokens),
                int(seed),
                quantization=quantization,
            )
            result_text = result["response"]
            image_paths = result["image_paths"]
            preview_images = result.get("preview_images") or []
        except Exception as exc:
            error_text = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )

        self._send_html(
            render_page(
                model_name=model_name,
                input_value=input_value,
                max_tokens=max_tokens,
                seed=seed,
                quantization=quantization,
                result=result_text,
                error=error_text,
                image_paths=image_paths,
                preview_images=preview_images,
            )
        )

    def _serve_image(self, path: Path):
        if not path.exists() or not path.is_file():
            self.send_error(404, "Image not found")
            return

        with Image.open(path) as image:
            preview = image.convert("RGB")
            preview.thumbnail((1200, 1600))
            buffer = BytesIO()
            preview.save(buffer, format="JPEG", quality=90)
            image_bytes = buffer.getvalue()

        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(image_bytes)))
        self.end_headers()
        self.wfile.write(image_bytes)

    def _send_html(self, html: str):
        payload = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    server = HTTPServer((HOST, PORT), ExtractionHandler)
    print(f"Serving on http://{HOST}:{PORT}")
    print("If you are on a remote machine, use SSH port forwarding to open it in your browser.")
    server.serve_forever()


if __name__ == "__main__":
    main()
