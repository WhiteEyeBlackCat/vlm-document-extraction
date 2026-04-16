const form = document.getElementById("extract-form");
const batchForm = document.getElementById("batch-form");
const statusEl = document.getElementById("status");
const timerEl = document.getElementById("timer");
const resultEl = document.getElementById("result");
const metaLineEl = document.getElementById("meta-line");
const batchMetaLineEl = document.getElementById("batch-meta-line");
const batchResultEl = document.getElementById("batch-result");
const galleryEl = document.getElementById("gallery");
const downloadBtn = document.getElementById("download-btn");
const submitBtn = document.getElementById("submit-btn");
const batchSubmitBtn = document.getElementById("batch-submit-btn");
const singleFileInput = document.getElementById("file");
const batchFileInput = document.getElementById("batch_files");
const folderPathInput = document.getElementById("folder_path");
const tabButtons = Array.from(document.querySelectorAll(".tab-btn"));
const tabPanels = Array.from(document.querySelectorAll(".tab-panel"));
const lightboxEl = document.getElementById("lightbox");
const lightboxContentEl = document.getElementById("lightbox-content");
const lightboxTitleEl = document.getElementById("lightbox-title");
const lightboxCloseBtn = document.getElementById("lightbox-close-btn");
const zoomInBtn = document.getElementById("zoom-in-btn");
const zoomOutBtn = document.getElementById("zoom-out-btn");
const zoomResetBtn = document.getElementById("zoom-reset-btn");

let timerHandle = null;
let startedAt = null;
let lastJsonText = "";
let previewRequestId = 0;
let lightboxZoom = 1;
let lightboxKind = null;
let lightboxSrc = "";
let lightboxItem = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function formatElapsed(ms) {
  const totalSeconds = ms / 1000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds - minutes * 60;
  return `${String(minutes).padStart(2, "0")}:${seconds.toFixed(1).padStart(4, "0")}`;
}

function startTimer() {
  startedAt = performance.now();
  timerEl.textContent = "00:00.0";
  if (timerHandle !== null) {
    window.clearInterval(timerHandle);
  }
  timerHandle = window.setInterval(() => {
    timerEl.textContent = formatElapsed(performance.now() - startedAt);
  }, 100);
}

function stopTimer() {
  if (timerHandle !== null) {
    window.clearInterval(timerHandle);
    timerHandle = null;
  }
  if (startedAt !== null) {
    timerEl.textContent = formatElapsed(performance.now() - startedAt);
  }
}

function renderBoxes(container, overlays) {
  for (const overlay of overlays || []) {
    const [x1, y1, x2, y2] = overlay.bbox;
    const box = document.createElement("div");
    box.className = "bbox-box";
    if (overlay.type) {
      box.dataset.type = overlay.type;
    }
    box.style.left = `${x1 * 100}%`;
    box.style.top = `${y1 * 100}%`;
    box.style.width = `${(x2 - x1) * 100}%`;
    box.style.height = `${(y2 - y1) * 100}%`;

    const label = document.createElement("div");
    label.className = "bbox-label";
    label.textContent = overlay.label;
    box.appendChild(label);
    container.appendChild(box);
  }
}

function buildAnnotatedItems(items, annotations, layoutRegions) {
  const pageMap = new Map(items.map((item, index) => [index + 1, { ...item, overlays: [] }]));

  for (const region of layoutRegions || []) {
    const target = pageMap.get(region.page);
    if (!target || !region.bbox) {
      continue;
    }
    const width = target.width || 1;
    const height = target.height || 1;
    target.overlays.push({
      bbox: [
        region.bbox[0] / width,
        region.bbox[1] / height,
        region.bbox[2] / width,
        region.bbox[3] / height,
      ],
      label: region.label,
      type: "layout",
    });
  }

  for (const annotation of annotations || []) {
    for (const match of annotation.matches || []) {
      const target = pageMap.get(match.page);
      if (!target || !match.bbox) {
        continue;
      }
      const width = target.width || 1;
      const height = target.height || 1;
      target.overlays.push({
        bbox: [
          match.bbox[0] / width,
          match.bbox[1] / height,
          match.bbox[2] / width,
          match.bbox[3] / height,
        ],
        label: annotation.path.split(".").slice(-1)[0],
        type: "field",
      });
    }
  }
  return Array.from(pageMap.values());
}

function renderGallery(items) {
  galleryEl.innerHTML = "";
  if (!items.length) {
    galleryEl.classList.add("empty");
    galleryEl.textContent = "尚無預覽";
    return;
  }

  galleryEl.classList.remove("empty");
  for (const item of items) {
    const figure = document.createElement("figure");
    figure.className = "thumb";
    figure.tabIndex = 0;
    figure.addEventListener("click", () => openLightbox(item));
    figure.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        openLightbox(item);
      }
    });

    if (item.kind === "pdf" && item.src) {
      const frame = document.createElement("iframe");
      frame.className = "pdf-frame";
      frame.src = item.src;
      frame.title = item.name;
      figure.appendChild(frame);
    } else if (item.src) {
      const stage = document.createElement("div");
      stage.className = "preview-stage";
      const img = document.createElement("img");
      img.src = item.src;
      img.alt = item.name;
      stage.appendChild(img);
      renderBoxes(stage, item.overlays);
      figure.appendChild(stage);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "pdf-card";
      placeholder.textContent = "PDF";
      figure.appendChild(placeholder);
    }

    const caption = document.createElement("figcaption");
    caption.textContent = item.name;
    figure.appendChild(caption);
    galleryEl.appendChild(figure);
  }
}

function renderLocalPreview(files) {
  if (!files.length) {
    renderGallery([]);
    return;
  }

  const items = files.map((file) => {
    if (file.type.startsWith("image/")) {
      return {
        name: file.name,
        src: URL.createObjectURL(file),
        kind: "image",
      };
    }
    if (file.type === "application/pdf" || file.name.toLowerCase().endsWith(".pdf")) {
      return {
        name: file.name,
        src: URL.createObjectURL(file),
        kind: "pdf",
      };
    }
    return {
      name: file.name,
      src: null,
      kind: "unknown",
    };
  });
  renderGallery(items);
}

function renderLightboxContent() {
  lightboxContentEl.innerHTML = "";
  if (!lightboxSrc || !lightboxItem) {
    return;
  }

  if (lightboxKind === "pdf") {
    const frame = document.createElement("iframe");
    frame.className = "lightbox-frame";
    frame.src = lightboxSrc;
    frame.title = lightboxTitleEl.textContent;
    lightboxContentEl.appendChild(frame);
    zoomInBtn.disabled = true;
    zoomOutBtn.disabled = true;
    zoomResetBtn.disabled = true;
    zoomResetBtn.textContent = "PDF";
    return;
  }

  const stage = document.createElement("div");
  stage.className = "lightbox-stage";
  const preview = document.createElement("div");
  preview.className = "lightbox-preview";
  const img = document.createElement("img");
  img.className = "lightbox-image";
  img.src = lightboxSrc;
  img.alt = lightboxTitleEl.textContent;
  img.style.transform = `scale(${lightboxZoom})`;
  preview.appendChild(img);
  renderBoxes(preview, lightboxItem.overlays);
  stage.appendChild(preview);
  lightboxContentEl.appendChild(stage);
  zoomInBtn.disabled = false;
  zoomOutBtn.disabled = false;
  zoomResetBtn.disabled = false;
  zoomResetBtn.textContent = `${Math.round(lightboxZoom * 100)}%`;
}

function openLightbox(item) {
  if (!item.src) {
    return;
  }
  lightboxItem = item;
  lightboxKind = item.kind || "image";
  lightboxSrc = item.src;
  lightboxZoom = 1;
  lightboxTitleEl.textContent = item.name || "Preview";
  lightboxEl.classList.remove("hidden");
  lightboxEl.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
  renderLightboxContent();
}

function closeLightbox() {
  lightboxEl.classList.add("hidden");
  lightboxEl.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
  lightboxContentEl.innerHTML = "";
  lightboxSrc = "";
  lightboxKind = null;
  lightboxItem = null;
}

function adjustZoom(delta) {
  if (lightboxKind === "pdf") {
    return;
  }
  lightboxZoom = Math.max(0.5, Math.min(4, lightboxZoom + delta));
  renderLightboxContent();
}

function setDownloadPayload(text) {
  lastJsonText = text;
  downloadBtn.disabled = !text;
}

downloadBtn.addEventListener("click", () => {
  if (!lastJsonText) return;
  const blob = new Blob([lastJsonText], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = "extraction_result.json";
  anchor.click();
  URL.revokeObjectURL(url);
});

async function loadOptions() {
  const response = await fetch("/api/options");
  const data = await response.json();

  const selectMap = {
    model_name: data.models,
    quantization: data.quantizations,
    gpu: data.gpus,
    batch_model_name: data.models,
    batch_quantization: data.quantizations,
    batch_gpu: data.gpus,
  };

  for (const [id, values] of Object.entries(selectMap)) {
    const select = document.getElementById(id);
    select.innerHTML = "";
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = value;
      select.appendChild(option);
    }
  }

  document.getElementById("model_name").value = data.defaults.model;
  document.getElementById("quantization").value = data.defaults.quantization;
  document.getElementById("gpu").value = data.defaults.gpu;
  document.getElementById("max_tokens").value = data.defaults.max_tokens;
  document.getElementById("batch_model_name").value = data.defaults.model;
  document.getElementById("batch_quantization").value = data.defaults.quantization;
  document.getElementById("batch_gpu").value = data.defaults.gpu;
  document.getElementById("batch_max_tokens").value = data.defaults.max_tokens;
}

function activateTab(tabName) {
  tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.dataset.panel === tabName);
  });
}

async function loadPathPreview(rawPath) {
  const path = rawPath.trim();
  if (!path) {
    if (!singleFileInput.files.length) {
      renderGallery([]);
      metaLineEl.textContent = "尚未執行";
    }
    return;
  }

  const currentRequestId = ++previewRequestId;
  metaLineEl.textContent = "載入預覽中...";

  try {
    const response = await fetch(`/api/preview?path=${encodeURIComponent(path)}`);
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Preview failed");
    }
    if (currentRequestId !== previewRequestId) {
      return;
    }
    renderGallery(data.gallery || []);
    metaLineEl.textContent = data.input_labels?.length
      ? `已載入 ${data.input_labels.length} 頁預覽`
      : "已載入預覽";
  } catch (error) {
    if (currentRequestId !== previewRequestId) {
      return;
    }
    renderGallery([]);
    metaLineEl.textContent = error.message || String(error);
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  setStatus("推論中");
  submitBtn.disabled = true;
  downloadBtn.disabled = true;
  resultEl.textContent = "執行中...";
  metaLineEl.textContent = "準備中";
  startTimer();

  try {
    const formData = new FormData(form);
    formData.set("layout_engine", "doclayout_yolo");
    formData.set("ocr_engine", "paddleocr");
    if (!singleFileInput.files.length) {
      formData.delete("file");
    }

    const response = await fetch("/api/extract", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Extraction failed");
    }

    stopTimer();
    setStatus(data.meta.json_valid ? "完成" : "完成，但 JSON 需檢查");
    metaLineEl.textContent =
      `${data.meta.model_name} | ${data.meta.device} | quant: ${data.meta.quantization} | ${data.meta.elapsed_seconds}s`;
    if (data.gallery.length) {
      renderGallery(buildAnnotatedItems(data.gallery, data.bbox_annotations, data.layout_regions));
    }

    if (data.parsed_json) {
      const text = JSON.stringify(data.parsed_json, null, 2);
      const notes = [];
      if (data.meta.layout_error) {
        notes.push(`layout error: ${data.meta.layout_error}`);
      }
      if (data.meta.ocr_error) {
        notes.push(`ocr error: ${data.meta.ocr_error}`);
      }
      if (data.meta.bbox_annotation_count === 0 && data.meta.ocr_block_count > 0) {
        notes.push("bbox annotations are empty, showing layout boxes only");
      }
      resultEl.textContent = notes.length ? `${notes.join("\n")}\n\n${text}` : text;
      setDownloadPayload(text);
    } else {
      const text = `${data.raw_response}\n\nJSON parse error: ${data.json_error}`;
      resultEl.textContent = text;
      setDownloadPayload("");
    }
  } catch (error) {
    stopTimer();
    setStatus("失敗");
    metaLineEl.textContent = "執行失敗";
    resultEl.textContent = error.message || String(error);
    setDownloadPayload("");
  } finally {
    submitBtn.disabled = false;
  }
});

batchForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  activateTab("batch");
  setStatus("批次處理中");
  batchSubmitBtn.disabled = true;
  batchResultEl.textContent = "執行中...";
  batchMetaLineEl.textContent = "準備中";
  startTimer();

  try {
    const formData = new FormData();
    formData.set("model_name", document.getElementById("batch_model_name").value);
    formData.set("quantization", document.getElementById("batch_quantization").value);
    formData.set("gpu", document.getElementById("batch_gpu").value);
    formData.set("layout_engine", "doclayout_yolo");
    formData.set("ocr_engine", "paddleocr");
    formData.set("max_tokens", document.getElementById("batch_max_tokens").value);
    formData.set("output_dir", document.getElementById("output_dir").value);
    for (const file of batchFileInput.files) {
      formData.append("files", file, file.name);
    }

    const response = await fetch("/api/batch", {
      method: "POST",
      body: formData,
    });
    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Batch extraction failed");
    }

    stopTimer();
    setStatus("批次完成");
    batchMetaLineEl.textContent =
      `${data.model_name} | ${data.device} | quant: ${data.quantization} | ${data.elapsed_seconds}s | ${data.output_dir}`;
    batchResultEl.textContent = data.results.map((item) => item.message).join("\n") || "無結果";
  } catch (error) {
    stopTimer();
    setStatus("批次失敗");
    batchMetaLineEl.textContent = "執行失敗";
    batchResultEl.textContent = error.message || String(error);
  } finally {
    batchSubmitBtn.disabled = false;
  }
});

singleFileInput.addEventListener("change", () => {
  activateTab("single");
  renderLocalPreview(Array.from(singleFileInput.files));
  metaLineEl.textContent = singleFileInput.files.length
    ? `已載入 ${singleFileInput.files.length} 個檔案預覽`
    : "尚未執行";
  if (singleFileInput.files.length) {
    folderPathInput.value = "";
  }
});

folderPathInput.addEventListener("change", () => {
  activateTab("single");
  if (folderPathInput.value.trim()) {
    singleFileInput.value = "";
  }
  loadPathPreview(folderPathInput.value);
});

folderPathInput.addEventListener("blur", () => {
  if (!singleFileInput.files.length) {
    loadPathPreview(folderPathInput.value);
  }
});

lightboxCloseBtn.addEventListener("click", closeLightbox);
zoomInBtn.addEventListener("click", () => adjustZoom(0.25));
zoomOutBtn.addEventListener("click", () => adjustZoom(-0.25));
zoomResetBtn.addEventListener("click", () => {
  lightboxZoom = 1;
  renderLightboxContent();
});
lightboxEl.addEventListener("click", (event) => {
  if (event.target.dataset.closeLightbox === "true") {
    closeLightbox();
  }
});
document.addEventListener("keydown", (event) => {
  if (lightboxEl.classList.contains("hidden")) {
    return;
  }
  if (event.key === "Escape") {
    closeLightbox();
  } else if (event.key === "+" || event.key === "=") {
    adjustZoom(0.25);
  } else if (event.key === "-") {
    adjustZoom(-0.25);
  }
});

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activateTab(button.dataset.tab);
  });
});

loadOptions().catch((error) => {
  setStatus("初始化失敗");
  resultEl.textContent = error.message || String(error);
});
