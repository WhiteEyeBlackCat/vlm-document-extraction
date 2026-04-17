/* ===== Industrial OCR — Admin Console App ===== */

// ── State ──────────────────────────────────────
const state = {
  page: "dashboard",
  options: null,
  stagedFiles: [],
  jobs: [],            // { id, filename, hash, status, size, ts, result }
  batchId: null,
  activeJobs: 0,
  extractionMode: "structured",
  logs: [],
  workspace: null,     // last extraction result
  sessionStats: { total: 0, success: 0, elapsed: [], conf: [] },
  logTimer: null,
};

// ── DOM references ─────────────────────────────
const pages    = { dashboard: "page-dashboard", ingest: "page-ingest", batch: "page-batch", workspace: "page-workspace" };
const navItems = document.querySelectorAll("[data-nav]");

// Dashboard
const dashDocsCount  = document.getElementById("dash-docs-count");
const dashDocsGreen  = document.getElementById("dash-docs-delta");
const dashSuccessEl  = document.getElementById("dash-success-rate");
const dashConfEl     = document.getElementById("dash-avg-conf");
const dashActiveEl   = document.getElementById("dash-active-jobs");
const dashProjectsList = document.getElementById("dash-projects-list");
const dashActivityLog  = document.getElementById("dash-activity-log");
const nodeCpu = document.getElementById("node-cpu");
const nodeMem = document.getElementById("node-mem");
const nodeLat = document.getElementById("node-lat");

// Ingest
const hiddenFile     = document.getElementById("hidden-file");
const dropZone       = document.getElementById("drop-zone");
const browseBtn      = document.getElementById("browse-btn");
const ingestUploadTrigger = document.getElementById("ingest-upload-trigger");
const stagedFilesList = document.getElementById("staged-files-list");
const stagedCountTxt = document.getElementById("staged-count-txt");
const clearStagedBtn = document.getElementById("clear-staged-btn");
const processBatchBtn = document.getElementById("process-batch-btn");
const ingestModelSel = document.getElementById("ingest-model");
const ingestQuantSel = document.getElementById("ingest-quant");
const ingestGpuSel   = document.getElementById("ingest-gpu");
const ingestMaxTok   = document.getElementById("ingest-max-tokens");
const throughputVal  = document.getElementById("throughput-val");
const extModes       = document.querySelectorAll(".ext-mode");

// Batch
const batchActiveCount  = document.getElementById("batch-active-count");
const batchFailedCount  = document.getElementById("batch-failed-count");
const batchCurrentId    = document.getElementById("batch-current-id");
const batchEfficiency   = document.getElementById("batch-efficiency");
const batchEffDelta     = document.getElementById("batch-efficiency-delta");
const capBar      = document.getElementById("cap-bar");
const capProcessed = document.getElementById("cap-processed");
const capRemaining = document.getElementById("cap-remaining");
const jobsTbody   = document.getElementById("jobs-tbody");
const orchLogsBody = document.getElementById("orch-logs-body");
const retryFailedBtn = document.getElementById("retry-failed-btn");
const downloadAllBtn = document.getElementById("download-all-btn");
const jobsPaginationInfo = document.getElementById("jobs-pagination-info");

// Workspace
const wsTitle       = document.getElementById("ws-title");
const wsStatusBadge = document.getElementById("ws-status-badge");
const wsFileCrumb   = document.getElementById("ws-file-crumb");
const wsBatchCrumb  = document.getElementById("ws-project-crumb");
const wsPreviewBody = document.getElementById("ws-preview-body");
const wsJsonBody    = document.getElementById("ws-json-body");
const wsJsonPre     = document.getElementById("ws-json-pre");
const wsAccuracy    = document.getElementById("ws-accuracy");
const wsConfidence  = document.getElementById("ws-confidence");
const wsTime        = document.getElementById("ws-time");
const wsCopyJsonBtn = document.getElementById("ws-copy-json-btn");
const wsRerunBtn    = document.getElementById("ws-rerun-btn");
const wsExportCsvBtn = document.getElementById("ws-export-csv-btn");
const wsZoomIn   = document.getElementById("ws-zoom-in");
const wsZoomOut  = document.getElementById("ws-zoom-out");
const wsZoomReset = document.getElementById("ws-zoom-reset");
const wsFullscreen = document.getElementById("ws-fullscreen");

// Lightbox
const lightboxEl      = document.getElementById("lightbox");
const lightboxContent = document.getElementById("lightbox-content");
const lightboxTitle   = document.getElementById("lightbox-title");
const lightboxClose   = document.getElementById("lightbox-close-btn");
const lightboxBackdrop = document.getElementById("lightbox-backdrop");
const zoomInBtn   = document.getElementById("zoom-in-btn");
const zoomOutBtn  = document.getElementById("zoom-out-btn");
const zoomResetBtn = document.getElementById("zoom-reset-btn");

let lightboxZoom = 1;
let lightboxKind = null;
let lightboxSrc  = "";
let lightboxItem = null;
let wsZoom = 1;
let wsGallery = [];

// ── Navigation ─────────────────────────────────
function navigate(page) {
  if (!pages[page]) return;
  state.page = page;
  Object.entries(pages).forEach(([k, id]) => {
    document.getElementById(id).classList.toggle("active", k === page);
  });
  navItems.forEach(btn => {
    btn.classList.toggle("active", btn.dataset.nav === page);
  });
  document.querySelectorAll(".header-nav-link").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.nav === page);
  });
  if (page === "dashboard") refreshDashboard();
  if (page === "batch") refreshBatchPage();
}

navItems.forEach(btn => btn.addEventListener("click", () => navigate(btn.dataset.nav)));
document.querySelectorAll(".header-nav-link").forEach(btn => btn.addEventListener("click", () => navigate(btn.dataset.nav)));

// ── Options ────────────────────────────────────
async function loadOptions() {
  try {
    const res  = await fetch("/api/options");
    const data = await res.json();
    state.options = data;

    populate(ingestModelSel, data.models);
    populate(ingestQuantSel, data.quantizations);
    populate(ingestGpuSel,   data.gpus, gpuLabel);

    if (data.defaults) {
      setVal(ingestModelSel, data.defaults.model);
      setVal(ingestQuantSel, data.defaults.quantization);
      setVal(ingestGpuSel,   data.defaults.gpu);
      ingestMaxTok.value = data.defaults.max_tokens || 2048;
    }
    updateThroughput();
  } catch {
    addLog("WARN", "Failed to load API options. Using defaults.");
  }
}

function populate(sel, arr, labelFn) {
  sel.innerHTML = "";
  (arr || []).forEach(v => {
    const o = document.createElement("option");
    o.value = v;
    o.textContent = labelFn ? labelFn(v) : v;
    sel.appendChild(o);
  });
}

function gpuLabel(v) {
  if (v === "auto") return "Auto (device_map)";
  return `GPU ${v}`;
}
function setVal(sel, val) { if ([...sel.options].some(o => o.value === val)) sel.value = val; }

function updateThroughput() {
  const throughput = Math.floor(80 + Math.random() * 80);
  throughputVal.textContent = `${throughput} pages/min`;
}

// ── Extraction modes ───────────────────────────
extModes.forEach(el => {
  el.addEventListener("click", () => {
    extModes.forEach(m => m.classList.remove("selected"));
    el.classList.add("selected");
    state.extractionMode = el.dataset.mode;
  });
});

// ── File staging ───────────────────────────────
browseBtn.addEventListener("click", () => hiddenFile.click());
ingestUploadTrigger.addEventListener("click", () => { navigate("ingest"); setTimeout(() => hiddenFile.click(), 100); });
hiddenFile.addEventListener("change", () => addFiles(Array.from(hiddenFile.files)));

dropZone.addEventListener("click", e => { if (e.target === dropZone || e.target.classList.contains("drop-zone-title") || e.target.classList.contains("drop-zone-sub") || e.target.classList.contains("drop-zone-icon")) hiddenFile.click(); });
dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("dragover"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
dropZone.addEventListener("drop", e => {
  e.preventDefault();
  dropZone.classList.remove("dragover");
  addFiles(Array.from(e.dataTransfer.files));
});

const LANG_OPTIONS = [
  { value: "auto", label: "Auto Detect" },
  { value: "en",    label: "English (US)" },
  { value: "zh-tw", label: "Chinese (TW)" },
  { value: "zh-cn", label: "Chinese (CN)" },
  { value: "ja",    label: "Japanese" },
  { value: "ko",    label: "Korean" },
  { value: "de",    label: "German" },
  { value: "fr",    label: "French" },
];

function addFiles(files) {
  const globalLang = document.getElementById("ingest-language").value || "auto";
  files.forEach(f => {
    if (!state.stagedFiles.find(s => s.name === f.name && s.size === f.size)) {
      state.stagedFiles.push({ file: f, lang: globalLang });
    }
  });
  renderStagedFiles();
}

function renderStagedFiles() {
  const count = state.stagedFiles.length;
  stagedCountTxt.textContent = `${count} file${count !== 1 ? "s" : ""} staged`;
  processBatchBtn.disabled = count === 0;

  stagedFilesList.innerHTML = "";
  state.stagedFiles.forEach((entry, i) => {
    const f = entry.file;
    const sizeStr = formatBytes(f.size);
    const langOpts = LANG_OPTIONS.map(o =>
      `<option value="${o.value}"${entry.lang === o.value ? " selected" : ""}>${o.label}</option>`
    ).join("");
    const div = document.createElement("div");
    div.className = "staged-file";
    div.innerHTML = `
      <div class="staged-file-top">
        <div class="staged-file-name"><span class="staged-file-icon">${fileIcon(f.name)}</span>${escHtml(f.name)}</div>
        <span class="staged-file-pct" id="sf-pct-${i}">Staged</span>
      </div>
      <div class="staged-file-progress"><div class="staged-file-bar" id="sf-bar-${i}" style="width:0%"></div></div>
      <div class="staged-file-sub">
        <span>${sizeStr}</span>
        <span style="display:flex;align-items:center;gap:6px">
          <select class="staged-file-lang" data-idx="${i}">${langOpts}</select>
          <button class="staged-file-remove" data-idx="${i}" title="Remove">&#10005;</button>
        </span>
      </div>
    `;
    stagedFilesList.appendChild(div);
  });

  // lang change
  stagedFilesList.querySelectorAll(".staged-file-lang").forEach(sel => {
    sel.addEventListener("change", () => {
      state.stagedFiles[+sel.dataset.idx].lang = sel.value;
    });
  });
  // remove
  stagedFilesList.querySelectorAll(".staged-file-remove").forEach(btn => {
    btn.addEventListener("click", () => {
      state.stagedFiles.splice(+btn.dataset.idx, 1);
      renderStagedFiles();
    });
  });
}

clearStagedBtn.addEventListener("click", () => {
  state.stagedFiles = [];
  hiddenFile.value = "";
  renderStagedFiles();
});

// ── Process batch ──────────────────────────────
processBatchBtn.addEventListener("click", async () => {
  if (!state.stagedFiles.length) return;

  const batchName = document.getElementById("ingest-batch-name").value.trim() ||
    `BATCH-${Math.floor(Math.random() * 900 + 100)}-XT-${new Date().getFullYear()}`;
  state.batchId = batchName;

  // Add jobs to state
  const now = new Date();
  const newJobs = state.stagedFiles.map((entry, i) => ({
    id: `${batchName}-${i}`,
    filename: entry.file.name,
    hash: shortHash(entry.file.name + entry.file.size),
    status: "pending",
    size: entry.file.size,
    lang: entry.lang,
    ts: new Date(now.getTime() + i * 1500),
    result: null,
    file: entry.file,
  }));
  state.jobs = [...state.jobs, ...newJobs];
  state.activeJobs += newJobs.length;

  navigate("batch");
  refreshBatchPage();

  processBatchBtn.disabled = true;
  addLog("CORE_ORCHESTRATOR", `Batch ${batchName} submitted — ${newJobs.length} file(s) queued.`);

  // Simulate file upload progress
  state.stagedFiles.forEach((_e, i) => animateUploadProgress(i));

  // Build FormData and send
  const fd = new FormData();
  fd.set("model_name",    ingestModelSel.value || "Qwen2B");
  fd.set("quantization",  ingestQuantSel.value || "none");
  fd.set("gpu",           ingestGpuSel.value   || "auto");
  fd.set("max_tokens",    ingestMaxTok.value    || "300");
  fd.set("output_dir",    "./output");
  state.stagedFiles.forEach(entry => fd.append("files", entry.file, entry.file.name));

  // Mark all as processing
  setTimeout(() => {
    newJobs.forEach(j => { j.status = "processing"; });
    refreshBatchPage();
    addLog("CORE_ORCHESTRATOR", `Processing ${newJobs.length} entities — pipeline active.`);
  }, 800);

  // Start fake log ticker
  const ticker = setInterval(() => addLog("INFO", fakeLogMsg()), 3000);

  try {
    const res  = await fetch("/api/batch", { method: "POST", body: fd });
    const data = await res.json();
    clearInterval(ticker);

    if (!res.ok) throw new Error(data.detail || "Batch failed");

    // Map results back to jobs
    data.results.forEach((item, i) => {
      if (newJobs[i]) {
        const ok = item.success ?? item.json_valid ?? (item.message && (item.message.startsWith("✅") || item.message.startsWith("⚠️")));
        newJobs[i].status  = ok ? "completed" : "failed";
        if (!ok && item.message) {
          const errShort = item.message.replace(/^❌\s*\S+\s*→\s*/, "").slice(0, 120);
          addLog("CRITICAL", `${newJobs[i].filename}: ${errShort}`);
        }
        newJobs[i].result  = item;
        newJobs[i].elapsed = item.elapsed_seconds;
        state.sessionStats.total++;
        if (item.success) state.sessionStats.success++;
        if (item.elapsed_seconds) state.sessionStats.elapsed.push(item.elapsed_seconds);
      }
    });
    state.activeJobs = Math.max(0, state.activeJobs - newJobs.length);
    addLog("INFO", `Batch ${batchName} → PARTIAL_COMPLETION. ${data.results.filter(r => r.success).length}/${newJobs.length} succeeded.`);
    addActivityEvent("batch", `Batch ${batchName} complete`, `${data.results.filter(r => r.success).length} / ${newJobs.length} files succeeded`);
  } catch (err) {
    clearInterval(ticker);
    newJobs.forEach(j => { if (j.status === "processing") j.status = "failed"; });
    state.activeJobs = Math.max(0, state.activeJobs - newJobs.length);
    addLog("CRITICAL", `Batch ${batchName} failed: ${err.message}`);
  }

  refreshBatchPage();
  refreshDashboard();
  // Clear staged files after submission
  state.stagedFiles = [];
  hiddenFile.value = "";
  renderStagedFiles();
  processBatchBtn.disabled = false;
});

function animateUploadProgress(i) {
  let pct = 0;
  const bar = document.getElementById(`sf-bar-${i}`);
  const pctEl = document.getElementById(`sf-pct-${i}`);
  if (!bar) return;
  const iv = setInterval(() => {
    pct = Math.min(100, pct + Math.random() * 25 + 5);
    bar.style.width = `${pct}%`;
    if (pctEl) pctEl.textContent = `${Math.round(pct)}%`;
    if (pct >= 100) clearInterval(iv);
  }, 300);
}

// ── Single file extraction (from workspace Re-Run) ──
async function runSingleExtract(file) {
  const fd = new FormData();
  fd.set("model_name",    ingestModelSel.value || "Qwen2B");
  fd.set("quantization",  ingestQuantSel.value || "none");
  fd.set("gpu",           ingestGpuSel.value   || "auto");
  fd.set("max_tokens",    ingestMaxTok.value    || "300");
  fd.set("layout_engine", "doclayout_yolo");
  fd.set("ocr_engine",    "paddleocr");
  fd.append("file", file, file.name);

  wsStatusBadge.textContent = "Extracting...";
  wsStatusBadge.className   = "badge badge-extracting";

  try {
    const res  = await fetch("/api/extract", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || "Extraction failed");

    state.workspace = data;
    renderWorkspace(data, file.name);
    addActivityEvent("extract", `Extracted ${file.name}`, `${data.meta.model_name} | ${data.meta.elapsed_seconds}s`);
    state.sessionStats.total++;
    if (data.meta.json_valid) state.sessionStats.success++;
    if (data.meta.elapsed_seconds) state.sessionStats.elapsed.push(data.meta.elapsed_seconds);
    // confidence is pushed inside renderWorkspace from real OCR scores
    refreshDashboard();
  } catch (err) {
    wsStatusBadge.textContent = "Failed";
    wsStatusBadge.className   = "badge badge-failed";
    wsJsonPre.innerHTML = `<span style="color:#f97583">${escHtml(err.message)}</span>`;
  }
}

// ── Batch jobs page ────────────────────────────
function refreshBatchPage() {
  const all       = state.jobs;
  const active    = all.filter(j => j.status === "processing" || j.status === "pending").length;
  const failed    = all.filter(j => j.status === "failed").length;
  const completed = all.filter(j => j.status === "completed").length;
  const total     = all.length;

  batchActiveCount.textContent = active;
  batchFailedCount.textContent = failed;
  batchCurrentId.textContent   = state.batchId ? `ID: ${state.batchId}` : "No batch loaded";

  const effPct = total > 0 ? Math.round((completed / total) * 100) : 0;
  batchEfficiency.textContent = total > 0 ? `${effPct}%` : "—";
  batchEffDelta.textContent   = total > 0 ? `+${Math.min(5, Math.floor(Math.random() * 3))}% vs last batch` : "";

  capProcessed.textContent = completed;
  capRemaining.textContent = total - completed;
  capBar.style.width       = total > 0 ? `${(completed / total) * 100}%` : "0%";

  // Table
  if (all.length === 0) {
    jobsTbody.innerHTML = `<tr><td colspan="7" style="text-align:center;padding:32px;color:var(--text-muted)">No jobs yet — upload documents to get started</td></tr>`;
    jobsPaginationInfo.textContent = "Displaying 0 entities";
    return;
  }
  jobsPaginationInfo.textContent = `Displaying 1–${all.length} of ${all.length} entities`;

  jobsTbody.innerHTML = all.map(j => `
    <tr data-job-id="${escHtml(j.id)}">
      <td><input type="checkbox" class="job-cb" /></td>
      <td>
        <div class="file-id-name">${escHtml(j.filename)}</div>
        <div class="file-id-hash">HASH: ${j.hash}</div>
      </td>
      <td>${statusBadgeHtml(j.status)}</td>
      <td><div class="complexity-bars">${complexityBars(j)}</div></td>
      <td>${formatBytes(j.size)}</td>
      <td>${formatTs(j.ts)}</td>
      <td>${j.status === "failed" ? `<button class="btn btn-secondary" style="padding:3px 8px;font-size:11px" onclick="retryJob('${escHtml(j.id)}')">Retry</button>` : j.status === "completed" ? `<button class="btn btn-secondary" style="padding:3px 8px;font-size:11px" onclick="viewJob('${escHtml(j.id)}')">View</button>` : ""}</td>
    </tr>
  `).join("");

  // Click row to view completed jobs
  jobsTbody.querySelectorAll("tr[data-job-id]").forEach(row => {
    row.addEventListener("click", e => {
      if (e.target.tagName === "BUTTON" || e.target.tagName === "INPUT") return;
      const job = state.jobs.find(j => j.id === row.dataset.jobId);
      if (job && job.status === "completed" && job.file) viewJob(job.id);
    });
  });
}

window.viewJob = function(jobId) {
  const job = state.jobs.find(j => j.id === jobId);
  if (!job || !job.file) return;
  navigate("workspace");
  wsFileCrumb.textContent = job.filename;
  wsBatchCrumb.textContent = state.batchId || "Batch";
  wsTitle.textContent = `Workspace: ${job.filename}`;
  if (state.workspace && job.result) {
    // reuse last extract data if same file; else re-run
  }
  runSingleExtract(job.file);
};

window.retryJob = function(jobId) {
  const job = state.jobs.find(j => j.id === jobId);
  if (!job) return;
  job.status = "pending";
  refreshBatchPage();
  addLog("INFO", `Retrying job for ${job.filename}...`);
  // Re-run single extract
  if (job.file) {
    job.status = "processing";
    refreshBatchPage();
    const fd = new FormData();
    fd.set("model_name", ingestModelSel.value || "Qwen2B");
    fd.set("quantization", ingestQuantSel.value || "none");
    fd.set("gpu", ingestGpuSel.value || "auto");
    fd.set("max_tokens", ingestMaxTok.value || "300");
    fd.set("layout_engine", "doclayout_yolo");
    fd.set("ocr_engine", "paddleocr");
    fd.append("file", job.file, job.file.name);
    fetch("/api/extract", { method: "POST", body: fd })
      .then(r => r.json())
      .then(data => {
        job.status  = data.meta ? (data.meta.json_valid ? "completed" : "failed") : "failed";
        job.result  = data;
        refreshBatchPage();
        addLog("INFO", `Retry of ${job.filename} → ${job.status}`);
      })
      .catch(err => {
        job.status = "failed";
        refreshBatchPage();
        addLog("CRITICAL", `Retry failed for ${job.filename}: ${err.message}`);
      });
  }
};

retryFailedBtn.addEventListener("click", () => {
  state.jobs.filter(j => j.status === "failed").forEach(j => window.retryJob(j.id));
});

downloadAllBtn.addEventListener("click", () => {
  const completed = state.jobs.filter(j => j.status === "completed" && j.result);
  if (!completed.length) { addLog("WARN", "No completed jobs to download."); return; }
  const blob = new Blob([JSON.stringify(completed.map(j => ({ file: j.filename, result: j.result })), null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = `${state.batchId || "batch"}_results.json`; a.click();
  URL.revokeObjectURL(url);
});

// ── Workspace ──────────────────────────────────
function renderWorkspace(data, filename) {
  wsStatusBadge.textContent = data.meta.json_valid ? "Extraction Complete" : "Extraction — JSON Warning";
  wsStatusBadge.className   = `badge ${data.meta.json_valid ? "badge-completed" : "badge-pending"}`;

  // Preview
  wsPreviewBody.innerHTML = "";
  const galleryItems = data.gallery || [];
  if (galleryItems.length) {
    wsGallery = buildAnnotatedItems(galleryItems, data.bbox_annotations, data.ocr_blocks);
    const div = document.createElement("div");
    div.className = "ws-gallery";
    wsGallery.forEach(item => {
      const fig = document.createElement("figure");
      fig.className = "";
      if (item.kind === "pdf" && item.src) {
        const frame = document.createElement("iframe");
        frame.className = "pdf-frame"; frame.src = item.src; frame.title = item.name;
        fig.appendChild(frame);
      } else if (item.src) {
        const stage = document.createElement("div");
        stage.className = "preview-stage";
        const img = document.createElement("img");
        img.src = item.src; img.alt = item.name;
        img.style.transform = `scale(${wsZoom})`;
        img.style.transformOrigin = "top center";
        stage.appendChild(img);
        renderBoxes(stage, item.overlays);
        fig.appendChild(stage);
        fig.addEventListener("click", () => openLightbox(item));
      }
      div.appendChild(fig);
    });
    wsPreviewBody.appendChild(div);
  } else {
    wsPreviewBody.innerHTML = `<div class="preview-placeholder"><div class="preview-placeholder-ico">&#128196;</div><div>${escHtml(filename)}</div></div>`;
  }

  // JSON
  if (data.parsed_json) {
    wsJsonPre.innerHTML = syntaxHighlight(JSON.stringify(data.parsed_json, null, 2));
  } else {
    wsJsonPre.innerHTML = `<span style="color:#f97583">${escHtml(data.raw_response || "")}</span>\n\n<span style="color:#f97583">JSON parse error: ${escHtml(data.json_error || "")}</span>`;
  }

  // Metrics — computed from real backend data
  const annotations  = data.bbox_annotations || [];
  const ocrBlocks    = data.ocr_blocks || [];
  const totalFields  = countLeafValues(data.parsed_json);

  // Field Accuracy: % of JSON leaf fields that were matched back to an OCR bbox
  if (data.meta.json_valid && totalFields > 0) {
    const matchedFields = annotations.length;
    wsAccuracy.textContent = `${Math.round((matchedFields / totalFields) * 100)}%`;
  } else {
    wsAccuracy.textContent = data.meta.json_valid ? "—" : "—";
  }

  // Text Confidence: average of PaddleOCR confidence scores across all matched blocks
  const confScores = annotations.flatMap(a => (a.matches || []).map(m => m.confidence)).filter(c => c != null && c > 0);
  if (confScores.length > 0) {
    const avg = confScores.reduce((s, v) => s + v, 0) / confScores.length;
    wsConfidence.textContent = `${(avg * 100).toFixed(1)}%`;
    state.sessionStats.conf.push(avg);
  } else if (ocrBlocks.length > 0) {
    const allConf = ocrBlocks.map(b => b.confidence).filter(c => c != null && c > 0);
    if (allConf.length > 0) {
      const avg = allConf.reduce((s, v) => s + v, 0) / allConf.length;
      wsConfidence.textContent = `${(avg * 100).toFixed(1)}%`;
      state.sessionStats.conf.push(avg);
    } else {
      wsConfidence.textContent = "—";
    }
  } else {
    wsConfidence.textContent = "—";
  }

  wsTime.textContent = data.meta.elapsed_seconds ? `${data.meta.elapsed_seconds}s` : "—";
}

wsRerunBtn.addEventListener("click", () => {
  const job = state.jobs.find(j => j.status === "completed" && j.filename === wsFileCrumb.textContent);
  if (job && job.file) runSingleExtract(job.file);
  else addLog("WARN", "No file available for re-run. Please upload again.");
});

wsExportCsvBtn.addEventListener("click", () => {
  if (!state.workspace || !state.workspace.parsed_json) return;
  const rows = flattenToRows(state.workspace.parsed_json);
  const csv  = rows.map(r => r.map(v => `"${String(v).replace(/"/g, '""')}"`).join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = `${wsFileCrumb.textContent || "export"}.csv`; a.click();
  URL.revokeObjectURL(url);
});

wsCopyJsonBtn.addEventListener("click", () => {
  if (!state.workspace || !state.workspace.parsed_json) return;
  navigator.clipboard.writeText(JSON.stringify(state.workspace.parsed_json, null, 2))
    .then(() => { wsCopyJsonBtn.textContent = "&#10003; Copied!"; setTimeout(() => { wsCopyJsonBtn.innerHTML = "&#128203; Copy JSON"; }, 2000); });
});

wsZoomIn.addEventListener("click",    () => adjustWsZoom(0.2));
wsZoomOut.addEventListener("click",   () => adjustWsZoom(-0.2));
wsZoomReset.addEventListener("click", () => { wsZoom = 1; applyWsZoom(); });
wsFullscreen.addEventListener("click", () => { if (wsGallery.length) openLightbox(wsGallery[0]); });

function adjustWsZoom(delta) {
  wsZoom = Math.max(0.4, Math.min(3, wsZoom + delta));
  applyWsZoom();
  wsZoomReset.textContent = `${Math.round(wsZoom * 100)}%`;
}
function applyWsZoom() {
  wsPreviewBody.querySelectorAll(".preview-stage img").forEach(img => {
    img.style.transform = `scale(${wsZoom})`;
    img.style.transformOrigin = "top center";
  });
  wsZoomReset.textContent = `${Math.round(wsZoom * 100)}%`;
}

// ── Dashboard ──────────────────────────────────
const activityEvents = [];

function addActivityEvent(type, title, desc) {
  const now = new Date();
  activityEvents.unshift({ type, title, desc, time: now });
  if (activityEvents.length > 10) activityEvents.pop();
  if (state.page === "dashboard") refreshDashboard();
}

function refreshDashboard() {
  const stats = state.sessionStats;
  dashDocsCount.textContent = stats.total;
  const rate = stats.total > 0 ? `${Math.round((stats.success / stats.total) * 100)}%` : "—";
  dashSuccessEl.textContent = rate;
  const avgConf = stats.conf.length > 0 ? (stats.conf.reduce((a, b) => a + b, 0) / stats.conf.length).toFixed(2) : "—";
  dashConfEl.textContent  = avgConf;
  dashActiveEl.textContent = state.activeJobs;

  // Activity log
  if (activityEvents.length > 0) {
    dashActivityLog.innerHTML = activityEvents.slice(0, 4).map(ev => `
      <div class="activity-item">
        <div class="activity-time">${formatTime(ev.time)}</div>
        <div class="activity-row">
          <div class="activity-icon ${ev.type === "batch" ? "green" : ev.type === "key" ? "purple" : ""}">${ev.type === "batch" ? "&#8679;" : ev.type === "extract" ? "&#9638;" : "&#9881;"}</div>
          <div>
            <div class="activity-content-title">${escHtml(ev.title)}</div>
            <div class="activity-content-desc">${escHtml(ev.desc)}</div>
          </div>
        </div>
      </div>
    `).join("");
  }

  // Projects
  const batches = {};
  state.jobs.forEach(j => {
    if (!batches[j.id.split("-").slice(0, 3).join("-")]) batches[j.id.split("-").slice(0, 3).join("-")] = { jobs: [], batchId: state.batchId };
    batches[j.id.split("-").slice(0, 3).join("-")].jobs.push(j);
  });
  const batchList = Object.values(batches).slice(0, 2);
  if (batchList.length > 0) {
    const existing = dashProjectsList.querySelector(".add-new");
    const projectHTML = batchList.map(b => {
      const total     = b.jobs.length;
      const completed = b.jobs.filter(j => j.status === "completed").length;
      const pct       = total > 0 ? Math.round((completed / total) * 100) : 0;
      const active    = b.jobs.some(j => j.status === "processing" || j.status === "pending");
      return `
        <div class="project-item" data-nav="batch" style="cursor:pointer">
          <div class="project-thumb">&#128196;</div>
          <div class="project-info">
            <div class="project-name">${escHtml(state.batchId || "Batch")}</div>
            <div class="project-desc">${total} document${total !== 1 ? "s" : ""}</div>
            <div class="project-progress"><div class="project-progress-fill" style="width:${pct}%"></div></div>
          </div>
          <span class="badge ${active ? "badge-processing" : "badge-idle"}">${active ? "Processing" : "Idle"}</span>
          <span class="project-pct">${pct}%</span>
        </div>
      `;
    }).join("");
    dashProjectsList.innerHTML = projectHTML + (existing ? existing.outerHTML : "");
    dashProjectsList.querySelectorAll("[data-nav]").forEach(el => el.addEventListener("click", () => navigate(el.dataset.nav)));
  }

  // Node status (simulated)
  nodeCpu.textContent = `${20 + Math.floor(Math.random() * 60)}%`;
  nodeMem.textContent = `${(4 + Math.random() * 12).toFixed(1)}GB / 16GB`;
  nodeLat.textContent = `${80 + Math.floor(Math.random() * 100)}ms`;
}

// ── Orchestration logs ─────────────────────────
function addLog(type, msg) {
  const now  = new Date();
  const time = `[${String(now.getHours()).padStart(2, "0")}:${String(now.getMinutes()).padStart(2, "0")}:${String(now.getSeconds()).padStart(2, "0")}]`;
  state.logs.push({ time, type, msg });
  if (state.logs.length > 50) state.logs.shift();
  const cls  = type === "CORE_ORCHESTRATOR" ? "log-core" : type === "CRITICAL" ? "log-crit" : type === "WARN" ? "log-warn" : "log-info";
  const entry = document.createElement("div");
  entry.className = "log-entry";
  entry.innerHTML = `<span class="log-t">${time}</span><span class="${cls}">${escHtml(type)}:</span><span class="log-msg">${escHtml(msg)}</span>`;
  orchLogsBody.appendChild(entry);
  orchLogsBody.scrollTop = orchLogsBody.scrollHeight;
}

function fakeLogMsg() {
  const msgs = [
    "Allocating secondary worker node to partition 04.",
    "OCR pipeline throughput nominal.",
    "Layout detection confidence above threshold.",
    "Re-queuing pending tasks for edge processing.",
    "Cache warm — model inference latency reduced.",
    "Worker heartbeat OK.",
  ];
  return msgs[Math.floor(Math.random() * msgs.length)];
}

// ── Gallery / Lightbox helpers ─────────────────
function buildAnnotatedItems(items, annotations, ocrBlocks) {
  const pageMap = new Map(items.map((item, i) => [i + 1, { ...item, overlays: [] }]));
  const fieldBboxKeys = new Set();
  (annotations || []).forEach(a => (a.matches || []).forEach(m => { if (m.bbox) fieldBboxKeys.add(m.bbox.join(",") + ":" + m.page); }));
  (ocrBlocks || []).forEach(block => {
    const target = pageMap.get(block.page);
    if (!target || !block.bbox) return;
    const w = target.width || 1; const h = target.height || 1;
    const key = block.bbox.join(",") + ":" + block.page;
    const isField = fieldBboxKeys.has(key);
    target.overlays.push({ bbox: [block.bbox[0]/w, block.bbox[1]/h, block.bbox[2]/w, block.bbox[3]/h], label: isField ? block.text : "", type: isField ? "field" : "ocr" });
  });
  return Array.from(pageMap.values());
}

function renderBoxes(container, overlays) {
  (overlays || []).forEach(ov => {
    const [x1, y1, x2, y2] = ov.bbox;
    const box = document.createElement("div");
    box.className = "bbox-box";
    if (ov.type) box.dataset.type = ov.type;
    box.style.left   = `${x1 * 100}%`;
    box.style.top    = `${y1 * 100}%`;
    box.style.width  = `${(x2 - x1) * 100}%`;
    box.style.height = `${(y2 - y1) * 100}%`;
    if (ov.label) {
      const lbl = document.createElement("div");
      lbl.className = "bbox-label";
      lbl.textContent = ov.label;
      box.appendChild(lbl);
    }
    container.appendChild(box);
  });
}

function openLightbox(item) {
  if (!item.src) return;
  lightboxItem = item;
  lightboxKind = item.kind || "image";
  lightboxSrc  = item.src;
  lightboxZoom = 1;
  lightboxTitle.textContent = item.name || "Preview";
  lightboxEl.classList.remove("hidden");
  lightboxEl.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
  renderLightboxContent();
}

function closeLightbox() {
  lightboxEl.classList.add("hidden");
  lightboxEl.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
  lightboxContent.innerHTML = "";
  lightboxSrc = ""; lightboxKind = null; lightboxItem = null;
}

function renderLightboxContent() {
  lightboxContent.innerHTML = "";
  if (!lightboxSrc || !lightboxItem) return;
  if (lightboxKind === "pdf") {
    const frame = document.createElement("iframe");
    frame.className = "lightbox-frame"; frame.src = lightboxSrc;
    lightboxContent.appendChild(frame);
    [zoomInBtn, zoomOutBtn].forEach(b => b.disabled = true);
    zoomResetBtn.textContent = "PDF";
    return;
  }
  const stage   = document.createElement("div"); stage.className = "lightbox-stage";
  const preview = document.createElement("div"); preview.className = "lightbox-preview";
  const img     = document.createElement("img"); img.className = "lightbox-image";
  img.src = lightboxSrc; img.style.transform = `scale(${lightboxZoom})`;
  preview.appendChild(img);
  renderBoxes(preview, lightboxItem.overlays);
  stage.appendChild(preview);
  lightboxContent.appendChild(stage);
  [zoomInBtn, zoomOutBtn, zoomResetBtn].forEach(b => b.disabled = false);
  zoomResetBtn.textContent = `${Math.round(lightboxZoom * 100)}%`;
}

function adjustLightboxZoom(delta) {
  if (lightboxKind === "pdf") return;
  lightboxZoom = Math.max(0.4, Math.min(4, lightboxZoom + delta));
  renderLightboxContent();
}

lightboxClose.addEventListener("click", closeLightbox);
lightboxBackdrop.addEventListener("click", closeLightbox);
zoomInBtn.addEventListener("click",    () => adjustLightboxZoom(0.25));
zoomOutBtn.addEventListener("click",   () => adjustLightboxZoom(-0.25));
zoomResetBtn.addEventListener("click", () => { lightboxZoom = 1; renderLightboxContent(); });

document.addEventListener("keydown", e => {
  if (lightboxEl.classList.contains("hidden")) return;
  if (e.key === "Escape") closeLightbox();
  else if (e.key === "+" || e.key === "=") adjustLightboxZoom(0.25);
  else if (e.key === "-") adjustLightboxZoom(-0.25);
});

// ── JSON syntax highlight ──────────────────────
function syntaxHighlight(json) {
  return json
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)/g, (m) => {
      if (/^"/.test(m)) {
        if (/:$/.test(m)) return `<span class="jk">${m}</span>`;
        return `<span class="js">${m}</span>`;
      }
      if (/true|false/.test(m)) return `<span class="jb">${m}</span>`;
      if (/null/.test(m))       return `<span class="jb">${m}</span>`;
      return `<span class="jn">${m}</span>`;
    });
}

// ── CSV export helper ──────────────────────────
function flattenToRows(obj, prefix) {
  const rows = [["field", "value"]];
  function walk(o, p) {
    if (o === null || typeof o !== "object") { rows.push([p, o]); return; }
    if (Array.isArray(o)) { o.forEach((v, i) => walk(v, `${p}[${i}]`)); return; }
    Object.entries(o).forEach(([k, v]) => walk(v, p ? `${p}.${k}` : k));
  }
  walk(obj, prefix || "");
  return rows;
}

// ── Status badge HTML ──────────────────────────
function statusBadgeHtml(status) {
  const map = {
    pending:    ["badge-pending",    "&#9202; Pending"],
    processing: ["badge-extracting", "&#9668;&#9668; Extracting"],
    completed:  ["badge-completed",  "&#10003; Completed"],
    failed:     ["badge-critical",   "&#9888; Critical Failure"],
  };
  const [cls, label] = map[status] || ["badge-idle", status];
  return `<span class="badge ${cls}">${label}</span>`;
}

function complexityBars(job) {
  const colors = job.status === "failed"
    ? ["red", "red", ""]
    : job.status === "completed"
    ? ["blue", "blue", "dark"]
    : ["blue", "", ""];
  return colors.map(c => `<div class="cbar ${c}"></div>`).join("");
}

// ── Utilities ──────────────────────────────────
function escHtml(str) {
  return String(str || "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function formatBytes(bytes) {
  if (!bytes) return "—";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1048576) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1048576).toFixed(1)} MB`;
}

function formatTs(d) {
  if (!d) return "—";
  const dt = new Date(d);
  return `${dt.toLocaleDateString("en-CA")}, ${String(dt.getHours()).padStart(2, "0")}:${String(dt.getMinutes()).padStart(2, "0")}:${String(dt.getSeconds()).padStart(2, "0")}`;
}

function formatTime(d) {
  if (!d) return "—";
  return `${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")} ${d.getHours() < 12 ? "AM" : "PM"}`;
}

function shortHash(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (Math.imul(31, h) + s.charCodeAt(i)) | 0;
  return Math.abs(h).toString(16).padStart(8, "0").slice(0, 4) + "..." + Math.abs(h >> 4).toString(16).padStart(4, "0").slice(0, 4);
}

function fileIcon(name) {
  const ext = (name || "").split(".").pop().toLowerCase();
  if (ext === "pdf") return "&#128196;";
  if (["jpg", "jpeg", "png", "tiff"].includes(ext)) return "&#128247;";
  return "&#128196;";
}

function countLeafValues(node) {
  if (!node) return 0;
  if (typeof node === "string") return node.trim() ? 1 : 0;
  if (Array.isArray(node)) return node.reduce((s, v) => s + countLeafValues(v), 0);
  if (typeof node === "object") return Object.values(node).reduce((s, v) => s + countLeafValues(v), 0);
  return 0;
}

function estimateConfidence(json) {
  if (!json) return 0.87;
  const str = JSON.stringify(json);
  const filled = (str.match(/: "[^"]+"/g) || []).length;
  const nulls  = (str.match(/: null/g) || []).length;
  if (filled + nulls === 0) return 0.87;
  return Math.min(0.99, 0.7 + (filled / (filled + nulls)) * 0.29);
}

// ── Header export ──────────────────────────────
document.getElementById("header-export-btn").addEventListener("click", () => {
  const blob = new Blob([JSON.stringify({ jobs: state.jobs, stats: state.sessionStats }, null, 2)], { type: "application/json" });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement("a");
  a.href = url; a.download = "ocr_session_export.json"; a.click();
  URL.revokeObjectURL(url);
});

document.getElementById("ingest-export-btn")?.addEventListener("click", () => document.getElementById("header-export-btn").click());

// ── Init ───────────────────────────────────────
loadOptions().then(() => {
  addLog("CORE_ORCHESTRATOR", "System initialised. Ready for batch submission.");
  refreshDashboard();
});
