/**
 * ML Inference Module
 *
 * Loads and runs local ONNX toxicity detection model:
 * - Uses ONNX Runtime with WebGL backend
 * - Transformers.js for text classification pipeline
 * - All processing happens locally (no CDN, no external APIs)
 * - Supports batched inference for performance
 */

/**
 * Transformers.js pipeline instance (initialized on first use).
 * @type {object|null}
 */
let pipe = null;

/**
 * Classification threshold (0.0-1.0). Scores above this are considered toxic.
 * @type {number}
 */
let threshold = 0.5;

/**
 * Import a local vendor module using chrome.runtime.getURL.
 *
 * @param {string} p - Relative path to module
 * @returns {Promise<Module>} Imported module
 */
async function importLocal(p) {
  return import(chrome.runtime.getURL(p));
}

/**
 * Check if a URL is accessible via HEAD request.
 *
 * @param {string} url - URL to check
 * @returns {Promise<boolean>} True if response is OK
 */
async function headOk(url) {
  const r = await fetch(url, { method: "HEAD" });
  return r.ok;
}

/**
 * Validate that all required model files are present locally.
 *
 * Checked files:
 * - config.json
 * - tokenizer.json
 * - tokenizer_config.json
 * - onnx/model_quantized.onnx
 *
 * @async
 * @throws {Error} If any required files are missing
 * @returns {Promise<void>}
 */
async function preflightModelFiles() {
  const base = chrome.runtime.getURL("models/toxic-bert/");
  const must = [
    base + "config.json",
    base + "tokenizer.json",
    base + "tokenizer_config.json",
    // special_tokens_map.json is optional - only check if you add it later
    base + "onnx/model_quantized.onnx"
  ];

  const results = await Promise.all(must.map(async (u) => [u, await headOk(u)]));
  const missing = results.filter(([, ok]) => !ok).map(([u]) => u);
  console.log("[preflight] model files:", Object.fromEntries(
    results.map(([u, ok]) => [u.replace(/^.*\/models\//, "models/"), ok])
  ));
  if (missing.length) {
    throw new Error("Missing local model assets:\n" + missing.join("\n"));
  }
}

/**
 * Initialize the ML pipeline (one-time setup, idempotent).
 *
 * Steps:
 * 1. Load ONNX Runtime (WebGL) from vendor
 * 2. Set globalThis.ort to prevent Transformers.js from using CDN
 * 3. Load Transformers.js from vendor
 * 4. Configure environment for local-only models
 * 5. Build text-classification pipeline for toxic-bert
 * 6. Validate all model files are present
 * 7. Load threshold setting from storage
 *
 * @async
 * @returns {Promise<void>}
 */
export async function initIfNeeded() {
  if (pipe) return;

  // 1) Load ONNX Runtime (WebGL ESM) locally and make it GLOBAL
  const ortModule = await importLocal("vendor/ort.webgl.min.mjs");
  const ort = ortModule.default ?? ortModule;
  // IMPORTANT: Set globally so Transformers.js doesn't pull from CDN
  globalThis.ort = ort;

  // 2) Load Transformers.js (local)
  const tjs = await importLocal("vendor/transformers.min.js");
  const env = tjs.env ?? tjs.default?.env ?? tjs;
  const pipeline = tjs.pipeline ?? tjs.default?.pipeline;

  // 3) Configure: local models only, paths, WebGL backend
  env.allowLocalModels = true;
  env.allowRemoteModels = false;
  env.localModelPath = chrome.runtime.getURL("models/");
  env.backends = env.backends || {};
  env.backends.onnx = env.backends.onnx || {};
  env.backends.onnx.backend = "webgl";
  env.useBrowserCache = false;  // Optional: disable HTTP caches
  env.remoteModelBaseUrl = "";  // Optional: hard stop (no fallback)

  // 4) Build pipeline - device stays "wasm" (Transformers.js only knows webgpu/wasm)
  pipe = await pipeline("text-classification", "toxic-bert", {
    quantized: true,
    dtype: "q8",
    device: "wasm"
  });

  // Validate local model files before use (fails fast)
  await preflightModelFiles();

  // Load threshold from storage
  const cfg = await chrome.storage.local.get(["threshold"]);
  if (typeof cfg.threshold === "number") threshold = cfg.threshold;
}

/**
 * Run toxicity detection on a single text.
 *
 * @async
 * @param {string} text - Text to classify
 * @returns {Promise<{toxic: boolean, score: number, labels: Array}>} Classification result
 * @example
 * const result = await runDetector("some text");
 * console.log(result.toxic, result.score, result.labels);
 */
export async function runDetector(text) {
  await initIfNeeded();
  const out = await pipe(text, { topk: null });

  // Single-label binary classification: find the "toxic" label
  const toxicLabel = out.find(o => o.label.toLowerCase() === "toxic" || o.label === "LABEL_1");
  const toxic = toxicLabel ? toxicLabel.score >= threshold : false;
  const score = toxicLabel ? toxicLabel.score : 0;

  return { toxic, score, labels: out };
}

/**
 * Run toxicity detection on multiple texts (optimized batching).
 *
 * This is the primary method used by the content script for performance.
 * Processes multiple texts in a single inference call.
 *
 * @async
 * @param {string[]} texts - Array of texts to classify
 * @returns {Promise<Array<{toxic: boolean, score: number, labels: Array}>>} Array of classification results
 * @example
 * const results = await runDetectorBatch(["text1", "text2", "text3"]);
 * results.forEach(r => console.log(r.toxic, r.score));
 */
export async function runDetectorBatch(texts) {
  await initIfNeeded();
  // Transformers pipeline accepts an array of texts and returns an array of results
  const outs = await pipe(texts, { topk: null });
  // outs: Array<Array<{label, score}>>
  return outs.map(out => {
    // Single-label binary classification: find the "toxic" label
    const toxicLabel = out.find(o => o.label.toLowerCase() === "toxic" || o.label === "LABEL_1");
    const toxic = toxicLabel ? toxicLabel.score >= threshold : false;
    const score = toxicLabel ? toxicLabel.score : 0;

    return { toxic, score, labels: out };
  });
}
