/**
 * Content Script - Toxicity Scanner
 *
 * Scans page DOM for toxic content using local ML inference:
 * - Triggered by background service worker or manual actions
 * - Walks DOM to collect visible text nodes
 * - Runs batched ML inference for classification
 * - Cloaks toxic content behind reveal banners
 * - Sends real-time progress updates to background
 *
 * All logic runs locally. Requires ml/inference.js in web_accessible_resources.
 */

// ---------------------------------------------------------------
// 0) Small utilities (DOM traversal, ML loader, cloaking UI)
// ---------------------------------------------------------------

/**
 * Walk visible text nodes in the document body.
 *
 * Filters:
 * - Skips nodes inside processed wrappers (data-tg-processed)
 * - Skips hidden/invisible elements
 * - Skips non-content elements (script/style/iframe/svg etc.)
 *
 * @generator
 * @param {Node} [root=document.body] - Starting node for traversal
 * @yields {Text} Visible text nodes
 */
function* walkTextNodes(root = document.body) {
  const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) => {
      if (!node?.nodeValue || !node.nodeValue.trim()) return NodeFilter.FILTER_REJECT;

      const el = node.parentElement;
      if (!el) return NodeFilter.FILTER_REJECT;
      if (el.closest("[data-tg-processed='1']")) return NodeFilter.FILTER_REJECT;

      // Skip hidden/invisible nodes
      const st = getComputedStyle(el);
      if (!st || st.visibility === "hidden" || st.display === "none") return NodeFilter.FILTER_REJECT;

      // Skip non-content containers
      const tag = el.tagName?.toUpperCase();
      if (["SCRIPT", "STYLE", "NOSCRIPT", "IFRAME", "CANVAS", "SVG"].includes(tag)) {
        return NodeFilter.FILTER_REJECT;
      }

      // Optional: skip editable inputs/areas (uncomment if needed)
      // if (el.closest("input, textarea, [contenteditable], [role='textbox']")) return NodeFilter.FILTER_REJECT;

      return NodeFilter.FILTER_ACCEPT;
    }
  });

  let n;
  while ((n = walker.nextNode())) yield n;
}

/**
 * Dynamically import the inference module.
 *
 * @returns {Promise<Module>} Inference module with runDetectorBatch and initIfNeeded
 */
async function getInference() {
  const url = chrome.runtime.getURL("ml/inference.js");
  return import(url);
}

/**
 * Create an accessible "reveal" banner for hidden toxic content.
 *
 * Features:
 * - Keyboard accessible (Tab + Enter/Space)
 * - ARIA labels with confidence scores
 * - Prevents event bubbling to parent links
 *
 * @param {Function} onReveal - Callback when user reveals content
 * @param {object} [debugInfo] - Classification results with labels and scores
 * @returns {HTMLElement} Banner element
 */
function createBanner(onReveal, debugInfo) {
  const banner = document.createElement("span");
  banner.className = "toxic-banner";
  banner.setAttribute("role", "button");
  banner.setAttribute("tabindex", "0");
  banner.textContent = "toxic content — click to reveal";

  if (debugInfo?.labels?.length) {
    const top = [...debugInfo.labels].sort((a, b) => b.score - a.score)[0];
    const t = `${top.label}: ${(top.score * 100).toFixed(1)}%`;
    banner.title = t;
    banner.setAttribute("aria-label", t);
  }

  // Unified reveal action
  const trigger = () => {
    onReveal();
    banner.remove();
  };

  // Prevent event bubbling to parent elements (especially links)
  const eat = (e) => {
    e.preventDefault();
    e.stopPropagation();
    e.stopImmediatePropagation();
  };
  ["pointerdown", "pointerup", "mousedown", "mouseup", "click", "touchstart", "touchend"].forEach(type => {
    banner.addEventListener(type, (e) => { eat(e); if (type === "click") trigger(); }, { capture: true });
  });

  // Keyboard: Enter/Space → reveal without scrolling/navigation
  banner.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      trigger();
    }
  });

  return banner;
}

/**
 * Replace a text node with a wrapper that hides content behind a reveal banner.
 *
 * Link Handling:
 * - If node is inside an <a>, disable link clicks until content is revealed
 * - Tracks banner count per link (multiple toxic texts in one link)
 * - Re-enables link only when all banners in that link are revealed
 *
 * @param {Text} textNode - Text node to cloak
 * @param {object} res - Classification result with labels and scores
 */
function cloakWholeNode(textNode, res) {
  const wrapper = document.createElement("span");
  wrapper.className = "toxic-wrapper";
  wrapper.dataset.tgProcessed = "1";

  const hidden = document.createElement("span");
  hidden.className = "toxic-hidden";
  hidden.textContent = textNode.nodeValue;

  // If inside <a>: disable link clicks while at least one banner is visible
  const anchor = textNode.parentElement?.closest?.("a");
  if (anchor) {
    // Track multiple banners in the same link
    const count = (parseInt(anchor.dataset.tgBlockCount || "0", 10) || 0) + 1;
    anchor.dataset.tgBlockCount = String(count);
    if (count === 1) {
      // Save original state and prevent clicks
      anchor.dataset.tgPrevPE = anchor.style.pointerEvents || "";
      anchor.style.pointerEvents = "none";
    }
  }

  const banner = createBanner(() => {
    // Reveal content
    hidden.classList.add("toxic-revealed");
    const badge = document.createElement("span");
    badge.className = "toxic-badge";
    badge.textContent = "revealed";
    wrapper.appendChild(badge);

    // Re-enable link if no other banners are left in the same <a>
    if (anchor) {
      const left = Math.max(0, (parseInt(anchor.dataset.tgBlockCount || "1", 10) || 1) - 1);
      anchor.dataset.tgBlockCount = String(left);
      if (left === 0) {
        anchor.style.pointerEvents = anchor.dataset.tgPrevPE || "";
        delete anchor.dataset.tgPrevPE;
      }
    }
  }, res);

  wrapper.appendChild(banner);
  wrapper.appendChild(hidden);
  textNode.parentNode.replaceChild(wrapper, textNode);
}

// --- Cancel scan on page unload (navigation away, hard reload) ---
window.addEventListener("beforeunload", () => {
  if (TG_SCAN.isRunning) TG_SCAN.aborted = true;
});

// ---------------------------------------------------------------
// 1) Global scan state + background/popup handshake
// ---------------------------------------------------------------

/**
 * Scan lifecycle state.
 * @type {{runId: number, aborted: boolean, isRunning: boolean}}
 */
let TG_SCAN = {
  runId: 0,
  aborted: false,
  isRunning: false
};

/**
 * Tab ID obtained from background handshake.
 * @type {number|null}
 */
let __TG_TAB_ID__ = null;

// Identify our tabId to the Service Worker (used in progress updates)
chrome.runtime.sendMessage({ type: "CS_HELLO" }, (res) => {
  if (res && typeof res.tabId === "number") __TG_TAB_ID__ = res.tabId;
});

/**
 * Send progress update to background for badge/popup updates.
 *
 * @param {object} p - Progress object with state, total, done, hits, etc.
 */
function sendProgress(p) {
  chrome.runtime.sendMessage({ type: "SCAN_PROGRESS", tabId: __TG_TAB_ID__, ...p }).catch(() => { });
}

/**
 * Remove any previous cloaked wrappers from the DOM (idempotent).
 */
function clearPreviousScan() {
  document.querySelectorAll(".toxic-wrapper").forEach((el) => el.replaceWith(...el.childNodes));
}

// ---------------------------------------------------------------
// 2) Main scan (triggered via RUN_SCAN message)
//
// Algorithm:
//  - Traverses DOM text nodes
//  - Batches them for ML inference (significant speed-up)
//  - Cloaks nodes classified as toxic
//  - Streams real-time progress back to background/popup
// ---------------------------------------------------------------

/**
 * Main page scanning function.
 *
 * @async
 * @returns {Promise<void>}
 */
async function scanPage() {
  if (TG_SCAN.isRunning) return;
  TG_SCAN.isRunning = true;

  TG_SCAN.runId++;
  TG_SCAN.aborted = false;
  const runId = TG_SCAN.runId;

  clearPreviousScan();

  // Collect all eligible text nodes (no pre-filter to avoid missing content).
  const nodes = [...(function* () { for (const n of walkTextNodes()) yield n; })()];
  const queue = nodes.slice();

  const total = queue.length;
  let done = 0;
  let hits = 0;

  sendProgress({ runId, state: "start", total, done, hits });

  if (total === 0) {
    sendProgress({ runId, state: "done", total: 0, done: 0, hits: 0 });
    TG_SCAN.isRunning = false;
    return;
  }

  try {
    // Load and warm up the local ML pipeline
    const t0 = performance.now();
    const { runDetectorBatch, initIfNeeded } = await getInference();
    await initIfNeeded();
    sendProgress({
      runId,
      state: "running",
      total,
      done,
      hits,
      note: "ml_ready",
      t_ml: (performance.now() - t0) | 0
    });

    // ---------- Batched inference parameters ----------
    const BATCH_SIZE = 16;    // 8–32 tends to be a good sweet spot
    const MIN_GAP_MS = 8;     // small pause to keep UI responsive
    const MAX_CHARS = 1024;   // clip very long nodes
    const MIN_LEN = 6;        // skip ultra-short fragments

    await new Promise((resolve) => {
      const pump = async () => {
        if (TG_SCAN.aborted || TG_SCAN.runId !== runId) return resolve();

        // 1) Fill a bucket with up to BATCH_SIZE eligible nodes
        const bucket = [];
        while (bucket.length < BATCH_SIZE && queue.length) {
          const node = queue.shift();

          if (!node?.isConnected) {
            done++;
            continue;
          }

          const raw = node.nodeValue ?? "";
          const text = raw.trim();
          if (text.length < MIN_LEN || !/\S+\s+\S+/.test(text)) {
            // too short / single token → skip
            done++;
            continue;
          }

          const clipped = text.length > MAX_CHARS ? text.slice(0, MAX_CHARS) : text;
          bucket.push({ node, text: clipped });
        }

        // 2) If bucket is empty, but queue still has nodes, it was pure skips → wait a tick
        if (bucket.length === 0) {
          if (queue.length === 0) return resolve();
          setTimeout(pump, MIN_GAP_MS);
          return;
        }

        try {
          // 3) Run batched inference
          const results = await runDetectorBatch(bucket.map((b) => b.text));

          // If user navigated away or a new run started while we were inferring, skip applying.
          if (TG_SCAN.aborted || TG_SCAN.runId !== runId) {
            done += bucket.length;
            sendProgress({ runId, state: done < total ? "running" : "finishing", total, done, hits });
            setTimeout(pump, MIN_GAP_MS);
            return; // exit this pump iteration early
          }

          // 4) Apply results to DOM
          for (let i = 0; i < bucket.length; i++) {
            const { node } = bucket[i];
            const res = results[i];
            if (res?.toxic && node?.isConnected) {
              cloakWholeNode(node, res);
              hits++;
            }
          }

        } catch {
          // ignore batch errors; still advance progress
        } finally {
          // 5) Update progress counters
          done += bucket.length;
          sendProgress({
            runId,
            state: done < total ? "running" : "finishing",
            total,
            done,
            hits
          });
        }

        // 6) Continue pumping until finished
        if (queue.length === 0 && done >= total) return resolve();
        setTimeout(pump, MIN_GAP_MS);
      };

      pump();
    });

    // Build the toxic list once at the end so popup navigation works immediately
    try { refreshToxicNodes(); } catch {}

    // Final state report
    sendProgress({ runId, state: TG_SCAN.aborted ? "aborted" : "done", total, done, hits });
  } catch (e) {
    console.error("Scan failed:", e);
    sendProgress({
      runId,
      state: "error",
      error: e?.message || String(e),
      total,
      done,
      hits
    });
  } finally {
    TG_SCAN.isRunning = false;
  }
}

// Expose a manual trigger for quick testing in the page console.
window.tgScanPage = scanPage;

// ---------------------------------------------------------------
// SPA route detection: re-scan on pushState/replaceState/popstate/hashchange
// ---------------------------------------------------------------
/*
(function setupSpaRescan() {
  let rescanTimer = null;
  const RESCAN_DELAY = 300;

  function scheduleRescan(reason = "spa") {
    // abort current run, then (debounced) start a new scan
    if (TG_SCAN.isRunning) TG_SCAN.aborted = true;
    clearTimeout(rescanTimer);
    rescanTimer = setTimeout(() => { scanPage().catch(console.error); }, RESCAN_DELAY);
  }

  // patch history methods
  const _pushState = history.pushState;
  const _replaceState = history.replaceState;
  history.pushState = function (...args) { const r = _pushState.apply(this, args); window.dispatchEvent(new Event("tg:navigate")); return r; };
  history.replaceState = function (...args) { const r = _replaceState.apply(this, args); window.dispatchEvent(new Event("tg:navigate")); return r; };

  window.addEventListener("tg:navigate", () => scheduleRescan("history"));
  window.addEventListener("popstate", () => scheduleRescan("popstate"));
  window.addEventListener("hashchange", () => scheduleRescan("hashchange"));
})();
*/

// ---------------------------------------------------------------
// 2.5  Navigation between toxic wrappers (for popup/control)
// ---------------------------------------------------------------

let __TOX_NODES__ = [];
let __TOX_IDX__ = -1;

/** Rebuild the list of toxic wrappers only (filter detached, stamp stable index). */
function refreshToxicNodes() {
  __TOX_NODES__ = Array.from(document.querySelectorAll(".toxic-wrapper"))
    .filter(n => n && n.isConnected);

  __TOX_NODES__.forEach((n, i) => { n.dataset.tgIndex = String(i); });

  __TOX_IDX__ = __TOX_NODES__.length ? Math.min(__TOX_IDX__, __TOX_NODES__.length - 1) : -1;
  if (__TOX_IDX__ < 0 && __TOX_NODES__.length) __TOX_IDX__ = 0;
  return __TOX_NODES__.length;
}

/** Ensure the internal list exists; compact stale entries; returns current length. */
function ensureToxicList() {
  if (!__TOX_NODES__ || __TOX_NODES__.length === 0) {
    return refreshToxicNodes();
  }
  const before = __TOX_NODES__.length;
  __TOX_NODES__ = __TOX_NODES__.filter(n => n && n.isConnected);
  if (__TOX_NODES__.length !== before) {
    __TOX_NODES__.forEach((n, i) => { n.dataset.tgIndex = String(i); });
    if (__TOX_NODES__.length === 0) __TOX_IDX__ = -1;
    else __TOX_IDX__ = Math.max(0, Math.min(__TOX_IDX__, __TOX_NODES__.length - 1));
  }
  return __TOX_NODES__.length;
}

/** Strong visual emphasis + scroll correction for sticky headers. */
function emphasizeWrapper(wrapper) {
  // remove any previous highlights
  document.querySelectorAll(".tg-focus-wrap").forEach(el => el.classList.remove("tg-focus-wrap"));
  document.querySelectorAll(".tg-pulse").forEach(el => el.classList.remove("tg-pulse"));

  wrapper.classList.add("tg-focus-wrap");

  const banner = wrapper.querySelector(".toxic-banner");
  const hidden = wrapper.querySelector(".toxic-hidden");
  if (banner && hidden && !hidden.classList.contains("toxic-revealed")) {
    banner.classList.add("tg-pulse");
    setTimeout(() => banner.classList.remove("tg-pulse"), 1200);
  }

  // auto-remove focus ring after a short while (keeps DOM clean)
  setTimeout(() => wrapper.classList.remove("tg-focus-wrap"), 1500);
}

/** Robust scroll into view (handles sticky headers & nested containers). */
function scrollWrapperIntoView(wrapper) {
  // Try native scrolling first
  wrapper.scrollIntoView({ block: "center", inline: "nearest", behavior: "smooth" });

  // Then compensate sticky headers (common on Reddit)
  // Read CSS var if page defines --tg-sticky-offset, else use fallback
  const stickyOffset =
    parseInt(getComputedStyle(document.documentElement).getPropertyValue('--tg-sticky-offset') || '0', 10) || 64;

  // After the smooth scroll, nudge by sticky offset
  requestAnimationFrame(() => {
    const rect = wrapper.getBoundingClientRect();
    if (rect.top < stickyOffset + 10) {
      window.scrollBy({ top: rect.top - stickyOffset - 10, behavior: "auto" });
    }
  });
}

/** Scroll to a specific toxic element index and emphasize it. */
function gotoToxic(index = 0) {
  if (!ensureToxicList()) return;

  __TOX_IDX__ = Math.max(0, Math.min(index, __TOX_NODES__.length - 1));
  const wrapper = __TOX_NODES__[__TOX_IDX__];

  if (!wrapper || !wrapper.isConnected) {
    refreshToxicNodes();
    if (!__TOX_NODES__.length) return;
  }

  // If element is not laid out yet (e.g., virtualized list), retry shortly
  if (wrapper.offsetParent === null) {
    setTimeout(() => gotoToxic(__TOX_IDX__), 80);
    return;
  }

  scrollWrapperIntoView(wrapper);
  emphasizeWrapper(wrapper);
}

/** Cycle forward/backward through matches. */
function nextToxic() {
  if (!ensureToxicList()) return;
  __TOX_IDX__ = (__TOX_IDX__ + 1) % __TOX_NODES__.length;
  gotoToxic(__TOX_IDX__);
}
function prevToxic() {
  if (!ensureToxicList()) return;
  __TOX_IDX__ = (__TOX_IDX__ - 1 + __TOX_NODES__.length) % __TOX_NODES__.length;
  gotoToxic(__TOX_IDX__);
}

// ---------------------------------------------------------------
// 3) Message listener (background/popup control)
// ---------------------------------------------------------------

chrome.runtime.onMessage.addListener((msg) => {
  if (msg?.type === "CANCEL_SCAN" && msg.runId === TG_SCAN.runId) {
    TG_SCAN.aborted = true;
  }
  if (msg?.type === "RUN_SCAN" && !TG_SCAN.isRunning) {
    scanPage().catch(console.error);
  }

  // Navigation commands from popup / action icon
  if (msg?.type === "ENSURE_TOXIC_LIST") {
    ensureToxicList();
    return;
  }
  if (msg?.type === "GOTO_TOXIC") {
    gotoToxic(msg.index ?? 0);
    return;
  }
  if (msg?.type === "NEXT_TOXIC") {
    nextToxic();
    return;
  }
  if (msg?.type === "PREV_TOXIC") {
    prevToxic();
    return;
  }
});

// ---------------------------------------------------------------
// 4) No auto-scan / no mutation-rescan (by design)
// ---------------------------------------------------------------
