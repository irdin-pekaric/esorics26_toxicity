# Developer Documentation

Comprehensive guide for developers who want to extend, customize, or contribute to Toxic Guardian.

## Table of Contents

- [Development Setup](#development-setup)
- [Architecture Overview](#architecture-overview)
- [Code Organization](#code-organization)
- [Building and Testing](#building-and-testing)
- [Extending Functionality](#extending-functionality)
- [Model Customization](#model-customization)
- [Performance Optimization](#performance-optimization)
- [Debugging](#debugging)
- [Contributing Guidelines](#contributing-guidelines)

---

## Development Setup

### Prerequisites

- Node.js 16+ and npm 7+
- Chrome 88+ or Firefox 89+
- Basic understanding of Chrome Extension APIs
- Familiarity with JavaScript ES modules

### Initial Setup

```bash
# Clone repository
git clone [https://github.com/irdin-pekaric/toxicity_WWW2026/tree/main/extension]
cd toxic-guardian

# Install dependencies
npm install

# Load extension in browser (see README.md)
```

### Development Tools

**Recommended Extensions:**
- Chrome DevTools (built-in)
- Extension Reloader (auto-reload on changes)

**Recommended Editor:**
- VS Code with ESLint and Prettier

---

## Architecture Overview

### Extension Components

Toxic Guardian follows the Manifest V3 architecture with three main components:

```
┌─────────────────┐
│  Popup (UI)     │ ← User interactions
└────────┬────────┘
         │ messages
┌────────▼─────────────────────────────────┐
│  Background Service Worker (Orchestrator) │
└────────┬─────────────────────────────────┘
         │ messages
┌────────▼────────┐
│  Content Script  │ ← Injected into pages
│  - DOM scanning  │
│  - ML inference  │
│  - Content cloak │
└─────────────────┘
```

### Data Flow

1. **Navigation Event** → Background detects → Sends `RUN_SCAN` to content script
2. **Content Script** → Collects text nodes → Batches for inference
3. **ML Module** → Classifies batches → Returns results
4. **Content Script** → Cloaks toxic nodes → Sends progress to background
5. **Background** → Updates badge → Broadcasts to popup
6. **Popup** → Displays progress → Provides navigation controls

---

## Code Organization

### File Structure

```
toxic-guardian/
├── background/
│   └── service_worker.js        # Orchestration, navigation, state
├── content/
│   ├── content.js               # DOM scanning, cloaking, SPA
│   └── highlighter.css          # Styling for toxic wrappers
├── ml/
│   ├── inference.js             # ONNX Runtime, Transformers.js
│   └── preprocess.js            # (Unused legacy - can be removed)
├── models/
│   └── toxic-bert/
│       ├── config.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── onnx/
│           └── model_quantized.onnx
├── popup/
│   ├── popup.html               # UI structure
│   └── popup.js                 # UI logic, message passing
├── vendor/
│   ├── ort.webgl.min.mjs        # ONNX Runtime WebGL
│   └── transformers.min.js      # Transformers.js
├── icons/                       # Extension icons (16-128px)
├── manifest.json                # Extension configuration
└── test_*.html                  # Manual testing pages
```

### Module Responsibilities

| Module | Responsibility |
|--------|----------------|
| `service_worker.js` | Tab state, navigation triggers, message routing |
| `content.js` | DOM traversal, ML inference, content manipulation |
| `inference.js` | Model loading, classification API |
| `popup.js` | User controls, progress display |
| `highlighter.css` | Visual styling for toxic content |

---

## Building and Testing

### No Build Step Required

The extension runs directly from source (no transpilation or bundling needed).

### Manual Testing

1. **Load test page:**
```bash
# Open in browser
open test_toxic_guardian.html
# or
open long_toxicity_test.html
```

2. **Trigger scan:**
   - Click extension icon → "Run"
   - Or run in console: `window.tgScanPage()`

3. **Check console:**
```javascript
// Background service worker console
chrome://extensions → Toxic Guardian → "service worker"

// Content script console
Inspect page → Console tab
```

### Automated Testing

Currently no automated test suite. Contributions welcome!

**Recommended Test Framework:**
- Jest for unit tests
- Puppeteer for integration tests

### Performance Profiling

```javascript
// In content script console
performance.mark('scan-start');
await window.tgScanPage();
performance.mark('scan-end');
performance.measure('scan', 'scan-start', 'scan-end');
console.table(performance.getEntriesByType('measure'));
```

---

## Extending Functionality

### Adding a New Message Type

**1. Define message handler in background:**

```javascript
// background/service_worker.js
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type === "MY_NEW_MESSAGE") {
    // Handle message
    sendResponse({ ok: true, data: "..." });
    return; // or return true for async
  }
});
```

**2. Send message from content script:**

```javascript
// content/content.js
chrome.runtime.sendMessage(
  { type: "MY_NEW_MESSAGE", payload: "..." },
  (response) => {
    console.log(response);
  }
);
```

**3. Send message from popup:**

```javascript
// popup/popup.js
chrome.runtime.sendMessage(
  { type: "MY_NEW_MESSAGE", payload: "..." },
  (response) => {
    console.log(response);
  }
);
```

---

### Adding a Custom Filter

**Example: Skip nodes in `.no-scan` containers**

```javascript
// content/content.js - in walkTextNodes()
if (el.closest(".no-scan")) {
  return NodeFilter.FILTER_REJECT;
}
```

---

### Adding Configuration Options

**1. Define schema:**

```javascript
// Stored in chrome.storage.local
{
  threshold: number,      // 0.0 - 1.0
  autoScan: boolean,      // New option
  sensitivity: string     // "low" | "medium" | "high"
}
```

**2. Add UI controls in popup:**

```html
<!-- popup/popup.html -->
<label>
  <input type="checkbox" id="autoScan" />
  Auto-scan pages
</label>
```

```javascript
// popup/popup.js
const autoScanEl = document.getElementById("autoScan");

// Load setting
chrome.storage.local.get(["autoScan"], (cfg) => {
  autoScanEl.checked = cfg.autoScan ?? true;
});

// Save setting
autoScanEl.addEventListener("change", () => {
  chrome.storage.local.set({ autoScan: autoScanEl.checked });
});
```

**3. Use setting in background:**

```javascript
// background/service_worker.js
async function maybeStartScan(tabId, url) {
  const { autoScan } = await chrome.storage.local.get(["autoScan"]);
  if (!autoScan) return; // Skip if disabled

  // ... existing logic
}
```

---

## Model Customization

### Replacing the Model

**1. Export your model to ONNX:**

```python
# Python example using Hugging Face
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")

# Export to ONNX
ort_model = ORTModelForSequenceClassification.from_pretrained(
    "your-model",
    export=True
)
ort_model.save_pretrained("models/your-model")
```

**2. Quantize the model:**

```python
from optimum.onnxruntime import ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig

quantizer = ORTQuantizer.from_pretrained("models/your-model")
qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
quantizer.quantize(save_dir="models/your-model", quantization_config=qconfig)
```

**3. Update inference.js:**

```javascript
// ml/inference.js
pipe = await pipeline("text-classification", "your-model", {
  quantized: true,
  dtype: "q8",
  device: "wasm"
});
```

**4. Update manifest.json:**

```json
{
  "web_accessible_resources": [{
    "resources": ["vendor/*", "models/your-model/**", "ml/**"],
    "matches": ["<all_urls>"]
  }]
}
```

---

## Performance Optimization

### Current Performance

- **Model load:** ~500-1000ms (one-time)
- **Inference:** ~10-20ms per batch (16 texts)
- **DOM manipulation:** ~1-2ms per node
- **Typical page:** 1-3 seconds for 500 nodes

### Optimization Techniques

#### 1. Increase Batch Size

```javascript
// content/content.js - in scanPage()
const BATCH_SIZE = 32; // Higher = faster but more memory
```

**Trade-offs:**
- Larger batches = faster inference
- But: Higher memory usage, longer freeze if UI blocks

---

#### 2. Reduce Pause Between Batches

```javascript
const MIN_GAP_MS = 0; // Remove pauses (may freeze UI)
```

**Trade-offs:**
- Faster scan completion
- But: UI may become unresponsive during scan

---

#### 3. Early Exit for Short Pages

```javascript
// content/content.js - in scanPage()
if (total < 10) {
  // Skip ML for very short pages
  sendProgress({ runId, state: "done", total, done: total, hits: 0 });
  return;
}
```

---

## Debugging

### Enable Verbose Logging

**Background Service Worker:**

```javascript
// background/service_worker.js
const DEBUG = true;

function log(...args) {
  if (DEBUG) console.log("[TG:BG]", ...args);
}

// Use log() throughout
```

**Content Script:**

```javascript
// content/content.js
const DEBUG = true;

function log(...args) {
  if (DEBUG) console.log("[TG:CS]", ...args);
}
```

---

### Inspect Extension State

**Background state:**

```javascript
// In service worker console
console.table([...tabState.entries()]);
```

**Content script state:**

```javascript
// In page console
console.log({
  runId: TG_SCAN.runId,
  aborted: TG_SCAN.aborted,
  isRunning: TG_SCAN.isRunning,
  toxicNodes: __TOX_NODES__.length
});
```

---

### Debug Model Loading

```javascript
// ml/inference.js - add logging
console.log("[Inference] Loading ONNX Runtime...");
const ortModule = await importLocal("vendor/ort.webgl.min.mjs");
console.log("[Inference] ONNX Runtime loaded:", ortModule);

console.log("[Inference] Loading Transformers.js...");
const tjs = await importLocal("vendor/transformers.min.js");
console.log("[Inference] Transformers.js loaded:", tjs);

console.log("[Inference] Building pipeline...");
pipe = await pipeline("text-classification", "toxic-bert", { ... });
console.log("[Inference] Pipeline ready:", pipe);
```

---

## Contributing Guidelines

### Code Style

- **Indentation:** 2 spaces
- **Quotes:** Double quotes for strings
- **Semicolons:** Yes
- **Line length:** Max 120 characters

### Commit Messages

Follow Conventional Commits:



### Pull Request Process

1. **Fork** the repository
2. **Create branch** from `main`: `git checkout -b feature/your-feature`
3. **Make changes** following code style
4. **Test thoroughly** (manual testing required)
5. **Update documentation** (README, API, DEVELOPER as needed)
6. **Commit** with clear messages
7. **Push** to your fork
8. **Open PR** with description of changes

## Resources

### Documentation

- [Chrome Extension API Reference](https://developer.chrome.com/docs/extensions/reference/)
- [Manifest V3 Migration Guide](https://developer.chrome.com/docs/extensions/mv3/intro/)
- [Transformers.js Documentation](https://huggingface.co/docs/transformers.js/)
- [ONNX Runtime Documentation](https://onnxruntime.ai/docs/)

### Tools

- [Chrome Extension Manifest Validator](https://developer.chrome.com/docs/extensions/mv3/manifest/)
- [ONNX Model Zoo](https://github.com/onnx/models)
- [Hugging Face Model Hub](https://huggingface.co/models)

