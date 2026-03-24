# Toxic Guardian
A browser extension that scans web pages for toxic and harassing content using local machine learning. All processing happens offline in your browser - no cloud APIs, no data collection, complete privacy.
## Features
- **100% Local Processing**: All toxicity detection runs locally using ONNX Runtime and Transformers.js
- **Real-time Scanning**: Automatically scans pages on load and navigation
- **Content Cloaking**: Hides toxic content behind clickable reveal banners
- **Navigation Controls**: Jump between detected toxic content with prev/next buttons
- **SPA Support**: Detects and rescans content on single-page application route changes
- **Zero Network Calls**: No external API dependencies, all models are bundled
- **Privacy First**: No data leaves your browser, no tracking, no telemetry
- **Accessible**: Keyboard navigation and ARIA labels for screen readers
## Installation
### From Source
1. Clone the repository:
```bash
git clone
cd toxic-guardian
```
2. Install dependencies:
```bash
npm install
```
3. Load the extension in your browser:
**Chrome/Edge:**
- Open `chrome://extensions/`
- Enable "Developer mode"
- Click "Load unpacked"
- Select the `toxic-guardian` directory
**Firefox:**
- Open `about:debugging#/runtime/this-firefox`
- Click "Load Temporary Add-on"
- Select the `manifest.json` file
## Usage
### Automatic Scanning
Once installed, Toxic Guardian automatically scans pages when you navigate to them. The extension icon badge shows the number of toxic content matches found.
### Manual Controls
Click the extension icon to open the popup with controls:
- **Run**: Manually trigger a scan of the current page
- **Cancel**: Abort an in-progress scan
- **Prev/Next**: Navigate between detected toxic content
### Revealing Content
When toxic content is detected, it's blurred and hidden behind a yellow banner. Click the banner to reveal the content. The banner shows the confidence score on hover.
### Context Menu
Right-click anywhere on a page and select "Scan page for toxic content" to manually trigger a scan.
## Architecture
### Extension Structure
```
toxic-guardian/
├── background/          # Background service worker
│   └── service_worker.js
├── content/            # Content scripts (injected into pages)
│   ├── content.js
│   └── highlighter.css
├── ml/                 # Machine learning inference
│   ├── inference.js
│   └── preprocess.js
├── models/            # Local ONNX models
│   └── toxic-bert/
├── popup/             # Extension popup UI
│   ├── popup.html
│   └── popup.js
├── vendor/            # Bundled dependencies
│   ├── ort.webgl.min.mjs
│   └── transformers.min.js
└── manifest.json      # Extension configuration
```
### How It Works
1. **Page Load**: Background service worker detects navigation and triggers a scan
2. **Text Collection**: Content script walks the DOM and collects visible text nodes
3. **Batched Inference**: Text is sent in batches to the local ML model (toxic-bert)
4. **Classification**: Each text fragment is classified for toxicity using 6 labels
5. **DOM Manipulation**: Toxic content is wrapped and hidden with reveal banners
6. **Progress Updates**: Real-time progress is sent to the popup and badge
### ML Model
The extension uses a quantized BERT model fine-tuned for toxicity detection:
- **Model**: toxic-bert (ONNX format)
- **Backend**: ONNX Runtime WebGL
- **Labels**: 6 toxicity categories (harassment, hate speech, etc.)
- **Threshold**: Configurable (default: 0.5)
- **Performance**: Batch processing (16 texts at a time) for optimal speed
### Key Technologies
- **Manifest V3**: Modern Chrome extension API
- **ONNX Runtime**: Fast inference engine (WebGL backend)
- **Transformers.js**: Hugging Face transformers in JavaScript
- **TreeWalker**: Efficient DOM text traversal
- **Web Accessible Resources**: Secure module loading
## Performance
- **Typical Page**: 1-3 seconds for ~500 text nodes
- **Model Loading**: 500-1000ms (one-time per session)
- **Batch Size**: 16 texts per inference call
- **Memory**: ~150-200MB for model and runtime
- **UI Responsiveness**: 8ms gaps between batches to prevent freezing
## Privacy & Security
- **No Network Calls**: All processing is local, no external APIs
- **No Data Collection**: Extension doesn't collect, store, or transmit any data
- **No Tracking**: No analytics, telemetry, or user behavior monitoring
- **Local Storage**: Only stores threshold preference
- **Secure Contexts**: Uses Content Security Policy and web accessible resources
## Limitations
- Only works on standard web pages (http/https/file protocols)
- Requires JavaScript to be enabled
- Model is English-only (no multilingual support yet)
- Cannot scan content inside iframes from different origins
- May miss dynamically loaded content (no mutation observer by design)
## Browser Compatibility
- **Chrome**: 88+
- **Edge**: 88+
- **Firefox**: 89+ (experimental)
- **Opera**: 74+
Requires support for:
- Manifest V3
- ES modules
- WebAssembly
- WebGL (for ONNX Runtime backend)
## Configuration
Threshold can be adjusted via Chrome storage:
```javascript
chrome.storage.local.set({ threshold: 0.7 }); // Higher = less sensitive
```
Default threshold is 0.5 (balanced sensitivity).
## Documentation
- **[User Guide](USER-GUIDE.md)**: Complete guide for end-users
- **[API Documentation](API.md)**: Technical API reference
- **[Developer Documentation](DEVELOPER.md)**: Guide for contributors and developers
## Development
See [DEVELOPER.md](./DEVELOPER.md) for detailed development documentation.
## Contributing
Contributions are welcome! Please read the developer documentation before submitting PRs.
### Areas for Improvement
- Multilingual model support
- User-configurable sensitivity settings in popup
- Option to auto-reveal/hide all matches
- Statistics and analytics dashboard
- Custom blocklists/allowlists
- Model updates and version management
## License
ISC
## Credits
- Model: toxic-bert from Hugging Face
- ONNX Runtime: Microsoft
- Transformers.js: Xenova

