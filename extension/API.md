# API Documentation

Technical documentation for Toxic Guardian's internal APIs and module interfaces.

## Table of Contents

- [Background Service Worker](#background-service-worker)
- [Content Script](#content-script)
- [ML Inference Module](#ml-inference-module)
- [Messaging Protocol](#messaging-protocol)
- [Storage Schema](#storage-schema)
- [CSS Classes](#css-classes)

---

## Background Service Worker

**File**: `background/service_worker.js`

### Purpose

Orchestrates scan lifecycle across tabs, manages navigation triggers, and maintains per-tab state.

### State Management

#### `tabState: Map<number, TabState>`

Per-tab state storage.

```typescript
interface TabState {
  url: string;           // Current URL
  inProgress: boolean;   // Whether scan is running
  lastRunId: number;     // Sequential run ID
  lastScanAt: number;    // Timestamp of last scan (ms)
  last: ScanProgress;    // Last progress message
}
```

### Functions

#### `canScan(url: string): boolean`

Determines if a URL is scannable.

**Parameters:**
- `url` - URL to check

**Returns:** `true` if URL starts with `http:`, `https:`, or `file:`

**Example:**
```javascript
canScan("https://example.com"); // true
canScan("chrome://extensions"); // false
```

---

#### `sendToTab(tabId: number, msg: object): Promise<void>`

Sends a message to a content script safely (catches errors).

**Parameters:**
- `tabId` - Tab ID to send to
- `msg` - Message object

**Returns:** Promise that resolves when sent (or fails silently)

---

#### `maybeStartScan(tabId: number, url: string): void`

Starts a scan if conditions are met.

**Conditions:**
- URL is scannable
- No scan currently in progress
- Either URL changed or 2+ seconds passed since last scan

**Parameters:**
- `tabId` - Tab to scan
- `url` - Current URL

**Side Effects:**
- Updates `tabState`
- Sends `RUN_SCAN` message to content script

---

## Message Handlers

### Background ↔ Content Script

| Direction | Type | Purpose |
|-----------|------|---------|
| CS → BG | `CS_HELLO` | Get tab ID |
| CS → BG | `SCAN_PROGRESS` | Send scan progress |
| BG → CS | `RUN_SCAN` | Start scan |
| BG → CS | `CANCEL_SCAN` | Abort scan |

### Storage Schema

#### `chrome.storage.local`

**threshold**: Classification threshold (0.0 - 1.0, default: 0.5)

**Usage:**
```javascript
// Get
const { threshold } = await chrome.storage.local.get(["threshold"]);

// Set
await chrome.storage.local.set({ threshold: 0.7 });
```

---

For complete API documentation, see the full version in the repository.
