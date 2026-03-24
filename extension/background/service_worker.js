const log = (...a) => console.debug("[TG BG]", ...a);

/**
 * Background Service Worker (Manifest V3)
 *
 * Orchestrates toxicity scanning across tabs:
 * - Manages per-tab scan state
 * - Triggers scans on navigation events
 * - Handles message routing between popup and content scripts
 * - Updates extension badge with detection counts
 */

// --- Abort previous scan when navigation starts (top frame) ---
chrome.webNavigation.onBeforeNavigate.addListener(({ tabId, url, frameId }) => {
  if (frameId !== 0) return; // only top-level
  const st = tabState.get(tabId);
  if (st?.inProgress) {
    sendToTab(tabId, { type: "CANCEL_SCAN", runId: st.lastRunId });
    st.inProgress = false;
    tabState.set(tabId, st);
  }
  // clear badge early
  chrome.action.setBadgeText({ tabId, text: "" });
});

// --- Re-scan after navigation completes or SPA route updates ---
/*
chrome.webNavigation.onCompleted.addListener(({ tabId, url, frameId }) => {
  if (frameId !== 0) return;
  if (canScan(url)) maybeStartScan(tabId, url);
});
chrome.webNavigation.onHistoryStateUpdated.addListener(({ tabId, url, frameId }) => {
  if (frameId !== 0) return;
  if (canScan(url)) maybeStartScan(tabId, url);
});
*/

// --- Also scan when user activates a different tab (current page must be scanned) ---
/*
chrome.tabs.onActivated.addListener(({ tabId }) => {
  chrome.tabs.get(tabId, (tab) => {
    if (tab?.id && canScan(tab.url)) maybeStartScan(tab.id, tab.url);
  });
});
*/

// --- Optional: when a tab starts loading via tabs API, cancel the old scan ---
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === "loading") {
    const st = tabState.get(tabId);
    if (st?.inProgress) {
      sendToTab(tabId, { type: "CANCEL_SCAN", runId: st.lastRunId });
      st.inProgress = false;
      tabState.set(tabId, st);
      chrome.action.setBadgeText({ tabId, text: "" });
    }
  }
});

// --- State -------------------------------------------------------

/**
 * Per-tab state storage.
 * @type {Map<number, {url: string, inProgress: boolean, lastRunId: number, lastScanAt: number, last: object}>}
 */
const tabState = new Map();

// --- Utils -------------------------------------------------------

/**
 * Checks if a URL can be scanned.
 * @param {string} url - URL to check
 * @returns {boolean} True if URL starts with http:, https:, or file:
 */
function canScan(url = "") {
  return /^(https?:|file:)/i.test(url);
}

/**
 * Sends a message to a tab's content script safely.
 * @param {number} tabId - Tab ID to send to
 * @param {object} msg - Message object
 * @returns {Promise<void>} Promise that resolves when sent (fails silently)
 */
async function sendToTab(tabId, msg) {
  try {
    await chrome.tabs.sendMessage(tabId, msg);
  } catch (error) {
    // Silently ignore errors (tab may have closed or navigated)
  }
}

/**
 * Starts a scan if conditions are met.
 *
 * Conditions:
 * - URL is scannable
 * - No scan currently in progress
 * - Either URL changed or 2+ seconds passed since last scan
 *
 * @param {number} tabId - Tab ID to scan
 * @param {string} url - Current URL
 */
function maybeStartScan(tabId, url) {
  log("maybeStartScan request", { tabId, url, from: (new Error()).stack.split("\n")[2]?.trim() });
  if (!canScan(url)) return;

  const st = tabState.get(tabId) || {};
  if (st.inProgress) return;

  // Debounce: don't re-scan same URL within 2 seconds
  if (st.url === url && st.lastScanAt && (Date.now() - st.lastScanAt) < 2000) return;

  st.url = url;
  st.inProgress = true;
  st.lastRunId = (st.lastRunId || 0) + 1;
  tabState.set(tabId, st);

  sendToTab(tabId, { type: "RUN_SCAN" })
  ;
}

// --- Context menu (optional) ------------------------------------

/**
 * Creates context menu item for manual scanning.
 */
chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: "scanPageToxicity", title: "Scan page for toxic content", contexts: ["page"] });
});
chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === "scanPageToxicity" && tab?.id && canScan(tab.url)) maybeStartScan(tab.id, tab.url);
});

// --- Navigation triggers (Background controls auto-start) ----------
/*
chrome.webNavigation.onCompleted.addListener(({ tabId, url }) => { if (canScan(url)) maybeStartScan(tabId, url); });
chrome.webNavigation.onHistoryStateUpdated.addListener(({ tabId, url }) => { if (canScan(url)) maybeStartScan(tabId, url); });
*/
chrome.tabs.onRemoved.addListener((tabId) => { tabState.delete(tabId); });

// --- Messaging ---------------------------------------------------

/**
 * Message handler for communication between popup, content scripts, and background.
 *
 * Handled message types:
 * - CS_HELLO: Content script handshake to get tab ID
 * - SCAN_PROGRESS: Progress updates from content script
 * - RUN_SCAN_ACTIVE_TAB: Request to scan active tab (from popup)
 * - GET_STATUS_FOR_ACTIVE_TAB: Request for active tab status (from popup)
 * - CANCEL_ACTIVE_SCAN: Request to cancel active scan (from popup)
 */
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  // Handshake: Content script requests its tabId
  if (msg?.type === "CS_HELLO") {
    sendResponse?.({ ok: true, tabId: sender?.tab?.id ?? null });
    return;
  }

  // Progress update from content script
  if (msg?.type === "SCAN_PROGRESS") {
    const tabId = (typeof msg.tabId === "number") ? msg.tabId : sender?.tab?.id;
    if (tabId != null) {
      const st = tabState.get(tabId) || {};
      st.inProgress = (msg.state === "start" || msg.state === "running" || msg.state === "finishing");
      if (msg.state === "done" || msg.state === "aborted" || msg.state === "error") {
        st.inProgress = false;
        st.lastScanAt = Date.now();
      }
      st.last = msg;
      tabState.set(tabId, st);

      // Update badge: show hit count on "done"
      if (msg.state === "done") {
        const badge = (msg?.hits ? String(msg.hits) : "");
        chrome.action.setBadgeText({ tabId, text: badge });
        if (badge) chrome.action.setBadgeBackgroundColor({ tabId, color: "#ef4444" });
      }
    }

    // Broadcast to popup
    chrome.runtime.sendMessage({ type: "SCAN_PROGRESS_BROADCAST", tabId, data: msg }).catch(() => {});
    sendResponse?.({ ok: true });
    return; // synchronous response
  }

  // Popup: scan active tab
  if (msg?.type === "RUN_SCAN_ACTIVE_TAB") {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => { if (tab?.id) maybeStartScan(tab.id, tab.url); });
    sendResponse?.({ ok: true });
    return;
  }

  // Popup: get status of active tab
  if (msg?.type === "GET_STATUS_FOR_ACTIVE_TAB") {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      const st = tab ? tabState.get(tab.id) : null;
      sendResponse?.({ ok: true, tabId: tab?.id ?? null, status: st ?? null });
    });
    return true; // async
  }

  // Popup: cancel scan
  if (msg?.type === "CANCEL_ACTIVE_SCAN") {
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (tab?.id) {
        const st = tabState.get(tab.id);
        sendToTab(tab.id, { type: "CANCEL_SCAN", runId: st?.lastRunId });
      }
    });
    sendResponse?.({ ok: true });
    return;
  }
});
