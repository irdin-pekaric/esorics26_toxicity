# Toxic Guardian User Guide

Welcome! This guide will help you get the most out of Toxic Guardian, your personal protection against toxic content online.

## What is Toxic Guardian?

Toxic Guardian is a browser extension that automatically scans web pages for toxic, harassing, or offensive content. When toxic content is detected, it's hidden behind a clickable banner - giving you control over what you see online.

**Key Benefits:**
- **Privacy First**: All scanning happens locally in your browser - nothing is sent to the cloud
- **You're in Control**: Toxic content is hidden but can be revealed with a single click
- **Works Everywhere**: Scans regular websites, social media, news sites, and more
- **Fast & Automatic**: Scans happen automatically as you browse

---

## Installation

### Step 1: Get the Extension

Currently, Toxic Guardian is available for manual installation:

1. Download the extension from [GitHub](https://github.com/irdin-pekaric/toxicity_WWW2026/tree/main/extension)
2. Unzip the downloaded file

### Step 2: Install in Your Browser

**For Chrome/Edge:**

1. Open your browser and go to `chrome://extensions/`
2. Turn on "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `toxic-guardian` folder you unzipped
5. The extension icon should appear in your toolbar

**For Firefox:**

1. Open Firefox and go to `about:debugging#/runtime/this-firefox`
2. Click "Load Temporary Add-on"
3. Navigate to the `toxic-guardian` folder and select `manifest.json`
4. The extension icon should appear in your toolbar

### Step 3: Pin the Extension (Optional)

For quick access, pin the extension to your toolbar:

1. Click the extensions icon (puzzle piece)
2. Find "Toxic Guardian (Local)"
3. Click the pin icon

---

## How to Use

### Automatic Scanning

Once installed, Toxic Guardian automatically scans pages as you browse:

1. Navigate to any website
2. Wait a moment while the page is scanned
3. The extension icon shows a badge with the number of toxic items found
4. Toxic content is automatically hidden with yellow banners

**That's it!** No configuration needed.

---

### Understanding the Badge

The extension icon displays a small badge:

- **No badge**: No toxic content found
- **Number badge (red)**: Number of toxic items detected on this page
- Example: A "3" means three pieces of toxic content were found

---

### Revealing Hidden Content

When toxic content is detected, you'll see a yellow banner that says:

```
toxic content — click to reveal
```

**To reveal:**
1. Click the banner with your mouse
2. Or use keyboard: Tab to the banner, then press Enter or Space

**After revealing:**
- The text becomes visible
- A small "revealed" badge appears
- You can read, select, and interact with the text normally

**Why reveal?** Sometimes you need to see the full context of a conversation or report abusive content. Toxic Guardian gives you the choice.

---

### Manual Scanning

If a page loads new content or you want to re-scan:

**Method 1: Extension Popup**
1. Click the extension icon
2. Click the "Run" button
3. Watch the progress bar

**Method 2: Right-Click Menu**
1. Right-click anywhere on the page
2. Select "Scan page for toxic content"

**Method 3: Browser Console** (Advanced)
1. Press F12 to open Developer Tools
2. Go to the Console tab
3. Type: `window.tgScanPage()`
4. Press Enter

---

### Navigating Between Toxic Content

Once a scan is complete, you can jump between detected items:

**Using the Popup:**
1. Click the extension icon to open the popup
2. Use the "Prev" and "Next" buttons to jump between items
3. Or use arrow keys (← and →) while the popup is open

**Keyboard Shortcuts:** (when popup is open)
- **Right Arrow (→)**: Next toxic item
- **Left Arrow (←)**: Previous toxic item

**Visual Feedback:**
- The page automatically scrolls to the item
- An orange highlight ring appears around the item
- The banner pulses to catch your attention

---

### Canceling a Scan

If a scan is taking too long or you want to stop it:

1. Click the extension icon
2. Click the "Cancel" button
3. The scan stops immediately

---

## Understanding Scan Results

### Progress Bar

When you open the extension popup during a scan, you'll see:

```
[Progress Bar: 75%]
234/312 · 75%
5 hits
State: running
```

**What this means:**
- **Progress bar**: Visual indicator of scan completion
- **234/312**: 234 out of 312 text elements scanned
- **75%**: Percentage complete
- **5 hits**: 5 toxic items found so far
- **State**: Current scan status

### Scan States

| State | Meaning |
|-------|---------|
| **ready** | No scan in progress |
| **starting...** | Scan requested, initializing |
| **running** | Actively scanning page content |
| **finishing** | Processing last batch |
| **done** | Scan completed successfully |
| **aborted** | Scan was cancelled |
| **error** | Something went wrong |

---

## What Content is Detected?

Toxic Guardian scans for six categories of harmful content:

1. **Harassment**: Personal attacks, bullying, intimidation
2. **Hate Speech**: Content targeting groups based on identity
3. **Insults**: Offensive name-calling and slurs
4. **Threats**: Expressions of intent to harm
5. **Profanity**: Explicit and vulgar language
6. **Sexual Content**: Inappropriate sexual comments

**Confidence Levels:**

When you hover over a toxic banner, you'll see the classification:
```
harassment: 87.3%
```

This means the system is 87.3% confident the content is harassment.

---

## Tips & Best Practices

### When to Reveal Content

**Good reasons to reveal:**
- You need full context of a discussion
- You're reporting abusive content to moderators
- You want to verify a false positive
- Content seems mislabeled

**You might skip revealing:**
- Content is clearly abusive based on surrounding context
- You're in a triggering or stressful state
- The banner provides enough information

---

### Dealing with False Positives

Sometimes innocent content gets flagged (false positives). This can happen with:

- Quotes discussing toxicity (like news articles)
- Historical texts or academic content
- Sarcasm and irony
- Strong opinions that aren't actually toxic

**What to do:**
1. Simply click to reveal - it's harmless
2. Remember the system is being cautious to protect you
3. False positives are rare and easily dismissed

---

### Dealing with False Negatives

Sometimes toxic content isn't detected (false negatives). This can happen with:

- Creative spelling to evade detection (e.g., "h@te")
- Subtle or coded language
- Non-English content
- Very short fragments

**What to do:**
1. Report the content through the website's official channels
2. Block or mute the user (if the platform supports it)
3. Remember: Toxic Guardian is a tool, not a replacement for platform moderation

---

## Privacy & Security

### What Data Does Toxic Guardian Collect?

**None.** Toxic Guardian:

- Does NOT send your browsing data anywhere
- Does NOT track which pages you visit
- Does NOT collect statistics or analytics
- Does NOT communicate with external servers
- Does NOT store your browsing history

**Everything happens locally on your device.**

---

### What Data is Stored?

The extension stores only one preference locally:

- **Threshold setting**: How sensitive the detection should be (default: 0.5)

This is stored using Chrome's `storage.local` API and never leaves your device.

---

## Troubleshooting

### Extension Icon Not Showing

**Try:**
1. Refresh the extensions page: `chrome://extensions/`
2. Make sure the extension is enabled (toggle switch on)
3. Restart your browser
4. Reinstall the extension

---

### Nothing is Being Detected

**Check:**
1. Does the page have actual content? (Some pages are mostly images)
2. Is JavaScript enabled in your browser?
3. Try manually scanning: Click icon → "Run"
4. Check the browser console for errors (F12 → Console)

---

### Too Much Content is Flagged

The detection might be too sensitive.

**To adjust:** (Advanced - requires browser console)
```javascript
// Make detection less sensitive (0.0 = everything, 1.0 = nothing)
chrome.storage.local.set({ threshold: 0.7 });
```

Then reload the extension.

---

### Too Little Content is Flagged

The detection might not be sensitive enough.

**To adjust:** (Advanced)
```javascript
// Make detection more sensitive
chrome.storage.local.set({ threshold: 0.3 });
```

Then reload the extension.

---

### Scans Take Forever

**Possible causes:**
- Very large pages (1000+ text elements)
- Computer is under heavy load
- Browser is low on memory

**Solutions:**
1. Click "Cancel" and try again
2. Close other tabs and applications
3. Wait for page to fully load before scanning
4. Restart browser if problem persists

---

## Frequently Asked Questions

### Does this work on all websites?

Almost! It works on:
- Regular websites (http/https)
- Local files (file://)
- Most web applications

It does NOT work on:
- Browser internal pages (chrome://, about:)
- PDF viewers
- Most browser extensions

---

### Can websites detect that I'm using this?

No. Websites cannot detect the extension because:
- It doesn't modify page behavior
- It only adds visual overlays
- It doesn't inject tracking code
- Changes are only visible to you

---

### Does this replace website moderation?

No. Toxic Guardian is a personal protection tool. It:
- Helps you control your exposure to toxic content
- Does not remove content from websites
- Does not report users to moderators
- Does not prevent others from seeing content

**Always report serious violations to website moderators.**

---

### Can I customize which words are flagged?

Not currently. The extension uses a machine learning model that evaluates:
- Context
- Tone
- Intent
- Combinations of words

It's more sophisticated than simple keyword blocking.

---

### Why is this extension free?

Toxic Guardian is an open-source project. It's free because:
- No servers to maintain (everything is local)
- No data collection or advertising
- Community-driven development
- Passion project by the creator

---

### How can I support the project?

- Star the project on GitHub
- Report bugs and issues
- Suggest improvements
- Share with friends who might benefit
- Contribute code (see DEVELOPER.md)

---

## Accessibility

Toxic Guardian is designed to be accessible:

### Screen Readers

- All banners have ARIA labels
- Role attributes for proper navigation

---


## Quick Reference

| Action | Method |
|--------|--------|
| Reveal toxic content | Click yellow banner |
| Manual scan | Icon → "Run" |
| Cancel scan | Icon → "Cancel" |
| Next item | Icon → "Next" or → key |
| Previous item | Icon → "Prev" or ← key |
| Console scan | `window.tgScanPage()` |
| Change threshold | `chrome.storage.local.set({ threshold: 0.7 })` |

---

**Version:** 0.1.0
**Last Updated:** October 2025
