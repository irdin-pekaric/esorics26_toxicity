// Example preprocess: convert string to a toy numeric feature
// Replace with real tokenization for your model (e.g., wordpiece, BPE, etc.)
export function simpleFeatures(text) {
  // Very naive: length and number of exclamation marks
  const len = Math.min(512, text.length);
  const bangs = (text.match(/!/g) || []).length;
  return new Float32Array([len, bangs]);
}

// For real models, implement:
// export function tokenize(text) -> { inputIds: Int32Array, attentionMask: Int32Array, ... }
// ...whatever your ONNX expects.
