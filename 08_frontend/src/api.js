const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// ngrok's free tier shows an HTML "you're about to visit..." interstitial to
// browser requests unless this header is present — without it, fetch() gets
// HTML back instead of JSON and res.json() throws. Harmless when the backend
// is just plain localhost (no ngrok in the path), so it's always safe to send.
const NGROK_SKIP_HEADER = { "ngrok-skip-browser-warning": "true" };

const DEMO = import.meta.env.VITE_DEMO === "1";

// Demo mode (VITE_DEMO=1): lets you try the whole UI without the backend running.
// A file name containing "fake" gives a FAKE verdict, anything else gives REAL.
async function demoResult(path, fields) {
  await new Promise((r) => setTimeout(r, 2200));
  const name = fields.file?.name || fields.questioned?.name || "sample";
  const fake = /fake|forg/i.test(name);
  const f = fake ? 96.4 : 3.1;
  return {
    file: name, modality: path.split("/").pop(), verdict: fake ? "FAKE" : "REAL",
    confidence_real_pct: +(100 - f).toFixed(1), confidence_fake_pct: f,
    confidence_level: "high", decision_margin_pct: +Math.abs(100 - 2 * f).toFixed(1),
    model_used: "DEMO - no backend connected, results are made up", model_reported_accuracy_pct: null, processing_time_sec: 2.2,
    evidence: fake
      ? ["Frequency-domain artifacts typical of generative models were found.", "Skin texture is unnaturally smooth around the eyes and mouth.", "Error-level analysis shows inconsistent compression across the face."]
      : ["No generative-model artifacts found in the frequency spectrum.", "Natural sensor noise pattern is consistent across the whole file.", "Compression history is uniform; no spliced regions detected."],
    sub_results: [], warnings: ["Demo mode: these numbers are made up to show the interface."],
    scores_breakdown: { model_score: fake ? 0.97 : 0.04, forensic_score: fake ? 0.91 : 0.08 },
  };
}

async function postFiles(path, fields) {
  if (DEMO) return demoResult(path, fields);
  const form = new FormData();
  for (const [key, value] of Object.entries(fields)) {
    if (value !== null && value !== undefined) form.append(key, value);
  }
  const res = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: NGROK_SKIP_HEADER,
    body: form,
  });

  const contentType = res.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) {
    throw new Error(
      "Backend returned a non-JSON response. If you're using an ngrok URL, " +
      "this usually means the ngrok interstitial page was served instead of " +
      "the API — check VITE_API_URL is correct and the tunnel is still running."
    );
  }

  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.message || data.detail || `Request failed (${res.status})`);
  }
  return data;
}

export const api = {
  health: () =>
    DEMO ? Promise.resolve({ status: "demo" }) : fetch(`${API_URL}/api/health`, { headers: NGROK_SKIP_HEADER }).then((r) => r.json()),
  image: (file) => postFiles("/api/image", { file }),
  document: (file) => postFiles("/api/document", { file }),
  audio: (file, reference) => postFiles("/api/audio", { file, reference }),
  video: (file, maxFrames) => postFiles("/api/video", { file, max_frames: maxFrames }),
  signature: (questioned, reference) =>
    postFiles("/api/signature", { questioned, reference }),
};
