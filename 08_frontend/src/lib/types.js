import { api } from "../api";

// Order here = order of the menu cards AND of the tutorial pages.
export const TYPES = [
  {
    id: "image", label: "Image", title: "Image Detection", verdictFake: "DEEP FAKE",
    gives: "Tells you if a photo is a real camera picture or AI-generated / face-swapped.",
    how: ["Choose one JPEG or PNG (a face works best, any photo is accepted).", "Press Run analysis.", "Read the verdict, the percentage and the evidence."],
    fields: [{ key: "file", label: "Image file", accept: "image/*", required: true }],
    run: (v) => api.image(v.file),
  },
  {
    id: "video", label: "Video", title: "Video Detection", verdictFake: "DEEP FAKE",
    gives: "Checks sampled frames, the audio track and frame-to-frame flicker for deepfake traces.",
    how: ["Choose a video file.", "Set how many frames to sample (more frames = slower but stronger).", "Press Run analysis and wait; videos take longer."],
    fields: [{ key: "file", label: "Video file", accept: "video/*", required: true }, { key: "frames", label: "Frames to sample", type: "number", def: 12 }],
    run: (v) => api.video(v.file, v.frames ?? 12),
  },
  {
    id: "signature", label: "Signature", title: "Signature Verification", verdictFake: "FORGED", verdictReal: "GENUINE",
    gives: "Compares a questioned signature with a known-genuine one to spot forgeries.",
    how: ["Add a known-genuine reference signature (strongly recommended).", "Add the questioned signature image.", "Press Run analysis."],
    fields: [{ key: "reference", label: "Known-genuine reference", accept: "image/*", optional: true }, { key: "file", label: "Questioned signature", accept: "image/*", required: true }],
    run: (v) => api.signature(v.file, v.reference),
    note: "Without a reference this runs a weak fallback check only. Add a reference for a real verdict.",
  },
  {
    id: "audio", label: "Audio", title: "Audio Detection", verdictFake: "DEEP FAKE",
    gives: "Separates a real human voice from AI-cloned or synthetic speech.",
    how: ["Choose the audio clip to check.", "Optionally add a known-real voice of the same person to compare speakers.", "Press Run analysis."],
    fields: [{ key: "file", label: "Audio to check", accept: "audio/*", required: true }, { key: "reference", label: "Known-real reference voice", accept: "audio/*", optional: true }],
    run: (v) => api.audio(v.file, v.reference),
  },
  {
    id: "document", label: "Document", title: "Document Detection", verdictFake: "FORGED", verdictReal: "GENUINE",
    gives: "Scans a PDF, DOCX or PPTX: every embedded image plus the authoring metadata.",
    how: ["Choose a PDF, DOCX or PPTX file.", "Press Run analysis.", "Check the per-image breakdown in the report."],
    fields: [{ key: "file", label: "Document file", accept: ".pdf,.docx,.pptx", required: true }],
    run: (v) => api.document(v.file),
  },
];
export const typeById = (id) => TYPES.find((t) => t.id === id);

// The backend uses REAL/FAKE for most detectors but GENUINE/FORGED for signatures, so accept both vocabularies.
export function verdictView(result, type) {
  const v = (result?.verdict || "").toUpperCase();
  if (v === "REAL" || v === "GENUINE") return { text: type?.verdictReal || "REAL", cls: "real", pct: result.confidence_real_pct, what: "authentic" };
  if (v === "FAKE" || v === "FORGED") return { text: type?.verdictFake || "DEEP FAKE", cls: "fake", pct: result.confidence_fake_pct, what: "manipulated" };
  return { text: "UNCERTAIN", cls: "unsure", pct: Math.max(result?.confidence_real_pct ?? 0, result?.confidence_fake_pct ?? 0), what: "inconclusive" };
}
