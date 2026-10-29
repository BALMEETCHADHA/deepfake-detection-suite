// Run with:  npm run check
// Guards the one place where a backend/frontend mismatch silently shows the WRONG verdict:
// signatures return GENUINE/FORGED, everything else REAL/FAKE. Both must map to green/red, never "unsure".
import { createServer } from "vite";

const v = await createServer({ root: process.cwd(), server: { middlewareMode: true }, appType: "custom", logLevel: "silent" });
const { verdictView, typeById, TYPES } = await v.ssrLoadModule("/src/lib/types.js");
await v.close();

const r = (verdict) => ({ verdict, confidence_real_pct: 70, confidence_fake_pct: 30 });
const cases = [
  ["REAL", "image", "real"], ["FAKE", "image", "fake"], ["UNCERTAIN", "image", "unsure"],
  ["GENUINE", "signature", "real"], ["FORGED", "signature", "fake"], ["UNCERTAIN", "signature", "unsure"],
  ["REAL", "document", "real"], ["FAKE", "document", "fake"], ["forged", "signature", "fake"],
];
let bad = 0;
for (const [verdict, id, want] of cases) {
  const got = verdictView(r(verdict), typeById(id)).cls;
  if (got !== want) { bad++; console.error(`FAIL  ${id}: backend verdict "${verdict}" shows "${got}", expected "${want}"`); }
}
// every detector must have a run() and at least one required file field
for (const t of TYPES) if (typeof t.run !== "function" || !t.fields.some((f) => f.required)) { bad++; console.error(`FAIL  detector "${t.id}" is misconfigured`); }
console.log(bad ? `${bad} check(s) failed` : `all ${cases.length + TYPES.length} frontend checks passed`);
process.exit(bad ? 1 : 0);
