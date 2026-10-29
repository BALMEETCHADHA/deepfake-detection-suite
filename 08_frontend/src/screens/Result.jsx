import { useRef, useState } from "react";
import { verdictView } from "../lib/types";
import { downloadImage, downloadPDF, downloadWord } from "../lib/report";
import { PdfIcon, WordIcon, PngIcon } from "../lib/icons";

export default function Result({ result, type, onPrevious, onExit, onRestart }) {
  const ref = useRef(null);
  const [busy, setBusy] = useState("");
  const v = verdictView(result, type);
  const { evidence = [], warnings = [], sub_results = [], scores_breakdown = {}, speaker_comparison } = result;
  const pct = (n) => `${Number(n).toFixed(1)}%`;

  async function run(kind, fn) {
    setBusy(kind);
    try { await fn(); } catch (e) { alert(`Could not create the ${kind}: ${e.message}`); }
    setBusy("");
  }

  return (
    <div className="page result">
      <header className="topbar"><span>FORENSIC SUITE // RESULT</span><span>STEP 3 / 3</span></header>
      <main className="center narrow">
        <div className="report-sheet" ref={ref}>
          <div className="rs-head"><span>{type.title.toUpperCase()}</span><span>{result.file}</span></div>
          <div className={`verdict ${v.cls}`}>{v.text}</div>
          <div className={`verdict-pct ${v.cls}`}>{pct(v.pct)} <small>{v.what}</small></div>
          <div className="split">
            <div className="real"><b>{pct(result.confidence_real_pct)}</b><span>Authentic</span></div>
            <div className="fake"><b>{pct(result.confidence_fake_pct)}</b><span>Manipulated</span></div>
          </div>
          {result.confidence_level && (
            <p className="meta">Evidence strength: <b>{result.confidence_level}</b> (decision margin {result.decision_margin_pct} pts){result.confidence_level === "low" && " — treat this verdict as tentative."}</p>
          )}

          <h3>Why this verdict</h3>
          {evidence.length ? <ul className="evidence">{evidence.map((e, i) => <li key={i}>{e}</li>)}</ul> : <p>No detailed evidence was returned.</p>}

          {speaker_comparison && <><h3>Speaker comparison</h3><p>Similarity {speaker_comparison.mfcc_cosine_similarity} · {speaker_comparison.verdict}</p></>}
          {sub_results.length > 0 && (
            <><h3>Breakdown ({sub_results.length})</h3>
              {sub_results.map((s, i) => <div className="row" key={i}><span>{s.file}</span><span>{s.verdict} · {pct(s.confidence_fake_pct)} fake</span></div>)}</>
          )}
          {warnings.map((w, i) => <div className="warning-box" key={i}>{w}</div>)}

          <h3>Technical details</h3>
          <div className="row"><span>Model</span><span>{result.model_used}</span></div>
          {result.model_reported_accuracy_pct != null && <div className="row"><span>Reported accuracy</span><span>{result.model_reported_accuracy_pct}%</span></div>}
          {result.processing_time_sec != null && <div className="row"><span>Processing time</span><span>{result.processing_time_sec}s</span></div>}
          {Object.entries(scores_breakdown).map(([k, val]) => <div className="row" key={k}><span>{k}</span><span>{typeof val === "object" ? JSON.stringify(val) : String(val)}</span></div>)}
          <p className="stamp">{new Date().toLocaleString()}</p>
        </div>

        <div className="downloads">
          <span className="label">DOWNLOAD FULL REPORT</span>
          <div>
            <button className="dl" disabled={!!busy} onClick={() => run("PDF", () => downloadPDF(ref.current, result.file))}><PdfIcon size={52} /><small>{busy === "PDF" ? "…" : "PDF"}</small></button>
            <button className="dl" disabled={!!busy} onClick={() => run("Word file", async () => downloadWord(result, v, type.title))}><WordIcon size={52} /><small>Word</small></button>
            <button className="dl" disabled={!!busy} onClick={() => run("image", () => downloadImage(ref.current, result.file))}><PngIcon size={52} /><small>{busy === "image" ? "…" : "Image"}</small></button>
          </div>
        </div>

        <div className="stack">
          <button className="btn ghost" onClick={onPrevious}>◂ Previous</button>
          <button className="btn ghost" onClick={onExit}>Exit</button>
          <button className="btn primary" onClick={onRestart}>↻ Restart</button>
        </div>
      </main>
    </div>
  );
}
