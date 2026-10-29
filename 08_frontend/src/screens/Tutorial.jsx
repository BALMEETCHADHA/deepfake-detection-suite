import { useState } from "react";
import { TYPES } from "../lib/types";
import { ICONS, ScanIcon, PdfIcon, WordIcon, PngIcon } from "../lib/icons";

const steps = [
  { key: "intro", title: "Welcome to the Forensic Suite", icon: ScanIcon,
    gives: "This system checks five kinds of evidence and tells you whether each one is REAL or a DEEP FAKE.",
    how: ["Pick what you want to check: Image, Video, Signature, Audio or Document.", "Upload your file and run the analysis.", "Read the big verdict, the percentage and the reasons, then download the report."] },
  ...TYPES.map((t, i) => ({ key: t.id, title: `${i + 1}. ${t.title}`, icon: ICONS[t.id], gives: t.gives, how: t.how })),
  { key: "result", title: "Reading the result", icon: ScanIcon,
    gives: "The result page shows one huge verdict: green REAL or red DEEP FAKE (amber UNCERTAIN when the evidence is too close to call).",
    how: ["Below it is the confidence percentage in the same colour.", "Then the reasons why, in plain text, plus model and technical details.", "At the bottom, download the full report as PDF, Word or Image, then use Previous, Exit or Restart."], extra: "downloads" },
];

export default function Tutorial({ onDone }) {
  const [i, setI] = useState(0);
  const s = steps[i];
  const Icon = s.icon;
  const last = i === steps.length - 1;
  return (
    <div className="page tutorial">
      <header className="topbar"><span>FORENSIC SUITE // TUTORIAL</span><span>STEP {i + 1} / {steps.length}</span></header>
      <div className="progress">{steps.map((_, k) => <i key={k} className={k <= i ? "on" : ""} />)}</div>
      <main className="tut-body" key={s.key}>
        <div className="tut-icon"><Icon size={84} /></div>
        <h1>{s.title}</h1>
        <p className="tut-gives">{s.gives}</p>
        <div className="tut-how">
          <div className="label">HOW TO USE</div>
          <ol>{s.how.map((h, k) => <li key={k}>{h}</li>)}</ol>
        </div>
        {s.extra === "downloads" && <div className="tut-dl"><PdfIcon size={40} /><WordIcon size={40} /><PngIcon size={40} /></div>}
      </main>
      <footer className="navbar">
        <button className="btn ghost" onClick={() => setI(i - 1)} disabled={i === 0}>◂ Previous</button>
        <div className="right">
          {!last && <button className="btn skip" onClick={onDone}>Skip</button>}
          <button className="btn primary" onClick={() => (last ? onDone() : setI(i + 1))}>{last ? "Start detection ▸" : "Next ▸"}</button>
        </div>
      </footer>
    </div>
  );
}
