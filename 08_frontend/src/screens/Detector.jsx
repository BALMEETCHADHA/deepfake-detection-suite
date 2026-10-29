import FileDrop from "../components/FileDrop";
import { ICONS } from "../lib/icons";

export default function Detector({ type, values, setValues, loading, error, onRun, onBack, onExit }) {
  const Icon = ICONS[type.id];
  const ready = type.fields.filter((f) => f.required).every((f) => values[f.key]);
  const set = (k, v) => setValues({ ...values, [k]: v });
  return (
    <div className="page">
      <header className="topbar"><span>FORENSIC SUITE // {type.label.toUpperCase()}</span><span>STEP 2 / 3 · UPLOAD</span></header>
      <main className="center narrow">
        <div className="tut-icon"><Icon size={72} /></div>
        <h1 className="page-title">{type.title}</h1>
        <p className="page-sub">{type.gives}</p>
        <div className="panel">
          <div className="field-row">
            {type.fields.map((f) =>
              f.type === "number" ? (
                <div className="field" key={f.key}>
                  <label className="field-label" htmlFor={f.key}>{f.label}</label>
                  <input id={f.key} className="number-input" type="number" min="2" max="60" value={values[f.key] ?? f.def} onChange={(e) => set(f.key, Math.min(60, Math.max(2, Math.round(Number(e.target.value)) || f.def)))} />
                </div>
              ) : (
                <FileDrop key={f.key} id={f.key} label={f.label} optional={f.optional} accept={f.accept} file={values[f.key]} onChange={(file) => set(f.key, file)} />
              )
            )}
          </div>
          {type.note && !values.reference && <div className="warning-box">{type.note}</div>}
          {error && <div className="error-banner">{error}</div>}
          <button className="btn primary big" disabled={!ready || loading} onClick={onRun}>{loading ? "Analysing…" : "Run analysis ▸"}</button>
        </div>
      </main>
      <footer className="navbar">
        <button className="btn ghost" onClick={onBack} disabled={loading}>◂ Previous</button>
        <button className="btn ghost" onClick={onExit} disabled={loading}>Exit</button>
      </footer>
      {loading && (
        <div className="overlay"><div className="scan-ring"><i /></div><p>ANALYSING EVIDENCE…</p><small>The first run of a detector also downloads its model and can take a minute.</small></div>
      )}
    </div>
  );
}
