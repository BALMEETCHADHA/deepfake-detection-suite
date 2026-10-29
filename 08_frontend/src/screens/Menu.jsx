import { TYPES } from "../lib/types";
import { ICONS } from "../lib/icons";

export default function Menu({ health, info, onPick, onExit, onTutorial }) {
  return (
    <div className="page">
      <header className="topbar">
        <span>FORENSIC SUITE // DETECTION</span>
        <span><i className={`dot ${health}`} />{health === "ok" ? "API connected" : health === "down" ? "API offline" : health === "demo" ? "DEMO MODE - results are not real" : "checking…"}</span>
      </header>
      {health === "ok" && info && "image_model_is_finetuned" in info && (
        <div className={`model-banner ${info.image_model_is_finetuned ? "good" : "warn"}`}>
          {info.image_model_is_finetuned
            ? "Image model: fine-tuned on your data"
            : "Image model: ORIGINAL PRETRAINED model. It is close to a coin flip on this project's test data, so real photos can be called fake. Load the fine-tuned model in the Colab notebook (section 3), then restart the server."}
          {info.face_detection === false && ` Face detection is off: ${info.face_detection_reason || "unknown reason"}.`}
        </div>
      )}
      <main className="center">
        <h1 className="page-title">What do you want to check?</h1>
        <p className="page-sub">Choose one of the five detectors</p>
        <div className="cards">
          {TYPES.map((t, i) => {
            const Icon = ICONS[t.id];
            return (
              <button key={t.id} className="card" onClick={() => onPick(t.id)}>
                <span className="card-n">0{i + 1}</span>
                <Icon size={64} />
                <strong>{t.label}</strong>
                <small>{t.gives}</small>
              </button>
            );
          })}
        </div>
      </main>
      <footer className="navbar">
        <button className="btn ghost" onClick={onTutorial}>◂ Tutorial</button>
        <button className="btn ghost" onClick={onExit}>Exit</button>
      </footer>
    </div>
  );
}
