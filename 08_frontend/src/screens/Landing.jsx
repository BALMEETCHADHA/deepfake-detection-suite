import realImg from "../assets/real.jpg";
import fakeImg from "../assets/fake.jpg";
import { useEffect, useRef } from "react";

export default function Landing({ health, onTutorial, onDetect }) {
  const ro = useRef(null);
  useEffect(() => {
    let raf;
    const tick = () => {
      const t = (Date.now() % 12000) / 12000;
      if (ro.current) {
        if (t < 0.1) ro.current.textContent = "ACQUIRING SUBJECT…";
        else if (t < 0.48) ro.current.textContent = `SCANNING FACIAL MESH… ${Math.min(100, Math.round(((t - 0.1) / 0.36) * 100))}%`;
      }
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, []);

  return (
    <div className="landing">
      <svg width="0" height="0" style={{ position: "absolute" }}>
        <filter id="warp">
          <feTurbulence type="fractalNoise" baseFrequency="0.02 0.06" numOctaves="2" seed="4">
            <animate attributeName="seed" values="4;9;4" dur="3s" repeatCount="indefinite" />
          </feTurbulence>
          <feDisplacementMap in="SourceGraphic" scale="9" />
        </filter>
      </svg>
      <header className="topbar"><span><i className={`dot ${health}`} />FORENSIC LAB // <b>{health === "ok" ? "ONLINE" : health === "down" ? "API OFFLINE" : health === "demo" ? "DEMO MODE" : "CHECKING"}</b></span><span>DEEPFAKE &amp; FORGERY DETECTION</span></header>
      <h1 className="hero-title">FORENSIC SUITE</h1>
      <p className="hero-sub">SEE WHAT’S REAL. EXPOSE WHAT’S NOT.</p>

      <div className="stage">
        <div className="brackets" /><div className="scan" />
        <div className="readout" ref={ro}>ACQUIRING SUBJECT…</div>
        <div className="face real"><img alt="Real subject" src={realImg} /></div>
        <div className="face fake"><div className="glitch"><img alt="Deepfake subject" src={fakeImg} /></div></div>
        <div className="frame l" />
        <div className="frame r">
          <span className="flag" style={{ left: "20%", top: "38%" }} /><span className="flag" style={{ left: "56%", top: "38%" }} /><span className="flag" style={{ left: "38%", top: "62%" }} />
        </div>
        <div className="tag l"><h2>REAL</h2><p>AUTHENTIC</p></div>
        <div className="tag r"><h2>DEEP FAKE</h2><p>MANIPULATED</p></div>
      </div>

      <div className="cta-row">
        <button className="cta" onClick={onTutorial}>TUTORIAL ▸</button>
        <button className="cta alt" onClick={onDetect}>DETECTION ▸</button>
      </div>
      <div className="hint">First time here? Take the tutorial. Already know the system? Go straight to detection.</div>
    </div>
  );
}
