import { useEffect, useState } from "react";
import { api } from "./api";
import { typeById } from "./lib/types";
import Landing from "./screens/Landing";
import Tutorial from "./screens/Tutorial";
import Menu from "./screens/Menu";
import Detector from "./screens/Detector";
import Result from "./screens/Result";

// Screens: landing -> tutorial -> menu -> detect -> result
// Landing also has a DETECTION button that jumps straight to the menu.
export default function App() {
  const [screen, setScreen] = useState("landing");
  const [typeId, setTypeId] = useState(null);
  const [inputs, setInputs] = useState({});
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [health, setHealth] = useState("checking");
  const [info, setInfo] = useState(null);       // what the backend reports about itself (which image model is live...)

  useEffect(() => {
    api.health().then((h) => { setHealth(h?.status === "demo" ? "demo" : "ok"); setInfo(h); }).catch(() => setHealth("down"));
  }, []);

  const type = typeById(typeId);
  const values = inputs[typeId] || {};

  async function run() {
    setLoading(true);
    setError(null);
    try {
      setResult(await type.run(values));
      setScreen("result");
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  const exit = () => { setInputs({}); setResult(null); setError(null); setTypeId(null); setScreen("landing"); };
  const restart = () => { setInputs({}); setResult(null); setError(null); setTypeId(null); setScreen("menu"); };

  if (screen === "tutorial") return <Tutorial onDone={() => setScreen("menu")} />;
  if (screen === "menu") return <Menu health={health} info={info} onPick={(id) => { setTypeId(id); setError(null); setScreen("detect"); }} onExit={exit} onTutorial={() => setScreen("tutorial")} />;
  if (screen === "detect" && type)
    return <Detector type={type} values={values} setValues={(v) => setInputs({ ...inputs, [typeId]: v })} loading={loading} error={error} onRun={run} onBack={() => setScreen("menu")} onExit={exit} />;
  if (screen === "result" && type && result)
    return <Result result={result} type={type} onPrevious={() => setScreen("detect")} onExit={exit} onRestart={restart} />;
  return <Landing health={health} onTutorial={() => setScreen("tutorial")} onDetect={() => setScreen("menu")} />;
}
