const BG = "#020611";
const safe = (s) => String(s || "report").replace(/\.[^.]+$/, "").replace(/[^a-z0-9_-]+/gi, "_");
const stamp = () => new Date().toISOString().slice(0, 10);

function save(blob, name) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

const snap = async (el) => {
  const { default: html2canvas } = await import("html2canvas");   // loaded on first download only
  return html2canvas(el, { backgroundColor: BG, scale: 2, useCORS: true });
};

export async function downloadImage(el, file) {
  const canvas = await snap(el);
  canvas.toBlob((b) => save(b, `report_${safe(file)}_${stamp()}.png`), "image/png");
}

export async function downloadPDF(el, file) {
  const canvas = await snap(el);
  const { jsPDF } = await import("jspdf");
  const pdf = new jsPDF("p", "mm", "a4");
  const W = pdf.internal.pageSize.getWidth();
  const H = pdf.internal.pageSize.getHeight();
  const imgH = (canvas.height * W) / canvas.width;
  const img = canvas.toDataURL("image/jpeg", 0.92);
  let y = 0;
  do {
    pdf.setFillColor(2, 6, 17);
    pdf.rect(0, 0, W, H, "F");
    pdf.addImage(img, "JPEG", 0, -y, W, imgH);
    y += H;
    if (y < imgH) pdf.addPage();
  } while (y < imgH);
  pdf.save(`report_${safe(file)}_${stamp()}.pdf`);
}

const esc = (s) => String(s ?? "").replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// Word-compatible HTML document (opens in Microsoft Word / LibreOffice / Google Docs).
export function downloadWord(result, view, typeTitle) {
  const color = view.cls === "real" ? "#0a8a4a" : view.cls === "fake" ? "#d11a2e" : "#b7791f";
  const list = (a) => (a?.length ? `<ul>${a.map((x) => `<li>${esc(x)}</li>`).join("")}</ul>` : "<p>None</p>");
  const rows = (o) => Object.entries(o || {}).map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(typeof v === "object" ? JSON.stringify(v) : v)}</td></tr>`).join("");
  const html = `<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word"><head><meta charset="utf-8"><title>Report</title></head>
<body style="font-family:Calibri,Arial,sans-serif">
<h2>${esc(typeTitle)} - Forensic Report</h2>
<p><b>File:</b> ${esc(result.file)}<br><b>Date:</b> ${esc(new Date().toLocaleString())}</p>
<h1 style="color:${color};font-size:40pt">${esc(view.text)}</h1>
<h2 style="color:${color}">${Number(view.pct).toFixed(1)}% ${esc(view.what)}</h2>
<p>Authentic: ${Number(result.confidence_real_pct).toFixed(1)}% &nbsp; | &nbsp; Manipulated: ${Number(result.confidence_fake_pct).toFixed(1)}%${result.confidence_level ? ` &nbsp; | &nbsp; Confidence: ${esc(result.confidence_level)}` : ""}</p>
<h3>Why this verdict</h3>${list(result.evidence)}
${result.sub_results?.length ? `<h3>Breakdown</h3><ul>${result.sub_results.map((s) => `<li>${esc(s.file)}: ${esc(s.verdict)} (${Number(s.confidence_fake_pct).toFixed(1)}% fake)</li>`).join("")}</ul>` : ""}
${result.warnings?.length ? `<h3>Warnings</h3>${list(result.warnings)}` : ""}
<h3>Technical details</h3><table border="1" cellpadding="4" cellspacing="0">
<tr><td>Model</td><td>${esc(result.model_used)}</td></tr>
<tr><td>Reported accuracy</td><td>${esc(result.model_reported_accuracy_pct ?? "n/a")}</td></tr>
<tr><td>Processing time (s)</td><td>${esc(result.processing_time_sec ?? "n/a")}</td></tr>
${rows(result.scores_breakdown)}</table></body></html>`;
  save(new Blob(["\ufeff", html], { type: "application/msword" }), `report_${safe(result.file)}_${stamp()}.doc`);
}
