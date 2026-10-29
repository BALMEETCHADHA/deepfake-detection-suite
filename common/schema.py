"""
common/schema.py
-----------------
Standardized result object used by EVERY modality module (image, document,
audio, video, signature) so the frontend / any consumer of this backend only
has to understand one response shape, no matter which detector produced it.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import List, Optional
import json
import os


@dataclass
class DetectionResult:
    file: str
    modality: str                       # "image" | "document" | "audio" | "video" | "signature"
    verdict: str                        # "REAL" | "FAKE" | "UNCERTAIN"
    confidence_real_pct: float
    confidence_fake_pct: float
    model_used: str
    model_reported_accuracy_pct: Optional[float]
    evidence: List[str] = field(default_factory=list)
    sub_results: List[dict] = field(default_factory=list)  # e.g. per-frame / per-image-in-doc
    processing_time_sec: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    warnings: List[str] = field(default_factory=list)
    # How the final number was produced (raw model score, face crop, calibration,
    # forensic fusion...). Lets a reader audit a 53/47 result instead of trusting it blindly.
    scores_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())

    def print_report(self) -> None:
        bar = "=" * 60
        print(bar)
        print(f" DEEPFAKE / FORGERY DETECTION REPORT  [{self.modality.upper()}]")
        print(bar)
        print(f" File            : {self.file}")
        print(f" VERDICT         : {self.verdict}")
        print(f" Confidence REAL : {self.confidence_real_pct:.2f}%")
        print(f" Confidence FAKE : {self.confidence_fake_pct:.2f}%")
        print(f" Model used      : {self.model_used}")
        if self.model_reported_accuracy_pct is not None:
            print(f" Model's own benchmark accuracy: {self.model_reported_accuracy_pct:.2f}%")
        print(" Evidence / proof:")
        for i, e in enumerate(self.evidence, 1):
            print(f"   {i}. {e}")
        if self.warnings:
            print(" Warnings:")
            for w in self.warnings:
                print(f"   - {w}")
        if self.processing_time_sec is not None:
            print(f" Processed in    : {self.processing_time_sec:.2f}s")
        print(bar)


def default_uncertain_band() -> float:
    """Points within which two scores count as a coin flip. Env UNCERTAIN_BAND overrides (0 = never answer UNCERTAIN)."""
    try:
        return max(0.0, float(os.environ.get("UNCERTAIN_BAND", "6")))
    except ValueError:
        return 6.0


def verdict_from_scores(fake_pct: float, real_pct: float, uncertain_band: Optional[float] = None) -> str:
    """Shared decision rule: if the two scores are within `uncertain_band` points
    of each other near 50/50, report UNCERTAIN instead of forcing a call.

    BUGFIX (was 10.0): a 10-point band meant any result between roughly 45/55
    was forced to UNCERTAIN, which swallowed clearly-leaning results (e.g.
    REAL 53% vs FAKE 47%, a 6-point lead) across every modality, since all
    five detectors share this one function. 6.0 keeps genuinely ambiguous
    near-coin-flip results (e.g. 51/49) as UNCERTAIN while letting anything
    with a real lead resolve to REAL or FAKE.
    """
    if uncertain_band is None:
        uncertain_band = default_uncertain_band()
    if abs(fake_pct - real_pct) <= uncertain_band:
        return "UNCERTAIN"
    return "FAKE" if fake_pct > real_pct else "REAL"
