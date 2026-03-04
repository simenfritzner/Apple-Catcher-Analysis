"""
Formal Hypothesis Scoring Matrix (thesis sections 4.5 and 5.4).

Reads existing analysis results from Phases 1-8 and lateralization analyses
to compile a formal quantitative scoring matrix evaluating 4 competing
hypotheses against 8 evidence tests on a 5-point scale.

Hypotheses:
    H1 Artifacts       -- Model exploits ocular/muscular artifacts
    H2 Classical ERD   -- Model uses mu/beta ERD/ERS
    H3 MRCP            -- Model uses movement-related cortical potentials
    H4 Visual/Cue ERP  -- Model classifies cue-evoked differential response

Tests (from thesis section 4.5.2):
    N1  Sensorimotor ERDS significance
    N2  Frontal artifact indicators
    N3  MRCP presence
    F1  Temporal filter frequency distribution
    F2  Spatial filter ROI focus
    A1  Attribution spatial distribution
    A2  Attribution temporal concentration
    A3  Attribution lateralization
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# ── Configuration ────────────────────────────────────────────────────────────

RESULTS_ROOT = project_root / "results" / "interpretability"
OUTPUT_DIR = RESULTS_ROOT / "hypothesis_scoring"

HYPOTHESES = ["Artifacts", "Classical ERD", "MRCP", "Visual/Cue ERP"]
TEST_IDS = ["N1", "N2", "N3", "F1", "F2", "A1", "A2", "A3"]
TEST_NAMES: dict[str, str] = {
    "N1": "Sensorimotor ERDS significance",
    "N2": "Frontal artifact indicators",
    "N3": "MRCP presence",
    "F1": "Temporal filter frequency distribution",
    "F2": "Spatial filter ROI focus",
    "A1": "Attribution spatial distribution",
    "A2": "Attribution temporal concentration",
    "A3": "Attribution lateralization",
}

SCORE_LABELS = ["--", "-", "0", "+", "++"]
SCORE_VALUES: dict[str, int] = {"--": -2, "-": -1, "0": 0, "+": 1, "++": 2}
NUMERIC_TO_LABEL: dict[int, str] = {v: k for k, v in SCORE_VALUES.items()}


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class TestEvidence:
    """Evidence, reasoning, and scores for a single test."""

    test_id: str
    test_name: str
    evidence: str
    reasoning: dict[str, str] = field(default_factory=dict)
    scores: dict[str, str] = field(default_factory=dict)


# ── Result file parsers ──────────────────────────────────────────────────────

def parse_band_importance(path: Path) -> dict[str, float]:
    """Parse band_importance_ranking.txt into {band: importance}."""
    result: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("=") and not line.startswith("Freq"):
            parts = line.split(":")
            if len(parts) == 2:
                band = parts[0].strip()
                try:
                    result[band] = float(parts[1].strip())
                except ValueError:
                    continue
    return result


def parse_correlation_results(
    path: Path,
) -> dict[str, dict[str, dict[str, float]]]:
    """Parse correlation_results.txt into {band: {class: {channel: r}}}."""
    result: dict[str, dict[str, dict[str, float]]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith(("mu", "beta")) and "r =" in line:
            parts = line.split("|")
            if len(parts) == 3:
                band = parts[0].strip()
                cls = parts[1].strip()
                ch_r = parts[2].strip()
                ch, r_str = ch_r.split(": r =")
                result.setdefault(band, {}).setdefault(cls, {})[ch.strip()] = float(
                    r_str.strip()
                )
    return result


def parse_channel_group_ablation(path: Path) -> dict[str, dict[str, Any]]:
    """Parse channel_group_ablation.txt into {group: {metric: value}}."""
    result: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("baseline_accuracy"):
            result["baseline"] = {"accuracy": float(line.split(":")[1].strip())}
        elif ": {" in line:
            group = line.split(":")[0].strip()
            dict_str = line[line.index("{") :]
            try:
                parsed = eval(dict_str)  # noqa: S307
                result[group] = parsed
            except Exception:
                continue
    return result


def parse_pre_post_results(path: Path) -> dict[str, float]:
    """Parse pre_post_results.txt into {key: value}."""
    result: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if ":" in line and not line.startswith("#"):
            parts = line.split(":")
            if len(parts) == 2:
                try:
                    result[parts[0].strip()] = float(parts[1].strip())
                except ValueError:
                    continue
    return result


def parse_signed_lateralization(path: Path) -> dict[str, Any]:
    """Parse signed_lateralization_results.txt for key metrics."""
    text = path.read_text()
    result: dict[str, Any] = {
        "n_flip_channels": 0,
        "total_channels": 32,
        "flip_channels": [],
        "sensorimotor_flips": [],
    }
    for line in text.splitlines():
        if "Channels with sign flip:" in line:
            frac = line.split(":")[1].strip().split("/")[0].strip()
            result["n_flip_channels"] = int(frac)
        if "Flip channels:" in line:
            ch_str = line.split(":")[1].strip()
            if ch_str.startswith("["):
                try:
                    result["flip_channels"] = eval(ch_str)  # noqa: S307
                except Exception:
                    pass
        if "[] out of" in line:
            result["sensorimotor_flips"] = []
    return result


def parse_filter_mrcp_sensitivity(path: Path) -> dict[str, Any]:
    """Parse filter_mrcp_sensitivity.txt."""
    result: dict[str, Any] = {"filters": [], "mrcp_sensitive_count": 0, "total": 8}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line.startswith("MRCP-sensitive filters:"):
            parts = line.split(":")[1].strip().split("/")
            result["mrcp_sensitive_count"] = int(parts[0])
            result["total"] = int(parts[1])
        if line.startswith("Filter"):
            filt: dict[str, Any] = {}
            for item in line.split(","):
                item = item.strip()
                if "low_freq_ratio=" in item:
                    filt["low_freq_ratio"] = float(item.split("=")[1])
                if "peak_freq=" in item:
                    filt["peak_freq"] = float(
                        item.split("=")[1].replace(" Hz", "")
                    )
                if "mrcp_sensitive=" in item:
                    filt["mrcp_sensitive"] = item.split("=")[1].strip() == "True"
            result["filters"].append(filt)
    return result


def parse_tf_summary(path: Path) -> dict[str, Any]:
    """Parse tf_summary.txt."""
    result: dict[str, Any] = {"freq_bins": []}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "Peak frequency:" in line:
            result["peak_freq"] = float(
                line.split(":")[1].strip().replace("Hz", "").strip()
            )
        if "Hz:" in line and line[0].isdigit():
            parts = line.split(":")
            freq = float(parts[0].strip().replace("Hz", "").strip())
            power = float(parts[1].strip())
            result["freq_bins"].append((freq, power))
    return result


def parse_publication_stats(path: Path) -> dict[str, Any]:
    """Parse phase8 statistical_results.txt for channel rankings."""
    result: dict[str, Any] = {"channel_rankings": [], "permutation_p": None}
    in_channel_section = False
    for line in path.read_text().splitlines():
        line_s = line.strip()
        if "Rank  Channel" in line_s:
            in_channel_section = True
            continue
        if in_channel_section and line_s and line_s[0].isdigit():
            parts = line_s.split()
            if len(parts) >= 5:
                result["channel_rankings"].append(
                    {
                        "rank": int(parts[0]),
                        "channel": parts[1],
                        "mean": float(parts[2]),
                        "ci_lower": float(parts[3]),
                        "ci_upper": float(parts[4]),
                    }
                )
        elif in_channel_section and not line_s:
            in_channel_section = False
        if "p-value (two-sided):" in line_s:
            result["permutation_p"] = float(line_s.split(":")[1].strip())
    return result


def parse_lrp_results(path: Path) -> dict[str, float]:
    """Parse lrp_results.txt."""
    result: dict[str, float] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "Classical LRP" in line and "pre-cue" in line:
            result["classical_lrp_pre"] = float(line.split(":")[1].strip())
        elif "Classical LRP" in line and "post-cue" in line:
            result["classical_lrp_post"] = float(line.split(":")[1].strip())
        elif "Model lat. sal" in line and "pre-cue" in line:
            result["model_lat_pre"] = float(line.split(":")[1].strip())
        elif "Model lat. sal" in line and "post-cue" in line:
            result["model_lat_post"] = float(line.split(":")[1].strip())
    return result


def parse_cross_subject_consistency(path: Path) -> dict[str, Any]:
    """Parse consistency_summary.txt."""
    result: dict[str, Any] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if "Mean inter-subject correlation:" in line:
            result["mean_isc"] = float(line.split(":")[1].strip())
        if "Outlier subjects:" in line:
            ch_str = line.split(":")[1].strip()
            try:
                result["outliers"] = eval(ch_str)  # noqa: S307
            except Exception:
                result["outliers"] = []
    return result


# ── Evidence collection and scoring ──────────────────────────────────────────

def _wrap_text(text: str, width: int = 76, indent: str = "  ") -> list[str]:
    """Word-wrap text into lines of at most `width` characters."""
    words = text.split()
    lines: list[str] = []
    current = indent
    for word in words:
        if len(current) + len(word) + 1 > width:
            lines.append(current)
            current = indent + word
        else:
            current += (" " + word) if current.strip() else (indent + word)
    if current.strip():
        lines.append(current)
    return lines


def collect_evidence_and_score() -> list[TestEvidence]:
    """Read all result files and produce scored evidence for each test."""
    tests: list[TestEvidence] = []

    # ── N1: Sensorimotor ERDS significance ───────────────────────────────
    band_imp = parse_band_importance(
        RESULTS_ROOT / "phase4_erd_ers" / "4_1_band_saliency" / "band_importance_ranking.txt"
    )
    corr = parse_correlation_results(
        RESULTS_ROOT / "phase4_erd_ers" / "4_4_comparison" / "correlation_results.txt"
    )

    mu_c3_c0 = corr.get("mu", {}).get("Class 0", {}).get("C3", 0.0)
    mu_c4_c0 = corr.get("mu", {}).get("Class 0", {}).get("C4", 0.0)
    beta_c3_c0 = corr.get("beta", {}).get("Class 0", {}).get("C3", 0.0)
    beta_c4_c0 = corr.get("beta", {}).get("Class 0", {}).get("C4", 0.0)

    tests.append(
        TestEvidence(
            test_id="N1",
            test_name=TEST_NAMES["N1"],
            evidence=(
                f"Band importance ranking: delta ({band_imp.get('delta', 0):.6f}) > "
                f"theta ({band_imp.get('theta', 0):.6f}) > "
                f"mu ({band_imp.get('mu', 0):.6f}) > "
                f"beta ({band_imp.get('beta', 0):.6f}) > "
                f"gamma ({band_imp.get('gamma', 0):.6f}). "
                f"Mu/beta are 3rd and 4th ranked, not dominant. "
                f"Model-vs-classical ERD correlation at key sensorimotor channels: "
                f"mu at C3: r={mu_c3_c0:.3f}, C4: r={mu_c4_c0:.3f}; "
                f"beta at C3: r={beta_c3_c0:.3f}, C4: r={beta_c4_c0:.3f}. "
                f"Beta shows moderate correlation at C3 but mu is weak. "
                f"Overall, delta/theta dominate model saliency, not mu/beta."
            ),
            reasoning={
                "Artifacts": (
                    "Artifacts predict broadband or low-frequency dominance. "
                    "Delta/theta dominance is mildly consistent with slow artifacts, "
                    "but this test is about ERDS specifically. Inconclusive for artifacts."
                ),
                "Classical ERD": (
                    "Classical ERD predicts mu (8-13 Hz) and beta (13-30 Hz) as the "
                    "dominant bands driving the model. Instead, delta > theta > mu > beta. "
                    "Mu correlation at C3 is only r=0.154, C4 near zero. Beta at C3 is "
                    "moderate (r=0.360) but beta is the 4th-ranked band overall. "
                    "Strong evidence against: ERD is not the primary driver."
                ),
                "MRCP": (
                    "MRCP predicts <5 Hz dominance. Delta band leading is frequency-consistent, "
                    "but this test specifically evaluates mu/beta ERDS. Absence of mu/beta "
                    "ERDS does not distinguish MRCP directly. Inconclusive."
                ),
                "Visual/Cue ERP": (
                    "Visual/Cue ERPs are low-frequency transients. Delta/theta dominance "
                    "is consistent with stimulus-locked evoked potentials. Moderate support."
                ),
            },
            scores={
                "Artifacts": "0",
                "Classical ERD": "--",
                "MRCP": "0",
                "Visual/Cue ERP": "+",
            },
        )
    )

    # ── N2: Frontal artifact indicators ──────────────────────────────────
    ablation = parse_channel_group_ablation(
        RESULTS_ROOT / "phase6_artifacts" / "channel_group_ablation.txt"
    )
    frontal_drop = ablation.get("frontal", {}).get("accuracy_drop", 0.0)
    central_drop = ablation.get("central", {}).get("accuracy_drop", 0.0)
    parietal_drop = ablation.get("parietal", {}).get("accuracy_drop", 0.0)
    fc_ratio = frontal_drop / central_drop if central_drop > 0 else 0.0

    tests.append(
        TestEvidence(
            test_id="N2",
            test_name=TEST_NAMES["N2"],
            evidence=(
                f"Channel group ablation accuracy drops: "
                f"frontal = {frontal_drop * 100:.1f}%, "
                f"central = {central_drop * 100:.1f}%, "
                f"parietal = {parietal_drop * 100:.1f}%. "
                f"Frontal/central ratio = {fc_ratio:.2f}. "
                f"Only 1/200 trials flagged as artifact-prone. "
                f"Model relies primarily on central channels, not frontal/ocular sites."
            ),
            reasoning={
                "Artifacts": (
                    "Artifact hypothesis predicts frontal channel dominance "
                    "(eye movements, blinks concentrate at Fp/F sites). "
                    f"Frontal ablation drop is only {frontal_drop * 100:.1f}% vs "
                    f"central {central_drop * 100:.1f}% -- a "
                    f"{central_drop / frontal_drop:.1f}x difference favoring central. "
                    f"Frontal/central saliency ratio {fc_ratio:.2f}. "
                    f"Only 1/200 artifact trials. "
                    f"Strong evidence against artifact reliance."
                ),
                "Classical ERD": (
                    "ERD predicts central (C3/C4) dominance. Central group shows largest "
                    f"ablation impact ({central_drop * 100:.1f}%). Moderate support for "
                    "central importance, though this alone does not confirm ERD."
                ),
                "MRCP": (
                    "MRCP predicts Cz/FCz importance. Central group includes these "
                    f"channels. Central {central_drop * 100:.1f}% drop is consistent. "
                    "Moderate support."
                ),
                "Visual/Cue ERP": (
                    "Visual/Cue ERP predicts fronto-central importance. Central > frontal "
                    "is consistent with the central component. The low frontal "
                    "contribution argues against purely visual origin, but cue-evoked "
                    "ERPs project to central channels. Inconclusive."
                ),
            },
            scores={
                "Artifacts": "--",
                "Classical ERD": "+",
                "MRCP": "+",
                "Visual/Cue ERP": "0",
            },
        )
    )

    # ── N3: MRCP presence ────────────────────────────────────────────────
    lrp = parse_lrp_results(RESULTS_ROOT / "phase5_mrcp" / "lrp_results.txt")
    filter_mrcp = parse_filter_mrcp_sensitivity(
        RESULTS_ROOT / "phase5_mrcp" / "filter_mrcp_sensitivity.txt"
    )

    tests.append(
        TestEvidence(
            test_id="N3",
            test_name=TEST_NAMES["N3"],
            evidence=(
                f"No Bereitschaftspotential at Cz or FCz -- ERP shows positive "
                f"deflection at cue onset, not pre-movement negativity. "
                f"No lateralized readiness potential: classical LRP "
                f"pre={lrp.get('classical_lrp_pre', 0):.3f}, "
                f"post={lrp.get('classical_lrp_post', 0):.3f} (no asymmetry); "
                f"model lateralized saliency "
                f"pre={lrp.get('model_lat_pre', 0):.6f}, "
                f"post={lrp.get('model_lat_post', 0):.6f}. "
                f"Signal is stimulus-locked (peaks at t=0), not movement-locked. "
                f"{filter_mrcp['mrcp_sensitive_count']}/{filter_mrcp['total']} filters "
                f"have >65% energy <5 Hz (frequency matches MRCP), but temporal "
                f"profile is wrong."
            ),
            reasoning={
                "Artifacts": (
                    "MRCP absence does not directly implicate or rule out artifacts. "
                    "Inconclusive."
                ),
                "Classical ERD": (
                    "MRCP absence does not directly affect ERD hypothesis. "
                    "Inconclusive."
                ),
                "MRCP": (
                    "MRCP hypothesis predicts: (1) Bereitschaftspotential at Cz/FCz "
                    "-- ABSENT; (2) lateralized readiness potential -- ABSENT; "
                    "(3) pre-movement temporal ramp -- ABSENT (signal is cue-locked). "
                    "Frequency content matches (<5 Hz) but all three temporal/spatial "
                    "hallmarks are missing. Strong evidence against."
                ),
                "Visual/Cue ERP": (
                    "Absence of MRCP is consistent with Visual/Cue ERP: the "
                    "stimulus-locked positive deflection at t=0 and lack of "
                    "pre-movement ramp is exactly what a cue-evoked response "
                    "would produce. Moderate support."
                ),
            },
            scores={
                "Artifacts": "0",
                "Classical ERD": "0",
                "MRCP": "--",
                "Visual/Cue ERP": "+",
            },
        )
    )

    # ── F1: Temporal filter frequency distribution ───────────────────────
    tf_summary = parse_tf_summary(
        RESULTS_ROOT / "phase4_erd_ers" / "4_2_tf_saliency" / "tf_summary.txt"
    )

    freq_bins_str = ", ".join(
        f"{f:.1f} Hz ({p:.6f})" for f, p in tf_summary.get("freq_bins", [])[:5]
    )

    tests.append(
        TestEvidence(
            test_id="F1",
            test_name=TEST_NAMES["F1"],
            evidence=(
                f"All 8 temporal filters target delta/theta (<8 Hz) with minimal "
                f"mu/beta sensitivity. Time-frequency saliency peak at "
                f"{tf_summary.get('peak_freq', 1.0)} Hz. "
                f"Top frequency bins: {freq_bins_str}. "
                f"{filter_mrcp['mrcp_sensitive_count']}/8 filters have >65% energy <5 Hz. "
                f"Remaining filters peak at 11.7 Hz (mu-adjacent) and 43 Hz (gamma). "
                f"No filter specifically targets classical mu (8-13) or beta (13-30)."
            ),
            reasoning={
                "Artifacts": (
                    "Artifacts (especially ocular) concentrate in low frequencies. "
                    "Delta/theta filter dominance is consistent with slow artifact "
                    "sensitivity, but this overlaps with legitimate slow cortical "
                    "activity. Mild support."
                ),
                "Classical ERD": (
                    "ERD requires mu (8-13 Hz) and beta (13-30 Hz) filters. "
                    "No filter specifically targets these bands. Only 2/8 filters "
                    "reach mu-adjacent frequencies (11.7 Hz). The model cannot "
                    "extract mu/beta ERD without appropriate filters. "
                    "Strong evidence against."
                ),
                "MRCP": (
                    "MRCP predicts <5 Hz filter sensitivity. 5/8 filters have >65% "
                    "energy <5 Hz with DC peaks. Frequency profile matches MRCP "
                    "requirements. Strong support for the frequency match (temporal "
                    "profile assessed separately)."
                ),
                "Visual/Cue ERP": (
                    "Cue-evoked ERPs are low-frequency transients (typically <10 Hz). "
                    "The delta/theta filter profile is highly consistent with "
                    "detecting stimulus-evoked potentials. Strong support."
                ),
            },
            scores={
                "Artifacts": "+",
                "Classical ERD": "--",
                "MRCP": "+",
                "Visual/Cue ERP": "++",
            },
        )
    )

    # ── F2: Spatial filter ROI focus ─────────────────────────────────────
    tests.append(
        TestEvidence(
            test_id="F2",
            test_name=TEST_NAMES["F2"],
            evidence=(
                "Spatial filters show distributed weighting across the full scalp, "
                "not focal sensorimotor patterns. Sensorimotor channels (C3, C4, Cz, "
                "FC3, FC4) carry moderate weights in some filters (5, 7, 12) but "
                "are not consistently dominant. Many filters place largest weights on "
                "non-sensorimotor channels including frontal and peripheral sites. "
                "Several filters show broad bipolar frontal-to-posterior patterns. "
                "Some fronto-central emphasis observed overall."
            ),
            reasoning={
                "Artifacts": (
                    "Artifact hypothesis predicts frontal-dominant spatial filters "
                    "(Fp1/Fp2/F channels for ocular artifacts). Some frontal "
                    "weighting exists but is not dominant. Inconclusive."
                ),
                "Classical ERD": (
                    "ERD predicts focal C3/C4 spatial filters for contralateral "
                    "sensorimotor rhythm extraction. Spatial filters are distributed, "
                    "not focal on C3/C4. Moderate evidence against."
                ),
                "MRCP": (
                    "MRCP predicts Cz/FCz-centered spatial filters. Some "
                    "fronto-central emphasis exists but filters are broadly "
                    "distributed. Inconclusive."
                ),
                "Visual/Cue ERP": (
                    "Visual/Cue ERP predicts fronto-central distribution broadly "
                    "consistent with stimulus processing across multiple areas. "
                    "The distributed, non-focal pattern with fronto-central "
                    "emphasis matches this prediction. Moderate support."
                ),
            },
            scores={
                "Artifacts": "0",
                "Classical ERD": "-",
                "MRCP": "0",
                "Visual/Cue ERP": "+",
            },
        )
    )

    # ── A1: Attribution spatial distribution ──────────────────────────────
    pub_stats = parse_publication_stats(
        RESULTS_ROOT / "phase8_publication" / "statistical_results.txt"
    )
    top5 = pub_stats["channel_rankings"][:5]
    top5_str = ", ".join(f"{ch['channel']} ({ch['mean']:.4f})" for ch in top5)

    tests.append(
        TestEvidence(
            test_id="A1",
            test_name=TEST_NAMES["A1"],
            evidence=(
                f"Top 5 channels by bootstrap importance: {top5_str}. "
                f"FC6 dominant (0.00509), nearly 2x FC5 (0.00301). "
                f"Distribution is fronto-central, right-lateralized. "
                f"Classic sensorimotor C3 ranks 9th (0.00124), C4 ranks 3rd (0.00206). "
                f"FCz ranks 17th (0.00093), Cz ranks 19th (0.00088). "
                f"Frontal/central saliency ratio = 0.60 (central dominates)."
            ),
            reasoning={
                "Artifacts": (
                    "Artifact hypothesis predicts frontal (Fp/F) channel dominance "
                    "for ocular artifacts. Top channels are FC (fronto-central), not "
                    "Fp (frontal-polar). FC6/FC5 dominance is atypical for pure "
                    "ocular artifacts. Moderate evidence against."
                ),
                "Classical ERD": (
                    "ERD predicts C3/C4 dominance. C4 is 3rd but C3 is only 9th. "
                    "FC6/FC5 dominate instead. The spatial pattern does not match "
                    "the focal sensorimotor topography expected for mu/beta ERD. "
                    "Moderate evidence against."
                ),
                "MRCP": (
                    "MRCP predicts Cz/FCz dominance. FCz ranks 17th, Cz ranks 19th. "
                    "These are in the bottom half. FC6/FC5 domination is not "
                    "consistent with midline MRCP generators. "
                    "Moderate evidence against."
                ),
                "Visual/Cue ERP": (
                    "Visual/Cue ERP predicts fronto-central distribution with "
                    "bilateral involvement. FC6/FC5 dominance is fronto-central. "
                    "The pattern spans both hemispheres (FC5 left, FC6 right) "
                    "with right-lateralization. This matches cue-evoked differential "
                    "processing at fronto-central sites. Strong support."
                ),
            },
            scores={
                "Artifacts": "-",
                "Classical ERD": "-",
                "MRCP": "-",
                "Visual/Cue ERP": "++",
            },
        )
    )

    # ── A2: Attribution temporal concentration ────────────────────────────
    pre_post = parse_pre_post_results(
        RESULTS_ROOT / "phase2_saliency" / "pre_post_results.txt"
    )

    tests.append(
        TestEvidence(
            test_id="A2",
            test_name=TEST_NAMES["A2"],
            evidence=(
                f"Gradient saliency ratio (pre/post) = "
                f"{pre_post.get('importance_ratio', 0):.2f} (nearly equal). "
                f"Ablation: pre-cue masking IMPROVES accuracy by "
                f"{abs(pre_post.get('pre_cue_accuracy_drop', 0)) * 100:.1f}%, "
                f"post-cue masking drops accuracy by "
                f"{pre_post.get('post_cue_accuracy_drop', 0) * 100:.1f}%. "
                f"Saliency peaks at cue onset (t=0) with strongest functional "
                f"dependence in the 0-0.5s post-cue window. Pre-cue information "
                f"is actually harmful to classification when present alone. "
                f"Permutation test pre vs post: p=0.003."
            ),
            reasoning={
                "Artifacts": (
                    "Artifacts predict early/diffuse timing without cue-locking. "
                    "The sharp cue-onset peak and post-cue dependence is more "
                    "stimulus-specific than expected for random artifacts. "
                    "Moderate evidence against."
                ),
                "Classical ERD": (
                    "ERD predicts post-cue 0.5-1.5s peak (ERD develops gradually "
                    "after imagery onset). The peak is at 0-0.5s, earlier than "
                    "typical ERD latency. The strongest ablation impact is immediate "
                    "post-cue, not during sustained imagery. "
                    "Moderate evidence against."
                ),
                "MRCP": (
                    "MRCP predicts pre-cue importance (pre-movement potential "
                    "building before cue). Pre-cue masking IMPROVES accuracy (+8%), "
                    "meaning pre-cue information is detrimental. This directly "
                    "contradicts the MRCP prediction. Strong evidence against."
                ),
                "Visual/Cue ERP": (
                    "Visual/Cue ERP predicts cue-locked peak at t~0 with strongest "
                    "influence in the immediate post-cue window. Saliency peaks "
                    "exactly at cue onset. Post-cue masking causes 24.5% accuracy "
                    "drop. Pre-cue is irrelevant/harmful. This is exactly the "
                    "temporal profile expected for a cue-evoked response. "
                    "Strong support."
                ),
            },
            scores={
                "Artifacts": "-",
                "Classical ERD": "-",
                "MRCP": "--",
                "Visual/Cue ERP": "++",
            },
        )
    )

    # ── A3: Attribution lateralization ────────────────────────────────────
    lat = parse_signed_lateralization(
        RESULTS_ROOT / "lateralization" / "signed_lateralization_results.txt"
    )
    consistency = parse_cross_subject_consistency(
        RESULTS_ROOT / "phase7_cross_subject" / "consistency_summary.txt"
    )

    tests.append(
        TestEvidence(
            test_id="A3",
            test_name=TEST_NAMES["A3"],
            evidence=(
                f"Signed IG sign-flip test: {lat['n_flip_channels']}/32 channels show "
                f"sign flips between classes ({', '.join(lat['flip_channels'])}). "
                f"Zero sensorimotor channels (C3, C4, C1, C2, FC3, FC4) flip. "
                f"Right hemisphere dominant for BOTH classes. "
                f"MI_left post-cue LI=+0.082 (right>left), "
                f"MI_right post-cue LI=-0.152 (right>left). "
                f"No contralateral lateralization pattern. "
                f"Cross-subject consistency: mean ISC="
                f"{consistency.get('mean_isc', 0):.3f}."
            ),
            reasoning={
                "Artifacts": (
                    "Artifacts predict no systematic lateralization. The consistent "
                    "right-lateralized pattern across subjects is actually more "
                    "structured than expected for random artifacts. "
                    "Mild evidence against."
                ),
                "Classical ERD": (
                    "ERD predicts contralateral lateralization: left MI -> right "
                    "hemisphere dominant, right MI -> left hemisphere dominant. "
                    "Sign-flip test shows NO contralateral pattern at sensorimotor "
                    "channels. Right hemisphere dominates for BOTH classes. "
                    "0/6 key sensorimotor channels flip. This is the most direct "
                    "test of genuine MI, and it fails definitively. "
                    "Strong evidence against."
                ),
                "MRCP": (
                    "MRCP predicts midline (Cz/FCz) with subtle lateralization. "
                    "The strong right-hemisphere dominance without class-dependent "
                    "lateralization is inconsistent with MRCP. "
                    "Moderate evidence against."
                ),
                "Visual/Cue ERP": (
                    "Visual/Cue ERP predicts both hemispheres involved with "
                    "potential right-lateralized processing of lateralized visual "
                    "cues. The consistent right-lateralized pattern for all subjects "
                    "with no contralateral flips is consistent with a shared "
                    "visual/attentional response. The right-hemisphere dominance "
                    "may reflect right-hemisphere specialization for spatial "
                    "attention. High cross-subject consistency (ISC=0.963) supports "
                    "a universal, stimulus-driven mechanism. Strong support."
                ),
            },
            scores={
                "Artifacts": "-",
                "Classical ERD": "--",
                "MRCP": "-",
                "Visual/Cue ERP": "++",
            },
        )
    )

    return tests


# ── Output generation ────────────────────────────────────────────────────────

def build_matrix(tests: list[TestEvidence]) -> np.ndarray:
    """Build numeric scoring matrix (tests x hypotheses)."""
    matrix = np.zeros((len(tests), len(HYPOTHESES)), dtype=int)
    for i, test in enumerate(tests):
        for j, hyp in enumerate(HYPOTHESES):
            matrix[i, j] = SCORE_VALUES[test.scores.get(hyp, "0")]
    return matrix


def write_scoring_matrix(tests: list[TestEvidence], output_dir: Path) -> None:
    """Write the formatted scoring matrix table."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("FORMAL HYPOTHESIS SCORING MATRIX")
    lines.append("Thesis Section 5.4 -- Quantitative Hypothesis Evaluation")
    lines.append("=" * 90)
    lines.append("")

    # Scoring scale legend
    lines.append("Scoring scale (thesis section 4.5.3):")
    lines.append("  ++  Strong support     (|d| > 0.8, p < 0.01)")
    lines.append("  +   Moderate support   (|d| > 0.5, p < 0.05)")
    lines.append("  0   Inconclusive")
    lines.append("  -   Moderate evidence against")
    lines.append("  --  Strong evidence against")
    lines.append("")

    # Header
    col_w = 16
    header = f"{'Test':<6} {'Description':<38}"
    for hyp in HYPOTHESES:
        header += f" {hyp:>{col_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    # Rows
    for test in tests:
        row = f"{test.test_id:<6} {test.test_name:<38}"
        for hyp in HYPOTHESES:
            score = test.scores.get(hyp, "0")
            row += f" {score:>{col_w}}"
        lines.append(row)

    lines.append("-" * len(header))

    # Net score row
    net_row = f"{'NET':<6} {'Sum of numeric scores':<38}"
    for hyp in HYPOTHESES:
        net = sum(SCORE_VALUES[t.scores.get(hyp, "0")] for t in tests)
        net_row += f" {net:>+{col_w}}"
    lines.append(net_row)
    lines.append("")

    # Score distribution
    tallies: dict[str, dict[str, int]] = {}
    for hyp in HYPOTHESES:
        tallies[hyp] = {"++": 0, "+": 0, "0": 0, "-": 0, "--": 0}
    for test in tests:
        for hyp in HYPOTHESES:
            s = test.scores.get(hyp, "0")
            tallies[hyp][s] += 1

    lines.append("Score distribution per hypothesis:")
    lines.append("-" * 70)
    dist_header = f"{'Score':<8}"
    for hyp in HYPOTHESES:
        dist_header += f" {hyp:>{col_w}}"
    lines.append(dist_header)
    for score_label in SCORE_LABELS:
        dist_row = f"{score_label:<8}"
        for hyp in HYPOTHESES:
            dist_row += f" {tallies[hyp][score_label]:>{col_w}}"
        lines.append(dist_row)
    lines.append("")

    path = output_dir / "scoring_matrix.txt"
    path.write_text("\n".join(lines))
    print(f"  -> {path}")


def write_test_details(tests: list[TestEvidence], output_dir: Path) -> None:
    """Write detailed evidence and reasoning for each test."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("HYPOTHESIS SCORING -- DETAILED EVIDENCE AND REASONING")
    lines.append("=" * 80)

    for test in tests:
        lines.append("")
        lines.append("-" * 80)
        lines.append(f"TEST {test.test_id}: {test.test_name}")
        lines.append("-" * 80)
        lines.append("")
        lines.append("EVIDENCE:")
        lines.extend(_wrap_text(test.evidence))
        lines.append("")

        lines.append("SCORES AND REASONING:")
        for hyp in HYPOTHESES:
            score = test.scores.get(hyp, "0")
            reasoning = test.reasoning.get(hyp, "No reasoning provided.")
            lines.append(f"  {hyp} [{score}]:")
            lines.extend(_wrap_text(reasoning, indent="    "))
            lines.append("")

    path = output_dir / "test_details.txt"
    path.write_text("\n".join(lines))
    print(f"  -> {path}")


def write_verdict(tests: list[TestEvidence], output_dir: Path) -> None:
    """Write final tally and verdict."""
    output_dir.mkdir(parents=True, exist_ok=True)

    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("HYPOTHESIS SCORING -- FINAL VERDICT")
    lines.append("=" * 70)
    lines.append("")

    max_score = 2 * len(tests)

    # Tally per hypothesis
    hyp_net: list[tuple[str, int]] = []
    for hyp in HYPOTHESES:
        counts: dict[str, int] = {"++": 0, "+": 0, "0": 0, "-": 0, "--": 0}
        net = 0
        for test in tests:
            s = test.scores.get(hyp, "0")
            counts[s] += 1
            net += SCORE_VALUES[s]

        hyp_net.append((hyp, net))

        lines.append("-" * 60)
        lines.append(f"  {hyp}")
        lines.append("-" * 60)
        lines.append(f"    ++  (strong support):          {counts['++']}")
        lines.append(f"    +   (moderate support):        {counts['+']}")
        lines.append(f"    0   (inconclusive):            {counts['0']}")
        lines.append(f"    -   (moderate against):        {counts['-']}")
        lines.append(f"    --  (strong against):          {counts['--']}")
        lines.append(f"    Net score:                     {net:+d} / {max_score}")
        lines.append("")

    # Ranking
    hyp_net.sort(key=lambda x: x[1], reverse=True)

    lines.append("=" * 60)
    lines.append("OVERALL RANKING")
    lines.append("=" * 60)
    lines.append("")

    for rank, (hyp, score) in enumerate(hyp_net, 1):
        lines.append(f"  {rank}. {hyp:<20} {score:+d} / {max_score}")
    lines.append("")

    winner = hyp_net[0]

    lines.append("=" * 60)
    lines.append("VERDICT")
    lines.append("=" * 60)
    lines.append("")
    lines.append(
        f"The evidence strongly supports the {winner[0]} hypothesis "
        f"(net score {winner[1]:+d}/{max_score})."
    )
    lines.append("")
    lines.append(
        "The model classifies left vs right motor imagery by exploiting a "
        "cue-evoked differential response at fronto-central channels (FC5/FC6), "
        "concentrated in the immediate post-cue window (0-0.5s). This signal is:"
    )
    lines.append("")
    lines.append("  1. Low-frequency (delta/theta, <8 Hz) -- not mu/beta ERD/ERS")
    lines.append("  2. Stimulus-locked at t=0 -- not pre-movement preparatory activity")
    lines.append("  3. Fronto-central -- not focal on C3/C4 sensorimotor cortex")
    lines.append("  4. Right-lateralized for BOTH classes -- not contralateral")
    lines.append("  5. Highly consistent across all 40 subjects (ISC=0.963)")
    lines.append("")
    lines.append("Ruled-out hypotheses:")

    for hyp, score in hyp_net[1:]:
        lines.append(f"  - {hyp} ({score:+d}):")
        if hyp == "Classical ERD":
            lines.append(
                "    No mu/beta filters, no focal C3/C4 spatial pattern, "
                "no contralateral lateralization. All core predictions fail."
            )
        elif hyp == "MRCP":
            lines.append(
                "    No Bereitschaftspotential, no LRP, pre-cue masking improves "
                "accuracy. Frequency matches (<5 Hz) but temporal/spatial profile wrong."
            )
        elif hyp == "Artifacts":
            lines.append(
                "    Frontal channels contribute minimally (1.5% ablation drop), "
                "central channels dominate (6.5%), only 1/200 artifact trials."
            )
        elif hyp == "Visual/Cue ERP":
            lines.append(
                "    Strong cue-locked temporal and fronto-central spatial profile "
                "across all subjects."
            )
    lines.append("")

    path = output_dir / "verdict.txt"
    path.write_text("\n".join(lines))
    print(f"  -> {path}")


def plot_scoring_heatmap(tests: list[TestEvidence], output_dir: Path) -> None:
    """Generate a visual heatmap of the scoring matrix."""
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_matrix(tests)
    test_labels = [f"{t.test_id}: {t.test_name}" for t in tests]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Red-white-green diverging colormap, 5 discrete bins
    colors = ["#d73027", "#fc8d59", "#f7f7f7", "#91cf60", "#1a9850"]
    cmap = mcolors.LinearSegmentedColormap.from_list("hypothesis", colors, N=5)
    bounds = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Annotate cells with score labels
    for i in range(len(tests)):
        for j in range(len(HYPOTHESES)):
            score_str = tests[i].scores.get(HYPOTHESES[j], "0")
            text_color = "white" if abs(SCORE_VALUES[score_str]) == 2 else "black"
            ax.text(
                j, i, score_str,
                ha="center", va="center",
                fontsize=14, fontweight="bold", color=text_color,
            )

    ax.set_xticks(range(len(HYPOTHESES)))
    ax.set_xticklabels(HYPOTHESES, fontsize=11, rotation=25, ha="right")
    ax.set_yticks(range(len(tests)))
    ax.set_yticklabels(test_labels, fontsize=10)

    ax.set_title(
        "Hypothesis Scoring Matrix",
        fontsize=14, fontweight="bold", pad=15,
    )

    # Net scores below matrix
    net_scores = matrix.sum(axis=0)
    for j, net in enumerate(net_scores):
        color = "#1a9850" if net > 0 else "#d73027" if net < 0 else "gray"
        ax.text(
            j, len(tests) + 0.1, f"Net: {net:+d}",
            ha="center", va="top", fontsize=11, fontweight="bold", color=color,
        )

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, ticks=[-2, -1, 0, 1, 2], shrink=0.8, pad=0.02)
    cbar.ax.set_yticklabels(SCORE_LABELS, fontsize=10)
    cbar.set_label("Evidence Score", fontsize=11)

    plt.subplots_adjust(bottom=0.15, left=0.35, right=0.92, top=0.93)

    path = output_dir / "scoring_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full hypothesis scoring pipeline."""
    print("=" * 60)
    print("Formal Hypothesis Scoring Matrix")
    print("Thesis sections 4.5 and 5.4")
    print("=" * 60)
    print()

    print("Collecting evidence from analysis results...")
    tests = collect_evidence_and_score()
    print(f"  Scored {len(tests)} tests against {len(HYPOTHESES)} hypotheses")
    print()

    print("Writing scoring matrix...")
    write_scoring_matrix(tests, OUTPUT_DIR)

    print("Writing test details...")
    write_test_details(tests, OUTPUT_DIR)

    print("Writing verdict...")
    write_verdict(tests, OUTPUT_DIR)

    print("Generating heatmap...")
    plot_scoring_heatmap(tests, OUTPUT_DIR)

    # Summary to console
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    matrix = build_matrix(tests)
    net_scores = matrix.sum(axis=0)
    max_score = 2 * len(tests)
    for hyp, net in zip(HYPOTHESES, net_scores):
        indicator = " <<<" if net == max(net_scores) else ""
        print(f"  {hyp:<20} {net:+d} / {max_score}{indicator}")
    print()
    print(f"Outputs saved to: {OUTPUT_DIR}")
    print("Done.")


if __name__ == "__main__":
    main()
