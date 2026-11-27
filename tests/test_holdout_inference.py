from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from scripts.compute_holdout_inference import compute_holdout_inference

ROOT = Path(__file__).resolve().parents[1]


def test_archived_holdouts_and_weights_untouched_and_inference_consistent():
    holdout_path = ROOT / "results" / "neural_pinning_holdouts.json"
    weights_path = ROOT / "results" / "neural_feedback_weights.npz"
    inf_path = ROOT / "results" / "holdout_paired_inference.json"

    assert holdout_path.exists() and weights_path.exists() and inf_path.exists()
    holdouts = json.loads(holdout_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(weights_path.read_bytes()).hexdigest() == holdouts["protocol"]["weights_sha256"]

    inf = json.loads(inf_path.read_text(encoding="utf-8"))
    assert inf["metadata"]["source_holdout_sha256"] == hashlib.sha256(holdout_path.read_bytes()).hexdigest()

    c80 = {c["contrast_id"]: c for c in inf["paired_contrasts"]["additional_80"]["total_j"]}
    assert c80["gnn_minus_local_linear"]["win_fraction_a_lower"] == "80/80"
    assert c80["gnn_minus_local_linear"]["mean_diff"] == pytest.approx(-20.801434, abs=1e-5)
    assert c80["full_linear_minus_gnn"]["win_fraction_a_lower"] == "80/80"
    assert c80["full_linear_minus_gnn"]["mean_diff"] == pytest.approx(-5.063079, abs=1e-5)
    assert c80["gnn_minus_finite_lqr"]["mean_diff"] == pytest.approx(-69.288800, abs=1e-5)
    assert c80["gnn_minus_care"]["mean_diff"] == pytest.approx(-109.651939, abs=1e-5)
    assert c80["gnn_minus_self_only"]["mean_diff"] == pytest.approx(-522.979041, abs=1e-5)
    assert c80["proportional_minus_random_proportional"]["mean_diff"] == pytest.approx(-557.268665, abs=1e-5)
    for rec in c80.values():
        assert rec["wilcoxon_p_holm_exact"] < 1e-13

    recomputed = compute_holdout_inference(ROOT / "results")
    assert recomputed["paired_contrasts"] == inf["paired_contrasts"]


def test_readme_has_zero_math_leaks():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    # Remove fenced code blocks before checking inline math spans
    stripped = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Remove valid GitHub $`...`$ inline math spans
    without_valid = re.sub(r"\$`[^`\n]+`\$", "", stripped)
    assert "$" not in without_valid, f"Stray '$' found in README.md: {without_valid}"
