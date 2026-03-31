#!/usr/bin/env python3
"""
# test_onnx_inference.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path

import numpy as np

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
TESTS_DIR = SCRIPT_PATH.parent
TOOL_DIR = TESTS_DIR.parent
REPO_ROOT = TOOL_DIR.parent.parent

# Add sub-directories directly to path to avoid triggering
# __init__.py import chains which pull in orchestral-dependent tools
DATA_ACCESS_DIR = TOOL_DIR / "data_access"
TRAINING_DIR = TOOL_DIR / "training"
DEPLOYMENT_DIR = TOOL_DIR / "deployment"
sys.path.insert(0, str(DATA_ACCESS_DIR))
sys.path.insert(0, str(TRAINING_DIR))
sys.path.insert(0, str(DEPLOYMENT_DIR))

from histogram_utils import (
    histogram_to_record,
    write_histograms_jsonl,
    generate_synthetic_histograms,
    normalize_histogram,
)

OUTPUT_DIR = TESTS_DIR / "test_data" / "inference_output"


def _create_test_model_and_data():
    """Helper: train a small model and prepare test data."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None, None, None

    # Simple autoencoder
    input_dim = 64

    class SimpleAE(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 8),
                nn.ReLU(),
            )
            self.decoder = nn.Sequential(
                nn.Linear(8, 32),
                nn.ReLU(),
                nn.Linear(32, input_dim),
                nn.Sigmoid(),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))

    model = SimpleAE()
    model.eval()

    # Export to ONNX
    os.makedirs(str(OUTPUT_DIR), exist_ok=True)
    onnx_path = str(OUTPUT_DIR / "test_model.onnx")
    dummy = torch.randn(1, input_dim)
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
    )

    # Generate test data JSONL
    hists, labels = generate_synthetic_histograms(
        n_histograms=50, n_bins=input_dim,
        anomaly_fraction=0.2, seed=123
    )
    records = []
    for i, h in enumerate(hists):
        h_norm = normalize_histogram(h, "unit_area")
        rec = histogram_to_record(
            name=f"ECAL/test_hist_{i}",
            counts=h_norm,
            run=370000,
            lumi_section=i + 1,
            subsystem="ECAL",
        )
        records.append(rec)

    data_path = str(OUTPUT_DIR / "test_histograms.jsonl")
    write_histograms_jsonl(records, data_path)

    return onnx_path, data_path, labels


def test_onnx_inference():
    """Test ONNXInferenceTool end-to-end."""
    print(">> Testing ONNXInferenceTool...\n")

    try:
        import onnxruntime
    except ImportError:
        print("[⚠] onnxruntime not installed, skipping inference test\n")
        return True

    try:
        import torch
    except ImportError:
        print("[⚠] PyTorch not installed (needed to create test model), skipping\n")
        return True

    try:
        from onnx_inference import ONNXInferenceTool
    except ImportError:
        print("[⚠] orchestral not installed, skipping inference test\n")
        return True

    onnx_path, data_path, labels = _create_test_model_and_data()
    if onnx_path is None:
        print("[⚠] Could not create test model, skipping\n")
        return True

    scores_path = str(OUTPUT_DIR / "scores.jsonl")

    tool = ONNXInferenceTool(
        model_path=os.path.relpath(onnx_path, str(REPO_ROOT)),
        input_data_path=os.path.relpath(data_path, str(REPO_ROOT)),
        output_path=os.path.relpath(scores_path, str(REPO_ROOT)),
        normalization="unit_area",
        batch_size=16,
        base_directory=str(REPO_ROOT),
    )
    tool._setup()
    result_str = tool._run()
    result = json.loads(result_str)

    if result.get("status") != "ok":
        print(f"[✗] Inference failed: {result}\n")
        return False

    assert result["n_histograms"] == 50, f"Expected 50, got {result['n_histograms']}"
    assert result["mean_score"] >= 0, "Mean score should be non-negative"
    assert os.path.exists(scores_path), "Scores file should exist"

    # Verify output records
    with open(scores_path, "r") as f:
        score_records = [json.loads(line) for line in f]
    assert len(score_records) == 50
    assert all("anomaly_score" in r for r in score_records)
    assert all("histogram_name" in r for r in score_records)

    print(f"[✓] Inference test passed (mean_score={result['mean_score']:.6f}, "
          f"max_score={result['max_score']:.6f})\n")
    return True


def test_anomaly_scorer():
    """Test AnomalyScorerTool."""
    print(">> Testing AnomalyScorerTool...\n")

    try:
        from anomaly_scorer import AnomalyScorerTool
    except ImportError:
        print("[⚠] orchestral not installed, skipping scorer test\n")
        return True

    scores_path = str(OUTPUT_DIR / "scores.jsonl")
    if not os.path.exists(scores_path):
        print("[⚠] No scores file found, skipping scorer test\n")
        return True

    scored_path = str(OUTPUT_DIR / "scored.jsonl")

    tool = AnomalyScorerTool(
        scores_path=os.path.relpath(scores_path, str(REPO_ROOT)),
        output_path=os.path.relpath(scored_path, str(REPO_ROOT)),
        threshold_method="percentile",
        warning_threshold=90.0,
        bad_threshold=98.0,
        base_directory=str(REPO_ROOT),
    )
    tool._setup()
    result_str = tool._run()
    result = json.loads(result_str)

    if result.get("status") != "ok":
        print(f"[✗] Scorer failed: {result}\n")
        return False

    summary = result["summary"]
    assert summary["n_total"] == 50
    assert summary["n_good"] + summary["n_warning"] + summary["n_bad"] == 50
    assert summary["overall_verdict"] in ("GOOD", "WARNING", "BAD")

    print(f"[✓] Scorer test passed (verdict={summary['overall_verdict']}, "
          f"good={summary['n_good']}, warn={summary['n_warning']}, bad={summary['n_bad']})\n")
    return True


def test_alert_manager():
    """Test AlertManagerTool."""
    print(">> Testing AlertManagerTool...\n")

    try:
        from alert_manager import AlertManagerTool
    except ImportError:
        print("[⚠] orchestral not installed, skipping alert test\n")
        return True

    scored_path = str(OUTPUT_DIR / "scored.jsonl")
    if not os.path.exists(scored_path):
        print("[⚠] No scored file found, skipping alert test\n")
        return True

    alert_dir = str(OUTPUT_DIR / "alerts")

    tool = AlertManagerTool(
        scored_path=os.path.relpath(scored_path, str(REPO_ROOT)),
        output_dir=os.path.relpath(alert_dir, str(REPO_ROOT)),
        top_n=5,
        report_title="Test DQM Report",
        base_directory=str(REPO_ROOT),
    )
    tool._setup()
    result_str = tool._run()
    result = json.loads(result_str)

    if result.get("status") != "ok":
        print(f"[✗] Alert manager failed: {result}\n")
        return False

    # Verify report exists
    report_path = os.path.join(str(REPO_ROOT), result["report_path"])
    assert os.path.exists(report_path), "Markdown report should exist"

    # Verify alert payload exists
    alert_path = os.path.join(str(REPO_ROOT), result["alert_path"])
    assert os.path.exists(alert_path), "Alert payload should exist"

    # Verify report content
    with open(report_path, "r") as f:
        report_content = f.read()
    assert "Test DQM Report" in report_content
    assert "Summary" in report_content
    assert "Top" in report_content

    # Verify alert payload
    with open(alert_path, "r") as f:
        alert_data = json.load(f)
    assert "overall_verdict" in alert_data
    assert "top_anomalies" in alert_data
    assert len(alert_data["top_anomalies"]) <= 5

    print(f"[✓] Alert manager test passed (verdict={result['overall_verdict']}, "
          f"alerts={result['n_alerts']})\n")
    return True


def test_end_to_end_pipeline():
    """Test the full pipeline: data → inference → scoring → alerts."""
    print(">> Testing end-to-end pipeline...\n")

    try:
        import torch
        import onnxruntime
    except ImportError:
        print("[⚠] PyTorch or onnxruntime not installed, skipping E2E test\n")
        return True

    try:
        from autoencoder_trainer import AutoencoderTrainerTool
        from onnx_inference import ONNXInferenceTool
        from anomaly_scorer import AnomalyScorerTool
        from alert_manager import AlertManagerTool
    except ImportError:
        print("[⚠] orchestral not installed, skipping E2E test\n")
        return True

    e2e_dir = str(OUTPUT_DIR / "e2e")
    os.makedirs(e2e_dir, exist_ok=True)

    # Step 1: Train
    print("  [1/4] Training model...")
    trainer = AutoencoderTrainerTool(
        output_dir=os.path.relpath(os.path.join(e2e_dir, "model"), str(REPO_ROOT)),
        use_synthetic_data=True,
        n_synthetic=200,
        base_directory=str(REPO_ROOT),
    )
    trainer._setup()
    train_result = json.loads(trainer._run())
    assert train_result["status"] == "ok", f"Training failed: {train_result}"

    # Step 2: Generate test data (with anomalies) and save as JSONL
    hists, labels = generate_synthetic_histograms(
        n_histograms=50, n_bins=64, anomaly_fraction=0.2, seed=999
    )
    records = []
    for i, h in enumerate(hists):
        h_norm = normalize_histogram(h, "unit_area")
        rec = histogram_to_record(
            name=f"ECAL/e2e_hist_{i}",
            counts=h_norm,
            run=370001,
            lumi_section=i + 1,
            subsystem="ECAL",
        )
        records.append(rec)
    e2e_data_path = os.path.join(e2e_dir, "test_data.jsonl")
    write_histograms_jsonl(records, e2e_data_path)

    # Step 3: Inference
    print("  [2/4] Running inference...")
    inference = ONNXInferenceTool(
        model_path=train_result["model_path"],
        input_data_path=os.path.relpath(e2e_data_path, str(REPO_ROOT)),
        output_path=os.path.relpath(os.path.join(e2e_dir, "scores.jsonl"), str(REPO_ROOT)),
        base_directory=str(REPO_ROOT),
    )
    inference._setup()
    inf_result = json.loads(inference._run())
    assert inf_result["status"] == "ok", f"Inference failed: {inf_result}"

    # Step 4: Score
    print("  [3/4] Scoring anomalies...")
    scorer = AnomalyScorerTool(
        scores_path=inf_result["output_path"],
        output_path=os.path.relpath(os.path.join(e2e_dir, "scored.jsonl"), str(REPO_ROOT)),
        threshold_method="percentile",
        base_directory=str(REPO_ROOT),
    )
    scorer._setup()
    score_result = json.loads(scorer._run())
    assert score_result["status"] == "ok", f"Scoring failed: {score_result}"

    # Step 5: Alert
    print("  [4/4] Generating alerts...")
    alerter = AlertManagerTool(
        scored_path=score_result["output_path"],
        output_dir=os.path.relpath(os.path.join(e2e_dir, "report"), str(REPO_ROOT)),
        top_n=5,
        base_directory=str(REPO_ROOT),
    )
    alerter._setup()
    alert_result = json.loads(alerter._run())
    assert alert_result["status"] == "ok", f"Alerting failed: {alert_result}"

    print(f"\n[✓] End-to-end pipeline passed!")
    print(f"    Train loss: {train_result['metrics']['final_val_loss']:.6f}")
    print(f"    Inference: {inf_result['n_histograms']} histograms scored")
    print(f"    Verdict: {alert_result['overall_verdict']} ({alert_result['n_alerts']} alerts)\n")
    return True


def cleanup_test_files():
    """Remove test-generated files and directories."""
    print("\n>> Cleaning up test files...\n")
    if OUTPUT_DIR.exists():
        try:
            shutil.rmtree(OUTPUT_DIR)
            print(f"[✓] Removed: {OUTPUT_DIR.name}")
        except Exception as e:
            print(f"[⚠] Failed to remove {OUTPUT_DIR.name}: {e}")
    else:
        print("[i] No test files to clean up")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for DQM deployment tools")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep test-generated files")
    args = parser.parse_args()

    all_passed = True

    tests = [
        test_onnx_inference,
        test_anomaly_scorer,
        test_alert_manager,
        test_end_to_end_pipeline,
    ]

    for test_fn in tests:
        try:
            result = test_fn()
            if result is False:
                all_passed = False
        except Exception as e:
            print(f"[✗] {test_fn.__name__} failed: {e}\n")
            import traceback
            traceback.print_exc()
            all_passed = False

    if not args.keep_files:
        cleanup_test_files()
    else:
        print("\n[i] Keeping test files (--keep-files flag set)\n")

    if all_passed:
        print("[✓] All deployment tests passed!\n")
        sys.exit(0)
    else:
        print("[✗] Some tests failed!\n")
        sys.exit(1)
