"""
# dqm_anomaly_demo.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
DQM Anomaly Detection Demo
===========================

End-to-end demonstration of the ML4DQM anomaly detection pipeline using
HEPTAPOD tools.  This demo uses synthetic histogram data to illustrate
the full workflow without requiring CMS-internal data or infrastructure.

Pipeline:
  1. Generate synthetic DQM histograms (normal + anomalous).
  2. Train a convolutional autoencoder on "good" reference data.
  3. Run inference on a new dataset containing anomalies.
  4. Score anomalies and apply thresholds.
  5. Generate a human-readable anomaly report.

Usage:
  python examples/workflows/dqm_anomaly_demo.py

This script can also be used as an Orchestral agent demo by uncommenting
the agent section at the bottom.
"""

import sys
import json
import os
from pathlib import Path

# Add repository root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Helper: pretty-print a JSON result from a tool
# ---------------------------------------------------------------------------

def _print_result(step_name: str, result_str: str):
    result = json.loads(result_str)
    status = result.get("status", "unknown")
    icon = "✓" if status == "ok" else "✗"
    print(f"\n  [{icon}] {step_name}")
    for k, v in result.items():
        if k == "status":
            continue
        if isinstance(v, dict):
            print(f"      {k}:")
            for kk, vv in v.items():
                print(f"        {kk}: {vv}")
        elif isinstance(v, list) and len(v) > 5:
            print(f"      {k}: [{v[0]}, {v[1]}, ..., {v[-1]}] ({len(v)} items)")
        else:
            print(f"      {k}: {v}")
    return result


# ---------------------------------------------------------------------------
# Main demo pipeline
# ---------------------------------------------------------------------------

def run_demo():
    """Run the complete DQM anomaly detection pipeline."""
    from tools.dqm.data_access.histogram_utils import (
        generate_synthetic_histograms,
        normalize_histogram,
        histogram_to_record,
        write_histograms_jsonl,
    )
    from tools.dqm.training.autoencoder_trainer import AutoencoderTrainerTool
    from tools.dqm.deployment.onnx_inference import ONNXInferenceTool
    from tools.dqm.deployment.anomaly_scorer import AnomalyScorerTool
    from tools.dqm.deployment.alert_manager import AlertManagerTool

    # Create workspace
    workspace = REPO_ROOT / "dqm_demo_workspace"
    workspace.mkdir(exist_ok=True)
    base_dir = str(workspace)

    print("=" * 70)
    print("  HEPTAPOD ML4DQM — Anomaly Detection Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Generate reference (training) data — "good" histograms
    # ------------------------------------------------------------------
    print("\n[Step 1/5] Generating reference training data...")
    train_hists, _ = generate_synthetic_histograms(
        n_histograms=500, n_bins=64,
        anomaly_fraction=0.0,  # All good for training
        seed=42,
    )
    train_records = []
    for i, h in enumerate(train_hists):
        h_norm = normalize_histogram(h, "unit_area")
        rec = histogram_to_record(
            name=f"ECAL/EBOccupancy/occupancy_{i}",
            counts=h_norm,
            run=369000 + (i // 50),
            lumi_section=(i % 50) + 1,
            subsystem="ECAL",
        )
        train_records.append(rec)

    train_path = os.path.join(base_dir, "reference_histograms.jsonl")
    write_histograms_jsonl(train_records, train_path)
    print(f"  Generated {len(train_records)} reference histograms → {train_path}")

    # ------------------------------------------------------------------
    # Step 2: Train autoencoder
    # ------------------------------------------------------------------
    print("\n[Step 2/5] Training convolutional autoencoder...")
    trainer = AutoencoderTrainerTool(
        input_data_path="reference_histograms.jsonl",
        output_dir="model",
        base_directory=base_dir,
    )
    trainer._setup()
    train_result = _print_result("Training complete", trainer._run())

    if train_result.get("status") != "ok":
        print("\n[✗] Training failed! Aborting demo.")
        return

    # ------------------------------------------------------------------
    # Step 3: Generate test data with anomalies
    # ------------------------------------------------------------------
    print("\n[Step 3/5] Generating test data (20% anomalies)...")
    test_hists, test_labels = generate_synthetic_histograms(
        n_histograms=100, n_bins=64,
        anomaly_fraction=0.2,
        seed=999,
    )
    test_records = []
    for i, h in enumerate(test_hists):
        h_norm = normalize_histogram(h, "unit_area")
        rec = histogram_to_record(
            name=f"ECAL/EBOccupancy/test_occupancy_{i}",
            counts=h_norm,
            run=370000,
            lumi_section=i + 1,
            subsystem="ECAL",
        )
        test_records.append(rec)

    test_path = os.path.join(base_dir, "test_histograms.jsonl")
    write_histograms_jsonl(test_records, test_path)
    n_true_anomalies = int(test_labels.sum())
    print(f"  Generated {len(test_records)} test histograms "
          f"({n_true_anomalies} true anomalies)")

    # ------------------------------------------------------------------
    # Step 4: Run inference
    # ------------------------------------------------------------------
    print("\n[Step 4/5] Running model inference...")
    inferencer = ONNXInferenceTool(
        model_path=train_result["model_path"],
        input_data_path="test_histograms.jsonl",
        output_path="scores.jsonl",
        normalization="unit_area",
        base_directory=base_dir,
    )
    inferencer._setup()
    inf_result = _print_result("Inference complete", inferencer._run())

    # ------------------------------------------------------------------
    # Step 5: Score and generate report
    # ------------------------------------------------------------------
    print("\n[Step 5a/5] Scoring anomalies...")
    scorer = AnomalyScorerTool(
        scores_path="scores.jsonl",
        output_path="scored.jsonl",
        threshold_method="percentile",
        warning_threshold=90.0,
        bad_threshold=98.0,
        base_directory=base_dir,
    )
    scorer._setup()
    score_result = _print_result("Scoring complete", scorer._run())

    print("\n[Step 5b/5] Generating anomaly report...")
    alerter = AlertManagerTool(
        scored_path="scored.jsonl",
        output_dir="report",
        top_n=10,
        report_title="Run 370000 ECAL DQM Anomaly Report",
        base_directory=base_dir,
    )
    alerter._setup()
    alert_result = _print_result("Report generated", alerter._run())

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("  Demo Complete!")
    print("=" * 70)

    if alert_result.get("status") == "ok":
        print(f"\n  Overall verdict:  {alert_result['overall_verdict']}")
        print(f"  Total alerts:     {alert_result['n_alerts']}")
        print(f"  True anomalies:   {n_true_anomalies}")
        report_path = os.path.join(base_dir, alert_result["report_path"])
        print(f"\n  Full report:      {report_path}")
        print(f"  Alert payload:    {os.path.join(base_dir, alert_result['alert_path'])}")

    print(f"\n  All artifacts saved to: {base_dir}")
    print()


# ---------------------------------------------------------------------------
# Orchestral Agent mode (uncomment to use with web UI)
# ---------------------------------------------------------------------------

# To run this demo as an interactive agent with the web UI:
#
# from orchestral import Agent
# from orchestral.tools import (RunCommandTool, WriteFileTool, ReadFileTool,
#                               RunPythonTool, WebSearchTool)
# from orchestral.llm import GPT, Claude
# from tools.dqm.data_access import DQMIOReaderTool, DQMAPIClientTool
# from tools.dqm.training import AutoencoderTrainerTool, ModelRegistryTool
# from tools.dqm.deployment import ONNXInferenceTool, AnomalyScorerTool, AlertManagerTool
#
# # Load system prompt
# with open(REPO_ROOT / "prompts/examples/dqm/system/dqm_monitor_prompt.md") as f:
#     system_prompt = f.read()
#
# base_directory = str(REPO_ROOT / "dqm_demo_workspace")
#
# tools = [
#     RunCommandTool(base_directory=base_directory),
#     WriteFileTool(base_directory=base_directory),
#     ReadFileTool(base_directory=base_directory),
#     RunPythonTool(base_directory=base_directory, timeout=600),
#     WebSearchTool(),
#     DQMIOReaderTool(base_directory=base_directory),
#     DQMAPIClientTool(base_directory=base_directory),
#     AutoencoderTrainerTool(base_directory=base_directory),
#     ModelRegistryTool(base_directory=base_directory),
#     ONNXInferenceTool(base_directory=base_directory),
#     AnomalyScorerTool(base_directory=base_directory),
#     AlertManagerTool(base_directory=base_directory),
# ]
#
# LLM = GPT()  # or Claude(), Gemini(), get_ollama()
# agent = Agent(llm=LLM, tools=tools, system_prompt=system_prompt)
#
# import app.server as app_server
# app_server.run_server(agent, host="127.0.0.1", port=8000, open_browser=True)


if __name__ == "__main__":
    run_demo()
