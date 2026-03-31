"""
# anomaly_scorer.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import os
from typing import Optional

import numpy as np

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField


class AnomalyScorerTool(BaseTool):
    """
    Process raw reconstruction errors from ONNXInferenceTool into
    human-interpretable anomaly verdicts with configurable thresholds.

    This tool reads the ``dqmscore-1.0`` JSONL produced by the inference
    step, applies a thresholding strategy, and assigns quality flags
    (``"GOOD"``, ``"WARNING"``, ``"BAD"``) to each histogram and to
    the overall run.

    Inputs (runtime):
      - scores_path: Relative path to anomaly scores JSONL from
        ONNXInferenceTool.
      - output_path: Relative path for scored output JSONL.
      - threshold_method: Thresholding strategy. One of:
          ``"static"``     – Fixed threshold values.
          ``"percentile"`` – Thresholds from score distribution percentiles.
          ``"sigma"``      – Thresholds as multiples of standard deviation.
      - warning_threshold: Threshold for WARNING flag.
        For ``static``: absolute MSE value (default 0.01).
        For ``percentile``: percentile (default 95).
        For ``sigma``: number of sigma (default 2.0).
      - bad_threshold: Threshold for BAD flag.
        For ``static``: absolute MSE value (default 0.05).
        For ``percentile``: percentile (default 99).
        For ``sigma``: number of sigma (default 3.0).

    State:
      - base_directory: Sandbox root.

    Output (JSON):
      {
        "status": "ok",
        "output_path": "<relative>",
        "summary": {
          "n_total": <int>,
          "n_good": <int>,
          "n_warning": <int>,
          "n_bad": <int>,
          "overall_verdict": "GOOD" | "WARNING" | "BAD"
        }
      }
    """

    # ======================== Runtime fields ======================== #
    scores_path: str = RuntimeField(
        description="Relative path to anomaly scores JSONL from ONNXInferenceTool"
    )
    output_path: str = RuntimeField(
        description="Relative path for scored output JSONL"
    )
    threshold_method: str = RuntimeField(
        default="percentile",
        description="Thresholding strategy: 'static', 'percentile', or 'sigma'"
    )
    warning_threshold: float = RuntimeField(
        default=95.0,
        description="WARNING threshold (meaning depends on threshold_method)"
    )
    bad_threshold: float = RuntimeField(
        default=99.0,
        description="BAD threshold (meaning depends on threshold_method)"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        default=".", description="Base sandbox directory"
    )
    # ================================================================ #

    def _setup(self):
        self.base_directory = os.path.abspath(self.base_directory)
        if not os.path.isdir(self.base_directory):
            raise ValueError(f"Base directory does not exist: {self.base_directory}")

    def _safe_path(self, rel_or_abs: str) -> Optional[str]:
        if not rel_or_abs:
            return None
        full = os.path.abspath(os.path.join(self.base_directory, rel_or_abs))
        if full.startswith(self.base_directory + os.sep) or full == self.base_directory:
            return full
        return None

    def _compute_thresholds(self, scores: np.ndarray):
        """Compute warning and bad thresholds from the score distribution."""
        if self.threshold_method == "static":
            return self.warning_threshold, self.bad_threshold

        elif self.threshold_method == "percentile":
            warn = np.percentile(scores, self.warning_threshold)
            bad = np.percentile(scores, self.bad_threshold)
            return warn, bad

        elif self.threshold_method == "sigma":
            mean = scores.mean()
            std = scores.std()
            warn = mean + self.warning_threshold * std
            bad = mean + self.bad_threshold * std
            return warn, bad

        else:
            # Fall back to percentile
            return (np.percentile(scores, 95), np.percentile(scores, 99))

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Setup Error", reason=str(e))

        valid_methods = {"static", "percentile", "sigma"}
        if self.threshold_method not in valid_methods:
            return self.format_error(
                error="Invalid Parameter",
                reason=f"threshold_method must be one of {sorted(valid_methods)}",
                suggestion=f"Use: {', '.join(sorted(valid_methods))}"
            )

        # Resolve paths
        src = self._safe_path(self.scores_path)
        dst = self._safe_path(self.output_path)

        if not src:
            return self.format_error(
                error="Access Denied",
                reason="scores_path escapes base_directory"
            )
        if not dst:
            return self.format_error(
                error="Access Denied",
                reason="output_path escapes base_directory"
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason="Scores file not found",
                context=f"path={self.scores_path}"
            )

        # Read scores
        try:
            records = []
            with open(src, "r", encoding="utf-8") as fp:
                for line in fp:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        except Exception as e:
            return self.format_error(
                error="Read Error",
                reason=str(e),
                context=f"path={src}"
            )

        if not records:
            return self.format_error(
                error="Empty Data",
                reason="No score records found"
            )

        # Extract scores
        scores = np.array([r.get("anomaly_score", 0.0) for r in records], dtype=np.float64)

        # Compute thresholds
        warn_thresh, bad_thresh = self._compute_thresholds(scores)

        # Assign verdicts
        n_good = 0
        n_warning = 0
        n_bad = 0
        scored_records = []

        for i, rec in enumerate(records):
            score = scores[i]
            if score >= bad_thresh:
                verdict = "BAD"
                n_bad += 1
            elif score >= warn_thresh:
                verdict = "WARNING"
                n_warning += 1
            else:
                verdict = "GOOD"
                n_good += 1

            scored_rec = dict(rec)  # Copy
            scored_rec["verdict"] = verdict
            scored_rec["warning_threshold"] = float(warn_thresh)
            scored_rec["bad_threshold"] = float(bad_thresh)
            scored_rec["threshold_method"] = self.threshold_method
            scored_records.append(scored_rec)

        # Overall verdict
        if n_bad > 0:
            overall = "BAD"
        elif n_warning > 0:
            overall = "WARNING"
        else:
            overall = "GOOD"

        # Write output
        try:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            with open(dst, "w", encoding="utf-8") as fp:
                for rec in scored_records:
                    fp.write(
                        json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
                        + "\n"
                    )
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e)
            )

        summary = {
            "n_total": len(records),
            "n_good": n_good,
            "n_warning": n_warning,
            "n_bad": n_bad,
            "overall_verdict": overall,
            "warning_threshold": float(warn_thresh),
            "bad_threshold": float(bad_thresh),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
        }

        result = {
            "status": "ok",
            "output_path": os.path.relpath(dst, self.base_directory),
            "summary": summary,
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
