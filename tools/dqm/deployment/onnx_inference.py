"""
# onnx_inference.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import os
from typing import Optional, List

import numpy as np

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from ..data_access.histogram_utils import (
    read_histograms_jsonl,
    records_to_numpy,
)

# --------------------------------------------------------------------------- #
# Schema version for inference output
# --------------------------------------------------------------------------- #
DQMSCORE_SCHEMA_VERSION = "dqmscore-1.0"


class ONNXInferenceTool(BaseTool):
    """
    Run ONNX model inference on DQM histograms to compute per-histogram
    reconstruction errors for anomaly detection.

    This tool loads a pre-trained ONNX autoencoder model and computes
    the reconstruction of each input histogram.  The per-histogram
    mean squared error (MSE) between input and reconstruction serves
    as the anomaly score.

    Inputs (runtime):
      - model_path: Relative path to the ONNX model file (.onnx).
      - input_data_path: Relative path to histogram JSONL file.
      - output_path: Relative path for output anomaly scores JSONL.
      - normalization: (optional) Normalization to apply before inference.
        Must match what the model was trained on (default ``"unit_area"``).
      - batch_size: (optional) Inference batch size (default 256).

    State:
      - base_directory: Sandbox root for file operations.
      - onnx_providers: ONNX Runtime execution providers as comma-separated
        string (default ``"CPUExecutionProvider"``).

    Behavior:
      1. Loads the ONNX model with ONNX Runtime.
      2. Reads and normalizes histogram data from JSONL.
      3. Runs batched inference to compute reconstructions.
      4. Calculates per-histogram reconstruction error (MSE).
      5. Writes scores to output JSONL with ``dqmscore-1.0`` schema.

    Output (JSON):
      {
        "status": "ok",
        "output_path": "<relative>",
        "n_histograms": <int>,
        "mean_score": <float>,
        "max_score": <float>,
        "min_score": <float>
      }

    Errors:
      Returns self.format_error() on failures including:
        - ONNX Runtime not installed
        - Model file not found
        - Shape mismatch between model and data
    """

    # ======================== Runtime fields ======================== #
    model_path: str = RuntimeField(
        description="Relative path to the ONNX model file (.onnx)"
    )
    input_data_path: str = RuntimeField(
        description="Relative path to histogram JSONL file"
    )
    output_path: str = RuntimeField(
        description="Relative path for output anomaly scores JSONL"
    )
    normalization: str = RuntimeField(
        default="unit_area",
        description="Normalization to apply (must match training): unit_area, max_bin, z_score, minmax"
    )
    batch_size: int = RuntimeField(
        default=256,
        description="Inference batch size"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        default=".", description="Base sandbox directory"
    )
    onnx_providers: str = StateField(
        default="CPUExecutionProvider",
        description="ONNX Runtime execution providers (comma-separated)"
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

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Setup Error", reason=str(e))

        # Check ONNX Runtime
        try:
            import onnxruntime as ort
        except ImportError as e:
            return self.format_error(
                error="Dependency Missing",
                reason="onnxruntime is not installed",
                suggestion="Install with: pip install onnxruntime",
                context=str(e)
            )

        # Resolve paths
        model_abs = self._safe_path(self.model_path)
        data_abs = self._safe_path(self.input_data_path)
        out_abs = self._safe_path(self.output_path)

        if not model_abs:
            return self.format_error(
                error="Access Denied",
                reason="model_path escapes base_directory"
            )
        if not data_abs:
            return self.format_error(
                error="Access Denied",
                reason="input_data_path escapes base_directory"
            )
        if not out_abs:
            return self.format_error(
                error="Access Denied",
                reason="output_path escapes base_directory"
            )

        if not os.path.exists(model_abs):
            return self.format_error(
                error="File Not Found",
                reason="ONNX model file not found",
                context=f"path={self.model_path}",
                suggestion="Train a model first or provide correct path"
            )
        if not os.path.exists(data_abs):
            return self.format_error(
                error="File Not Found",
                reason="Input data file not found",
                context=f"path={self.input_data_path}"
            )

        # Load model
        try:
            providers = [p.strip() for p in self.onnx_providers.split(",")]
            session = ort.InferenceSession(model_abs, providers=providers)
            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name
            model_input_shape = session.get_inputs()[0].shape
        except Exception as e:
            return self.format_error(
                error="Model Error",
                reason=f"Failed to load ONNX model: {e}",
                suggestion="Verify the .onnx file is valid"
            )

        # Load data
        try:
            records = read_histograms_jsonl(data_abs)
            data_array = records_to_numpy(
                records, flatten=True, normalization=self.normalization
            )
        except Exception as e:
            return self.format_error(
                error="Data Error",
                reason=str(e),
                suggestion="Check input JSONL format"
            )

        if data_array.shape[0] == 0:
            return self.format_error(
                error="Empty Data",
                reason="No histograms in input file"
            )

        # Check dimension compatibility
        expected_dim = model_input_shape[-1] if model_input_shape[-1] is not None else data_array.shape[1]
        if data_array.shape[1] != expected_dim:
            return self.format_error(
                error="Shape Mismatch",
                reason=(
                    f"Data dimension ({data_array.shape[1]}) does not match "
                    f"model input dimension ({expected_dim})"
                ),
                suggestion="Ensure data preprocessing matches training configuration"
            )

        # Run batched inference
        try:
            all_scores = []
            all_recons = []
            n = data_array.shape[0]

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                batch = data_array[start:end].astype(np.float32)
                recon = session.run([output_name], {input_name: batch})[0]

                # Ensure shapes match for MSE
                if recon.shape != batch.shape:
                    recon = recon.reshape(batch.shape)

                # Per-histogram MSE
                mse = np.mean((batch - recon) ** 2, axis=1)
                all_scores.extend(mse.tolist())
                all_recons.append(recon)

            all_recons = np.vstack(all_recons)
        except Exception as e:
            return self.format_error(
                error="Inference Error",
                reason=str(e),
                suggestion="Check model and data compatibility"
            )

        # Write output JSONL
        try:
            os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
            with open(out_abs, "w", encoding="utf-8") as fp:
                for i, score in enumerate(all_scores):
                    rec = {
                        "schema": DQMSCORE_SCHEMA_VERSION,
                        "histogram_index": i,
                        "anomaly_score": float(score),
                    }
                    # Carry forward metadata from input records
                    if i < len(records):
                        if "name" in records[i]:
                            rec["histogram_name"] = records[i]["name"]
                        if "run" in records[i]:
                            rec["run"] = records[i]["run"]
                        if "lumi_section" in records[i]:
                            rec["lumi_section"] = records[i]["lumi_section"]
                        if "subsystem" in records[i]:
                            rec["subsystem"] = records[i]["subsystem"]

                    fp.write(
                        json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
                        + "\n"
                    )
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                suggestion="Check disk space and permissions"
            )

        scores_arr = np.array(all_scores)
        result = {
            "status": "ok",
            "output_path": os.path.relpath(out_abs, self.base_directory),
            "n_histograms": len(all_scores),
            "mean_score": float(scores_arr.mean()),
            "max_score": float(scores_arr.max()),
            "min_score": float(scores_arr.min()),
            "std_score": float(scores_arr.std()),
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
