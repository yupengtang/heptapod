"""
# histogram_utils.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
"""
Histogram utility functions for CMS DQM data processing.

Provides shared helpers for normalization, rebinning, flattening, and
serialization of histogram data used across data-access, training, and
deployment tools.
"""

import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# --------------------------------------------------------------------------- #
# Schema version for DQM histogram JSONL files
# --------------------------------------------------------------------------- #
DQMHIST_SCHEMA_VERSION = "dqmhist-1.0"


# --------------------------------------------------------------------------- #
# Normalization helpers
# --------------------------------------------------------------------------- #

def normalize_histogram(counts: np.ndarray,
                        method: str = "unit_area") -> np.ndarray:
    """
    Normalize a histogram counts array.

    Args:
        counts: 1-D or 2-D array of bin counts.
        method: One of ``"unit_area"``, ``"max_bin"``, ``"z_score"``, or
                ``"minmax"``.

    Returns:
        Normalized array of the same shape.

    Raises:
        ValueError: If *method* is not recognised.
    """
    counts = np.asarray(counts, dtype=np.float64)

    if method == "unit_area":
        total = counts.sum()
        return counts / total if total > 0 else counts

    if method == "max_bin":
        max_val = counts.max()
        return counts / max_val if max_val > 0 else counts

    if method == "z_score":
        mean = counts.mean()
        std = counts.std()
        return (counts - mean) / std if std > 0 else counts - mean

    if method == "minmax":
        cmin, cmax = counts.min(), counts.max()
        denom = cmax - cmin
        return (counts - cmin) / denom if denom > 0 else counts - cmin

    raise ValueError(
        f"Unknown normalization method '{method}'. "
        f"Choose from: unit_area, max_bin, z_score, minmax"
    )


# --------------------------------------------------------------------------- #
# Rebinning
# --------------------------------------------------------------------------- #

def rebin_histogram(counts: np.ndarray,
                    bin_edges: np.ndarray,
                    target_nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rebin a 1-D histogram to *target_nbins* uniform bins.

    When the original number of bins is not evenly divisible by
    *target_nbins*, the last merged bin absorbs the remainder.

    Args:
        counts:       1-D array of length N (bin counts).
        bin_edges:    1-D array of length N+1.
        target_nbins: Desired number of output bins.

    Returns:
        Tuple of (new_counts, new_edges).
    """
    counts = np.asarray(counts, dtype=np.float64)
    bin_edges = np.asarray(bin_edges, dtype=np.float64)
    n_orig = len(counts)

    if target_nbins >= n_orig:
        return counts.copy(), bin_edges.copy()

    group_size = n_orig // target_nbins
    new_counts = np.zeros(target_nbins, dtype=np.float64)
    new_edges = np.zeros(target_nbins + 1, dtype=np.float64)

    for i in range(target_nbins):
        start = i * group_size
        # Last bin absorbs any remainder
        end = (i + 1) * group_size if i < target_nbins - 1 else n_orig
        new_counts[i] = counts[start:end].sum()
        new_edges[i] = bin_edges[start]
    new_edges[-1] = bin_edges[-1]

    return new_counts, new_edges


# --------------------------------------------------------------------------- #
# 2-D flattening
# --------------------------------------------------------------------------- #

def flatten_2d_histogram(counts_2d: np.ndarray) -> np.ndarray:
    """
    Flatten a 2-D histogram into a 1-D feature vector (row-major order).

    Args:
        counts_2d: 2-D array of shape (ny, nx).

    Returns:
        1-D array of length ny * nx.
    """
    return np.asarray(counts_2d, dtype=np.float64).ravel()


# --------------------------------------------------------------------------- #
# Serialization helpers  (dqmhist-1.0 JSONL)
# --------------------------------------------------------------------------- #

def histogram_to_record(name: str,
                        counts: np.ndarray,
                        bin_edges: Optional[np.ndarray] = None,
                        run: Optional[int] = None,
                        lumi_section: Optional[int] = None,
                        subsystem: Optional[str] = None,
                        extra_metadata: Optional[Dict[str, Any]] = None
                        ) -> Dict[str, Any]:
    """
    Build a single JSONL record for a histogram in the ``dqmhist-1.0``
    schema.

    Args:
        name:           Full DQM path, e.g. ``"ECAL/EBOccupancy"``.
        counts:         1-D or 2-D array of bin contents.
        bin_edges:      1-D array of bin edges (length = nbins+1 for 1-D).
                        If ``None``, uses integer bin indices.
        run:            CMS run number.
        lumi_section:   Luminosity section number.
        subsystem:      Detector subsystem identifier.
        extra_metadata: Arbitrary extra metadata to include.

    Returns:
        A dict ready for ``json.dumps``.
    """
    counts = np.asarray(counts, dtype=np.float64)
    shape = list(counts.shape)
    rec: Dict[str, Any] = {
        "schema": DQMHIST_SCHEMA_VERSION,
        "name": name,
        "shape": shape,
        "counts": counts.tolist(),
    }

    if bin_edges is not None:
        rec["bin_edges"] = np.asarray(bin_edges, dtype=np.float64).tolist()
    if run is not None:
        rec["run"] = int(run)
    if lumi_section is not None:
        rec["lumi_section"] = int(lumi_section)
    if subsystem is not None:
        rec["subsystem"] = subsystem
    if extra_metadata:
        rec["metadata"] = extra_metadata

    return rec


def write_histograms_jsonl(records: List[Dict[str, Any]],
                           output_path: str) -> int:
    """
    Write a list of histogram records to a JSONL file.

    Args:
        records:     List of dicts (from ``histogram_to_record``).
        output_path: File path to write.

    Returns:
        Number of records written.
    """
    import os
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fp:
        for rec in records:
            fp.write(
                json.dumps(rec, separators=(",", ":"), ensure_ascii=False)
                + "\n"
            )
    return len(records)


def read_histograms_jsonl(input_path: str) -> List[Dict[str, Any]]:
    """
    Read histogram records from a JSONL file.

    Args:
        input_path: Path to JSONL file.

    Returns:
        List of histogram record dicts.
    """
    records: List[Dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# --------------------------------------------------------------------------- #
# NumPy conversion
# --------------------------------------------------------------------------- #

def records_to_numpy(records: List[Dict[str, Any]],
                     flatten: bool = True,
                     normalization: Optional[str] = None
                     ) -> np.ndarray:
    """
    Convert a list of histogram records into a NumPy array suitable for
    ML model input.

    All histograms are zero-padded to the length of the longest one.

    Args:
        records:       Histogram record dicts with ``"counts"`` key.
        flatten:       If True, flatten 2-D histograms.
        normalization: Optional normalization method to apply
                       (``"unit_area"``, ``"max_bin"``, etc.).

    Returns:
        2-D array of shape ``(n_histograms, max_bins)``.
    """
    arrays = []
    for rec in records:
        arr = np.asarray(rec["counts"], dtype=np.float64)
        if flatten and arr.ndim > 1:
            arr = arr.ravel()
        if normalization:
            arr = normalize_histogram(arr, method=normalization)
        arrays.append(arr)

    if not arrays:
        return np.empty((0, 0), dtype=np.float64)

    max_len = max(a.size for a in arrays)
    padded = np.zeros((len(arrays), max_len), dtype=np.float64)
    for i, a in enumerate(arrays):
        padded[i, :a.size] = a.ravel()

    return padded


# --------------------------------------------------------------------------- #
# Synthetic data generation (for testing & demos)
# --------------------------------------------------------------------------- #

def generate_synthetic_histograms(n_histograms: int = 100,
                                  n_bins: int = 64,
                                  anomaly_fraction: float = 0.1,
                                  seed: int = 42
                                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic DQM-like histograms for testing and demonstration.

    Normal histograms follow a Gaussian-like distribution.  Anomalous
    histograms have additional spikes or missing regions.

    Args:
        n_histograms:    Number of histograms to generate.
        n_bins:          Number of bins per histogram.
        anomaly_fraction: Fraction of histograms that are anomalous.
        seed:            Random seed for reproducibility.

    Returns:
        Tuple of (histograms, labels) where histograms has shape
        ``(n_histograms, n_bins)`` and labels is 0 (normal) or 1 (anomaly).
    """
    rng = np.random.RandomState(seed)
    n_anomalies = int(n_histograms * anomaly_fraction)
    n_normal = n_histograms - n_anomalies

    # Normal histograms: Gaussian-like occupancy pattern
    x = np.linspace(-3, 3, n_bins)
    base = np.exp(-0.5 * x ** 2)

    normal_hists = np.array([
        base + rng.normal(0, 0.05, n_bins) for _ in range(n_normal)
    ])
    normal_hists = np.clip(normal_hists, 0, None)

    # Anomalous histograms: base pattern with defects
    anomaly_hists = []
    for _ in range(n_anomalies):
        h = base.copy() + rng.normal(0, 0.05, n_bins)
        defect_type = rng.choice(["spike", "dead_region", "shift"])

        if defect_type == "spike":
            spike_pos = rng.randint(0, n_bins)
            spike_width = rng.randint(1, 4)
            start = max(0, spike_pos - spike_width)
            end = min(n_bins, spike_pos + spike_width)
            h[start:end] += rng.uniform(0.5, 2.0)

        elif defect_type == "dead_region":
            dead_start = rng.randint(0, n_bins - 10)
            dead_width = rng.randint(5, 15)
            end = min(n_bins, dead_start + dead_width)
            h[dead_start:end] = 0.0

        elif defect_type == "shift":
            h = np.roll(h, rng.randint(-10, 10))
            h += rng.uniform(-0.3, 0.3)

        h = np.clip(h, 0, None)
        anomaly_hists.append(h)

    if anomaly_hists:
        anomaly_hists = np.array(anomaly_hists)
        histograms = np.vstack([normal_hists, anomaly_hists])
        labels = np.concatenate([
            np.zeros(n_normal, dtype=np.int32),
            np.ones(n_anomalies, dtype=np.int32)
        ])
    else:
        histograms = normal_hists
        labels = np.zeros(n_normal, dtype=np.int32)

    # Shuffle
    perm = rng.permutation(len(histograms))
    return histograms[perm], labels[perm]
