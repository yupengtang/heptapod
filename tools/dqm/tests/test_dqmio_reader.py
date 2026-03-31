#!/usr/bin/env python3
"""
# test_dqmio_reader.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
"""
import os
import sys
import json
import argparse
import shutil
from pathlib import Path

# Path setup
SCRIPT_PATH = Path(__file__).resolve()
TESTS_DIR = SCRIPT_PATH.parent
TOOL_DIR = TESTS_DIR.parent
REPO_ROOT = TOOL_DIR.parent.parent

# Add the data_access directory directly to path to avoid triggering
# the __init__.py import chain which pulls in orchestral-dependent tools
DATA_ACCESS_DIR = TOOL_DIR / "data_access"
sys.path.insert(0, str(DATA_ACCESS_DIR))

from histogram_utils import (
    normalize_histogram,
    rebin_histogram,
    flatten_2d_histogram,
    histogram_to_record,
    write_histograms_jsonl,
    read_histograms_jsonl,
    records_to_numpy,
    generate_synthetic_histograms,
    DQMHIST_SCHEMA_VERSION,
)

import numpy as np


def test_normalize_unit_area():
    """Test unit-area normalization."""
    print(">> Testing normalize_histogram (unit_area)...\n")
    counts = np.array([1.0, 2.0, 3.0, 4.0])
    result = normalize_histogram(counts, "unit_area")
    assert abs(result.sum() - 1.0) < 1e-10, f"Sum should be 1.0, got {result.sum()}"
    print("[✓] unit_area normalization passed\n")


def test_normalize_max_bin():
    """Test max-bin normalization."""
    print(">> Testing normalize_histogram (max_bin)...\n")
    counts = np.array([1.0, 2.0, 3.0, 4.0])
    result = normalize_histogram(counts, "max_bin")
    assert abs(result.max() - 1.0) < 1e-10, f"Max should be 1.0, got {result.max()}"
    print("[✓] max_bin normalization passed\n")


def test_normalize_z_score():
    """Test z-score normalization."""
    print(">> Testing normalize_histogram (z_score)...\n")
    counts = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = normalize_histogram(counts, "z_score")
    assert abs(result.mean()) < 1e-10, f"Mean should be ~0, got {result.mean()}"
    assert abs(result.std() - 1.0) < 1e-10, f"Std should be ~1, got {result.std()}"
    print("[✓] z_score normalization passed\n")


def test_normalize_minmax():
    """Test min-max normalization."""
    print(">> Testing normalize_histogram (minmax)...\n")
    counts = np.array([2.0, 4.0, 6.0, 8.0])
    result = normalize_histogram(counts, "minmax")
    assert abs(result.min()) < 1e-10, f"Min should be 0, got {result.min()}"
    assert abs(result.max() - 1.0) < 1e-10, f"Max should be 1, got {result.max()}"
    print("[✓] minmax normalization passed\n")


def test_normalize_zero_array():
    """Test normalization on zero array (edge case)."""
    print(">> Testing normalize_histogram (zero array)...\n")
    counts = np.zeros(10)
    result = normalize_histogram(counts, "unit_area")
    assert np.all(result == 0), "Zero array should remain zero"
    print("[✓] zero array normalization passed\n")


def test_rebin_histogram():
    """Test histogram rebinning."""
    print(">> Testing rebin_histogram...\n")
    counts = np.ones(100)
    edges = np.linspace(0, 100, 101)

    new_counts, new_edges = rebin_histogram(counts, edges, 10)
    assert len(new_counts) == 10, f"Expected 10 bins, got {len(new_counts)}"
    assert abs(new_counts.sum() - counts.sum()) < 1e-10, "Total counts should be preserved"
    assert len(new_edges) == 11, f"Expected 11 edges, got {len(new_edges)}"
    print("[✓] rebinning passed\n")


def test_rebin_identity():
    """Test rebinning when target >= original bins (should be identity)."""
    print(">> Testing rebin_histogram (identity)...\n")
    counts = np.array([1.0, 2.0, 3.0])
    edges = np.array([0.0, 1.0, 2.0, 3.0])

    new_counts, new_edges = rebin_histogram(counts, edges, 5)
    np.testing.assert_array_equal(new_counts, counts)
    np.testing.assert_array_equal(new_edges, edges)
    print("[✓] identity rebinning passed\n")


def test_flatten_2d():
    """Test 2-D histogram flattening."""
    print(">> Testing flatten_2d_histogram...\n")
    hist_2d = np.array([[1, 2], [3, 4], [5, 6]])
    flat = flatten_2d_histogram(hist_2d)
    assert flat.shape == (6,), f"Expected shape (6,), got {flat.shape}"
    np.testing.assert_array_equal(flat, [1, 2, 3, 4, 5, 6])
    print("[✓] 2D flattening passed\n")


def test_histogram_to_record():
    """Test histogram serialization to dqmhist-1.0 record."""
    print(">> Testing histogram_to_record...\n")
    counts = np.array([1.0, 2.0, 3.0])
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    rec = histogram_to_record(
        name="ECAL/test",
        counts=counts,
        bin_edges=edges,
        run=370000,
        lumi_section=1,
        subsystem="ECAL",
    )
    assert rec["schema"] == DQMHIST_SCHEMA_VERSION
    assert rec["name"] == "ECAL/test"
    assert rec["run"] == 370000
    assert rec["subsystem"] == "ECAL"
    assert len(rec["counts"]) == 3
    assert len(rec["bin_edges"]) == 4
    print("[✓] histogram_to_record passed\n")


def test_write_and_read_jsonl():
    """Test JSONL round-trip write/read."""
    print(">> Testing write/read JSONL round-trip...\n")
    output_path = str(TESTS_DIR / "test_data" / "test_output.jsonl")

    records = [
        histogram_to_record("hist1", np.array([1, 2, 3]), run=1),
        histogram_to_record("hist2", np.array([4, 5, 6]), run=2),
    ]
    n_written = write_histograms_jsonl(records, output_path)
    assert n_written == 2, f"Expected 2 records written, got {n_written}"

    read_back = read_histograms_jsonl(output_path)
    assert len(read_back) == 2
    assert read_back[0]["name"] == "hist1"
    assert read_back[1]["name"] == "hist2"
    print("[✓] JSONL round-trip passed\n")

    # Cleanup
    if os.path.exists(output_path):
        os.remove(output_path)


def test_records_to_numpy():
    """Test conversion of records to NumPy array."""
    print(">> Testing records_to_numpy...\n")
    records = [
        {"counts": [1.0, 2.0, 3.0]},
        {"counts": [4.0, 5.0]},  # Shorter, should be zero-padded
    ]
    arr = records_to_numpy(records, flatten=True, normalization=None)
    assert arr.shape == (2, 3), f"Expected shape (2,3), got {arr.shape}"
    assert arr[1, 2] == 0.0, "Shorter record should be zero-padded"
    print("[✓] records_to_numpy passed\n")


def test_records_to_numpy_normalized():
    """Test NumPy conversion with normalization."""
    print(">> Testing records_to_numpy (normalized)...\n")
    records = [{"counts": [1.0, 2.0, 3.0, 4.0]}]
    arr = records_to_numpy(records, normalization="unit_area")
    assert abs(arr[0].sum() - 1.0) < 1e-10
    print("[✓] records_to_numpy with normalization passed\n")


def test_generate_synthetic():
    """Test synthetic histogram generation."""
    print(">> Testing generate_synthetic_histograms...\n")
    hists, labels = generate_synthetic_histograms(
        n_histograms=100, n_bins=64, anomaly_fraction=0.1, seed=42
    )
    assert hists.shape == (100, 64), f"Expected (100,64), got {hists.shape}"
    assert labels.shape == (100,), f"Expected (100,), got {labels.shape}"
    assert np.all(hists >= 0), "All counts should be non-negative"
    n_anomalies = labels.sum()
    assert n_anomalies == 10, f"Expected 10 anomalies, got {n_anomalies}"
    print(f"[✓] Synthetic generation passed ({n_anomalies} anomalies)\n")


def cleanup_test_files():
    """Remove test-generated files."""
    print("\n>> Cleaning up test files...\n")
    cleanup_files = [
        TESTS_DIR / "test_data" / "test_output.jsonl",
    ]
    cleaned = 0
    for fp in cleanup_files:
        if fp.exists():
            try:
                fp.unlink()
                print(f"[✓] Removed: {fp.name}")
                cleaned += 1
            except Exception as e:
                print(f"[⚠] Failed to remove {fp.name}: {e}")
    if cleaned == 0:
        print("[i] No test files to clean up")
    else:
        print(f"\n[✓] Cleaned up {cleaned} file(s)\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run tests for DQM data access tools")
    parser.add_argument("--keep-files", action="store_true",
                        help="Keep test-generated files")
    args = parser.parse_args()

    all_passed = True
    tests = [
        test_normalize_unit_area,
        test_normalize_max_bin,
        test_normalize_z_score,
        test_normalize_minmax,
        test_normalize_zero_array,
        test_rebin_histogram,
        test_rebin_identity,
        test_flatten_2d,
        test_histogram_to_record,
        test_write_and_read_jsonl,
        test_records_to_numpy,
        test_records_to_numpy_normalized,
        test_generate_synthetic,
    ]

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"[✗] {test_fn.__name__} failed: {e}\n")
            all_passed = False

    if not args.keep_files:
        cleanup_test_files()
    else:
        print("\n[i] Keeping test files (--keep-files flag set)\n")

    if all_passed:
        print("[✓] All data access tests passed!\n")
        sys.exit(0)
    else:
        print("[✗] Some tests failed!\n")
        sys.exit(1)
