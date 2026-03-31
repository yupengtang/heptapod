"""
# dqmio_reader.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import os
from typing import Optional, List

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .histogram_utils import (
    DQMHIST_SCHEMA_VERSION,
    histogram_to_record,
    write_histograms_jsonl,
    normalize_histogram,
    rebin_histogram,
)


class DQMIOReaderTool(BaseTool):
    """
    Read DQM histogram data from ROOT files (DQMIO or legacy TDirectory
    format) and export them to JSONL in the ``dqmhist-1.0`` schema.

    Uses ``uproot`` for pure-Python ROOT access — no CMSSW environment
    required.

    Inputs (runtime):
      - root_file_path: Relative path to the input ROOT file (.root).
      - output_path: Relative path for output JSONL file.
      - subsystem_filter: (optional) Restrict to a subsystem prefix,
        e.g. ``"ECAL/"`` or ``"HCAL/"``.
      - histogram_name_filter: (optional) Comma-separated list of specific
        histogram names to extract.
      - run_number: (optional) CMS run number for metadata tagging.
      - normalization: (optional) Normalization method to apply.
        One of ``"unit_area"``, ``"max_bin"``, ``"z_score"``, ``"minmax"``.
      - target_nbins: (optional) Rebin all 1-D histograms to this many bins.

    State:
      - base_directory: Sandbox root for file operations.

    Behavior:
      1. Opens the ROOT file with uproot.
      2. Walks the directory tree to discover TH1 / TH2 histograms.
      3. Optionally filters by subsystem prefix or explicit name list.
      4. Optionally rebins and/or normalizes each histogram.
      5. Serializes to JSONL (``dqmhist-1.0`` schema).

    Output (JSON):
      {
        "status": "ok",
        "output_path": "<relative>",
        "n_histograms": <int>,
        "subsystems_found": ["ECAL", "HCAL", ...]
      }

    Errors:
      Returns self.format_error() on failures including:
        - File not found
        - uproot not installed
        - ROOT file parse errors
    """

    # ======================== Runtime fields ======================== #
    root_file_path: str = RuntimeField(
        description="Relative path to the DQM ROOT file (.root)"
    )
    output_path: str = RuntimeField(
        description="Relative path for output JSONL file (e.g. 'dqm/histograms.jsonl')"
    )
    subsystem_filter: Optional[str] = RuntimeField(
        default=None,
        description="Only extract histograms whose path starts with this prefix (e.g. 'ECAL/')"
    )
    histogram_name_filter: Optional[str] = RuntimeField(
        default=None,
        description="Comma-separated histogram names to extract (empty = all)"
    )
    run_number: Optional[int] = RuntimeField(
        default=None,
        description="CMS run number to tag in output metadata"
    )
    normalization: Optional[str] = RuntimeField(
        default=None,
        description="Normalization to apply: unit_area, max_bin, z_score, minmax"
    )
    target_nbins: Optional[int] = RuntimeField(
        default=None,
        description="Rebin all 1-D histograms to this number of bins"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    base_directory: str = StateField(
        default=".", description="Base sandbox directory for file operations"
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

    # ------------------------------------------------------------------ #
    # Internal helpers for walking ROOT directory trees
    # ------------------------------------------------------------------ #

    @staticmethod
    def _walk_root_file(root_dir, prefix: str = ""):
        """Recursively yield (full_path, object) for histogram-like objects."""
        import uproot

        for key in root_dir.keys(cycle=False):
            obj = root_dir[key]
            full_path = f"{prefix}{key}" if prefix else key

            # Recurse into directories
            if hasattr(obj, "keys"):
                yield from DQMIOReaderTool._walk_root_file(obj, full_path + "/")
            else:
                yield full_path, obj

    @staticmethod
    def _extract_histogram(obj) -> Optional[dict]:
        """
        Extract bin counts and edges from an uproot histogram object.

        Returns dict with 'counts', 'bin_edges', 'ndim' or None if not
        a histogram type.
        """
        import numpy as np

        # uproot TH1 / TH2
        try:
            values = obj.values()
            if values is not None:
                counts = np.asarray(values, dtype=np.float64)
                ndim = counts.ndim

                # Edges
                if ndim == 1:
                    edges = np.asarray(obj.axis().edges(), dtype=np.float64)
                    return {"counts": counts, "bin_edges": edges, "ndim": 1}
                elif ndim == 2:
                    return {"counts": counts, "bin_edges": None, "ndim": 2}
        except Exception:
            pass

        return None

    # ------------------------------------------------------------------ #

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Setup Error", reason=str(e))

        # Resolve paths
        src = self._safe_path(self.root_file_path)
        dst = self._safe_path(self.output_path)

        if not src:
            return self.format_error(
                error="Access Denied",
                reason="root_file_path escapes base_directory",
                suggestion="Use a relative path inside base_directory"
            )
        if not dst:
            return self.format_error(
                error="Access Denied",
                reason="output_path escapes base_directory",
                suggestion="Use a relative path inside base_directory"
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason="ROOT file not found",
                context=f"path={self.root_file_path}",
                suggestion="Verify the file path and try again"
            )

        # Import uproot
        try:
            import uproot
            import numpy as np
        except ImportError as e:
            return self.format_error(
                error="Dependency Missing",
                reason="uproot is not installed",
                suggestion="Install with: pip install uproot awkward",
                context=str(e)
            )

        # Parse name filter
        name_set = None
        if self.histogram_name_filter:
            name_set = {n.strip() for n in self.histogram_name_filter.split(",")}

        # Open and process
        try:
            root_file = uproot.open(src)
        except Exception as e:
            return self.format_error(
                error="Read Error",
                reason=f"Failed to open ROOT file: {e}",
                suggestion="Verify file is a valid ROOT file"
            )

        records = []
        subsystems_seen = set()

        try:
            for path, obj in self._walk_root_file(root_file):
                # Apply subsystem filter
                if self.subsystem_filter and not path.startswith(self.subsystem_filter):
                    continue

                # Apply name filter
                basename = path.rsplit("/", 1)[-1] if "/" in path else path
                if name_set and basename not in name_set:
                    continue

                # Extract histogram data
                hdata = self._extract_histogram(obj)
                if hdata is None:
                    continue

                counts = hdata["counts"]
                bin_edges = hdata.get("bin_edges")

                # Subsystem tracking
                subsys = path.split("/")[0] if "/" in path else "root"
                subsystems_seen.add(subsys)

                # Rebin (1-D only)
                if self.target_nbins and hdata["ndim"] == 1 and bin_edges is not None:
                    counts, bin_edges = rebin_histogram(
                        counts, bin_edges, self.target_nbins
                    )

                # Normalize
                if self.normalization:
                    counts = normalize_histogram(counts, method=self.normalization)

                rec = histogram_to_record(
                    name=path,
                    counts=counts,
                    bin_edges=bin_edges,
                    run=self.run_number,
                    subsystem=subsys,
                )
                records.append(rec)

        except Exception as e:
            return self.format_error(
                error="Processing Error",
                reason=f"Error walking ROOT file: {e}",
                suggestion="Check file structure and histogram types"
            )

        if not records:
            return self.format_error(
                error="No Data",
                reason="No histograms matched the specified filters",
                context=f"subsystem={self.subsystem_filter}, names={self.histogram_name_filter}",
                suggestion="Broaden filters or check the ROOT file contents"
            )

        # Write output
        try:
            os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
            write_histograms_jsonl(records, dst)
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=str(e),
                context=f"path={dst}",
                suggestion="Check disk space and permissions"
            )

        result = {
            "status": "ok",
            "output_path": os.path.relpath(dst, self.base_directory),
            "n_histograms": len(records),
            "subsystems_found": sorted(subsystems_seen),
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
