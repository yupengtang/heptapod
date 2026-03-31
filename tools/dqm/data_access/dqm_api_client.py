"""
# dqm_api_client.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import json
import os
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField

from .histogram_utils import (
    DQMHIST_SCHEMA_VERSION,
    histogram_to_record,
    write_histograms_jsonl,
)


class DQMAPIClientTool(BaseTool):
    """
    Query the CMS DQM GUI web API to fetch histogram data and run metadata.

    This tool provides agent-accessible access to CMS DQM monitoring data
    via the public REST endpoints exposed by the DQM GUI service.

    Inputs (runtime):
      - action: The API action to perform. One of:
          ``"list_runs"``     – list available runs for a dataset.
          ``"list_histograms"`` – list histogram paths for a run.
          ``"fetch_histogram"`` – download histogram data to JSONL.
          ``"run_summary"``   – get a summary of run quality status.
      - dataset: CMS dataset path (e.g. ``"/Global/Online/ALL"``).
      - run_number: (optional) Specific run number.
      - histogram_path: (optional) Full DQM histogram path for fetch.
      - output_path: (optional) Relative output path for fetched data.
      - max_results: (optional) Maximum number of results to return
        (default 50).

    State:
      - dqm_api_url: Base URL for the CMS DQM GUI API endpoint.
      - base_directory: Sandbox root for file operations.

    Behavior:
      1. Constructs the appropriate REST API request URL.
      2. Queries the DQM GUI API using ``requests``.
      3. Parses the JSON response into the requested format.
      4. For ``fetch_histogram``, writes data to JSONL.

    Output (JSON):
      For list_runs:
        {"status":"ok", "runs":[...], "n_runs":<int>}
      For list_histograms:
        {"status":"ok", "histograms":[...], "n_histograms":<int>}
      For fetch_histogram:
        {"status":"ok", "output_path":"<rel>", "n_histograms":<int>}
      For run_summary:
        {"status":"ok", "run":<int>, "summary":{...}}

    Errors:
      Returns self.format_error() on failures including:
        - Network errors / API unavailable
        - Invalid action
        - requests not installed
    """

    # ======================== Runtime fields ======================== #
    action: str = RuntimeField(
        description=(
            "API action: 'list_runs', 'list_histograms', "
            "'fetch_histogram', or 'run_summary'"
        )
    )
    dataset: str = RuntimeField(
        default="/Global/Online/ALL",
        description="CMS dataset path (e.g. '/Global/Online/ALL')"
    )
    run_number: Optional[int] = RuntimeField(
        default=None,
        description="CMS run number (required for most actions)"
    )
    histogram_path: Optional[str] = RuntimeField(
        default=None,
        description="Full DQM histogram path to fetch (for fetch_histogram)"
    )
    output_path: Optional[str] = RuntimeField(
        default=None,
        description="Relative output path for fetched histogram JSONL"
    )
    max_results: int = RuntimeField(
        default=50,
        description="Maximum number of results to return"
    )
    # ================================================================ #

    # ========================= State fields ========================= #
    dqm_api_url: str = StateField(
        default="https://cmsweb.cern.ch/dqm/offline",
        description="Base URL for the CMS DQM GUI API"
    )
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

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Setup Error", reason=str(e))

        # Validate action
        valid_actions = {"list_runs", "list_histograms", "fetch_histogram", "run_summary"}
        if self.action not in valid_actions:
            return self.format_error(
                error="Invalid Action",
                reason=f"action must be one of {sorted(valid_actions)}",
                context=f"Got: {self.action}",
                suggestion=f"Use one of: {', '.join(sorted(valid_actions))}"
            )

        # Import requests
        try:
            import requests
        except ImportError as e:
            return self.format_error(
                error="Dependency Missing",
                reason="requests is not installed",
                suggestion="Install with: pip install requests",
                context=str(e)
            )

        try:
            if self.action == "list_runs":
                return self._list_runs(requests)
            elif self.action == "list_histograms":
                return self._list_histograms(requests)
            elif self.action == "fetch_histogram":
                return self._fetch_histogram(requests)
            elif self.action == "run_summary":
                return self._run_summary(requests)
        except requests.exceptions.ConnectionError as e:
            return self.format_error(
                error="Connection Error",
                reason="Could not reach the DQM API",
                context=f"url={self.dqm_api_url}",
                suggestion=(
                    "Check network connectivity. If outside CERN, "
                    "you may need a CERN SSO certificate or VPN."
                )
            )
        except requests.exceptions.Timeout:
            return self.format_error(
                error="Timeout",
                reason="DQM API request timed out",
                suggestion="Try again later or reduce max_results"
            )
        except Exception as e:
            return self.format_error(
                error="API Error",
                reason=str(e),
                suggestion="Check the DQM API URL and parameters"
            )

        # Should not reach here
        return self.format_error(error="Unknown Error", reason="Unexpected code path")

    def _list_runs(self, requests_mod) -> str:
        """List available runs from the DQM GUI API."""
        url = f"{self.dqm_api_url}/data/json/samples"
        params = {"match": self.dataset}
        resp = requests_mod.get(url, params=params, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()

        runs = []
        samples = data.get("samples", [])
        for sample in samples[:self.max_results]:
            if "run" in sample:
                runs.append({
                    "run": sample["run"],
                    "dataset": sample.get("dataset", self.dataset),
                    "importversion": sample.get("importversion", 0),
                })

        result = {"status": "ok", "runs": runs, "n_runs": len(runs)}
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

    def _list_histograms(self, requests_mod) -> str:
        """List histogram paths for a given run."""
        if not self.run_number:
            return self.format_error(
                error="Missing Parameter",
                reason="run_number is required for list_histograms",
                suggestion="Provide a valid CMS run number"
            )

        url = f"{self.dqm_api_url}/data/json/archive/{self.run_number}{self.dataset}"
        resp = requests_mod.get(url, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()

        histograms = []
        contents = data.get("contents", [])
        for item in contents[:self.max_results]:
            if "obj" in item:
                histograms.append({
                    "name": item["obj"],
                    "type": item.get("type", "unknown"),
                    "path": item.get("path", ""),
                })

        result = {"status": "ok", "histograms": histograms, "n_histograms": len(histograms)}
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

    def _fetch_histogram(self, requests_mod) -> str:
        """Fetch specific histogram data and save to JSONL."""
        if not self.run_number or not self.histogram_path:
            return self.format_error(
                error="Missing Parameter",
                reason="run_number and histogram_path are required for fetch_histogram",
                suggestion="Provide both run_number and histogram_path"
            )

        if not self.output_path:
            return self.format_error(
                error="Missing Parameter",
                reason="output_path is required for fetch_histogram",
                suggestion="Provide a relative output path for the JSONL file"
            )

        dst = self._safe_path(self.output_path)
        if not dst:
            return self.format_error(
                error="Access Denied",
                reason="output_path escapes base_directory",
                suggestion="Use a relative path inside base_directory"
            )

        url = (
            f"{self.dqm_api_url}/data/json/archive/"
            f"{self.run_number}{self.dataset}/{self.histogram_path}"
        )
        resp = requests_mod.get(url, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()

        import numpy as np

        records = []
        contents = data.get("contents", [data]) if isinstance(data, dict) else [data]
        for item in contents:
            if "values" in item:
                counts = np.asarray(item["values"], dtype=np.float64)
                bin_edges = None
                if "xaxis" in item:
                    ax = item["xaxis"]
                    low = ax.get("first", {}).get("value", 0)
                    high = ax.get("last", {}).get("value", len(counts))
                    bin_edges = np.linspace(low, high, len(counts) + 1)

                name = item.get("obj", self.histogram_path)
                rec = histogram_to_record(
                    name=name,
                    counts=counts,
                    bin_edges=bin_edges,
                    run=self.run_number,
                    subsystem=name.split("/")[0] if "/" in name else None,
                )
                records.append(rec)

        if not records:
            return self.format_error(
                error="No Data",
                reason="No histogram data found in API response",
                suggestion="Check the histogram_path and run_number"
            )

        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        write_histograms_jsonl(records, dst)

        result = {
            "status": "ok",
            "output_path": os.path.relpath(dst, self.base_directory),
            "n_histograms": len(records),
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)

    def _run_summary(self, requests_mod) -> str:
        """Get a run quality summary."""
        if not self.run_number:
            return self.format_error(
                error="Missing Parameter",
                reason="run_number is required for run_summary",
                suggestion="Provide a valid CMS run number"
            )

        url = f"{self.dqm_api_url}/data/json/archive/{self.run_number}{self.dataset}"
        resp = requests_mod.get(url, timeout=30, verify=False)
        resp.raise_for_status()
        data = resp.json()

        summary = {
            "run": self.run_number,
            "dataset": self.dataset,
            "n_contents": len(data.get("contents", [])),
        }

        result = {"status": "ok", "run": self.run_number, "summary": summary}
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
