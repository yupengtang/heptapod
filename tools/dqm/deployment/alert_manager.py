"""
# alert_manager.py is a part of the HEPTAPOD package.
# Copyright (C) 2025 HEPTAPOD authors (see AUTHORS for details).
# HEPTAPOD is licensed under the GNU GPL v3 or later, see LICENSE for details.
# Please respect the MCnet Guidelines, see GUIDELINES for details.
"""
import datetime
import json
import os
from typing import Optional

from orchestral.tools.base.tool import BaseTool
from orchestral.tools.base.field_utils import RuntimeField, StateField


class AlertManagerTool(BaseTool):
    """
    Generate structured DQM anomaly reports and alerts from scored
    histogram data for human review.

    Reads the scored JSONL from AnomalyScorerTool and produces:
      - A Markdown summary report with top anomalies highlighted.
      - A JSON alert payload for downstream integration.

    Inputs (runtime):
      - scored_path: Relative path to scored anomaly JSONL from
        AnomalyScorerTool.
      - output_dir: Relative output directory for reports and alerts.
      - top_n: Number of top anomalies to highlight in the report
        (default 10).
      - report_title: (optional) Custom title for the Markdown report.

    State:
      - base_directory: Sandbox root.

    Output (JSON):
      {
        "status": "ok",
        "report_path": "<relative path to .md>",
        "alert_path": "<relative path to .json>",
        "n_alerts": <int>,
        "overall_verdict": "GOOD" | "WARNING" | "BAD"
      }
    """

    # ======================== Runtime fields ======================== #
    scored_path: str = RuntimeField(
        description="Relative path to scored anomaly JSONL from AnomalyScorerTool"
    )
    output_dir: str = RuntimeField(
        description="Relative output directory for reports and alerts"
    )
    top_n: int = RuntimeField(
        default=10,
        description="Number of top anomalies to highlight in report"
    )
    report_title: Optional[str] = RuntimeField(
        default=None,
        description="Custom title for the Markdown report"
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

    def _run(self) -> str:
        try:
            self._setup()
        except Exception as e:
            return self.format_error(error="Setup Error", reason=str(e))

        # Resolve paths
        src = self._safe_path(self.scored_path)
        out_dir = self._safe_path(self.output_dir)

        if not src:
            return self.format_error(
                error="Access Denied",
                reason="scored_path escapes base_directory"
            )
        if not out_dir:
            return self.format_error(
                error="Access Denied",
                reason="output_dir escapes base_directory"
            )
        if not os.path.exists(src):
            return self.format_error(
                error="File Not Found",
                reason="Scored data file not found",
                context=f"path={self.scored_path}"
            )

        os.makedirs(out_dir, exist_ok=True)

        # Read scored records
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
                reason=str(e)
            )

        if not records:
            return self.format_error(
                error="Empty Data",
                reason="No scored records found"
            )

        # Compute summary
        verdicts = [r.get("verdict", "GOOD") for r in records]
        n_good = verdicts.count("GOOD")
        n_warning = verdicts.count("WARNING")
        n_bad = verdicts.count("BAD")
        n_total = len(records)

        if n_bad > 0:
            overall = "BAD"
        elif n_warning > 0:
            overall = "WARNING"
        else:
            overall = "GOOD"

        # Sort by anomaly score descending for top-N
        sorted_records = sorted(
            records, key=lambda r: r.get("anomaly_score", 0.0), reverse=True
        )
        top_anomalies = sorted_records[:self.top_n]

        # ---- Generate Markdown report ---- #
        timestamp = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        title = self.report_title or "CMS DQM Anomaly Detection Report"

        verdict_emoji = {"GOOD": "✅", "WARNING": "⚠️", "BAD": "❌"}

        md_lines = [
            f"# {title}",
            "",
            f"**Generated:** {timestamp}",
            f"**Overall Verdict:** {verdict_emoji.get(overall, '❓')} **{overall}**",
            "",
            "---",
            "",
            "## Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Histograms | {n_total} |",
            f"| ✅ Good | {n_good} ({100*n_good/n_total:.1f}%) |",
            f"| ⚠️ Warning | {n_warning} ({100*n_warning/n_total:.1f}%) |",
            f"| ❌ Bad | {n_bad} ({100*n_bad/n_total:.1f}%) |",
            "",
        ]

        if top_anomalies:
            md_lines.extend([
                "---",
                "",
                f"## Top {len(top_anomalies)} Anomalies",
                "",
                "| Rank | Histogram | Score | Verdict |",
                "|------|-----------|-------|---------|",
            ])
            for i, rec in enumerate(top_anomalies, 1):
                name = rec.get("histogram_name", f"index-{rec.get('histogram_index', '?')}")
                score = rec.get("anomaly_score", 0.0)
                v = rec.get("verdict", "GOOD")
                md_lines.append(
                    f"| {i} | `{name}` | {score:.6f} | {verdict_emoji.get(v, '❓')} {v} |"
                )
            md_lines.append("")

        # Thresholding info
        if records:
            first = records[0]
            md_lines.extend([
                "---",
                "",
                "## Thresholding Configuration",
                "",
                f"- **Method:** {first.get('threshold_method', 'N/A')}",
                f"- **Warning Threshold:** {first.get('warning_threshold', 'N/A')}",
                f"- **Bad Threshold:** {first.get('bad_threshold', 'N/A')}",
                "",
            ])

        # Subsystem breakdown
        subsys_counts = {}
        for r in records:
            s = r.get("subsystem", "Unknown")
            v = r.get("verdict", "GOOD")
            if s not in subsys_counts:
                subsys_counts[s] = {"GOOD": 0, "WARNING": 0, "BAD": 0}
            subsys_counts[s][v] = subsys_counts[s].get(v, 0) + 1

        if subsys_counts and len(subsys_counts) > 1:
            md_lines.extend([
                "---",
                "",
                "## Breakdown by Subsystem",
                "",
                "| Subsystem | Good | Warning | Bad |",
                "|-----------|------|---------|-----|",
            ])
            for s in sorted(subsys_counts.keys()):
                c = subsys_counts[s]
                md_lines.append(
                    f"| {s} | {c.get('GOOD',0)} | {c.get('WARNING',0)} | {c.get('BAD',0)} |"
                )
            md_lines.append("")

        md_lines.extend([
            "---",
            "",
            "*Report generated by HEPTAPOD ML4DQM AlertManagerTool*",
        ])

        report_path = os.path.join(out_dir, "anomaly_report.md")
        try:
            with open(report_path, "w", encoding="utf-8") as fp:
                fp.write("\n".join(md_lines))
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=f"Failed to write report: {e}"
            )

        # ---- Generate JSON alert payload ---- #
        alert_payload = {
            "timestamp_utc": timestamp,
            "overall_verdict": overall,
            "n_total": n_total,
            "n_good": n_good,
            "n_warning": n_warning,
            "n_bad": n_bad,
            "top_anomalies": [
                {
                    "histogram_name": r.get("histogram_name", f"index-{r.get('histogram_index', '?')}"),
                    "anomaly_score": r.get("anomaly_score", 0.0),
                    "verdict": r.get("verdict", "GOOD"),
                    "run": r.get("run"),
                    "lumi_section": r.get("lumi_section"),
                    "subsystem": r.get("subsystem"),
                }
                for r in top_anomalies
            ],
            "subsystem_breakdown": subsys_counts,
        }

        alert_path = os.path.join(out_dir, "alert_payload.json")
        try:
            with open(alert_path, "w", encoding="utf-8") as fp:
                json.dump(alert_payload, fp, indent=2)
        except Exception as e:
            return self.format_error(
                error="Write Error",
                reason=f"Failed to write alert payload: {e}"
            )

        n_alerts = n_warning + n_bad

        result = {
            "status": "ok",
            "report_path": os.path.relpath(report_path, self.base_directory),
            "alert_path": os.path.relpath(alert_path, self.base_directory),
            "n_alerts": n_alerts,
            "overall_verdict": overall,
        }
        return json.dumps(result, separators=(",", ":"), ensure_ascii=False)
