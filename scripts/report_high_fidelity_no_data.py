from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.high_fidelity.reporting import render_high_fidelity_no_data_summary  # noqa: E402

RESULTS_FILE = ROOT / "reports" / "results" / "high_fidelity_no_data_results.json"
SUMMARY_FILE = ROOT / "reports" / "results" / "high_fidelity_no_data_summary.md"


def write_high_fidelity_no_data_report(
    *,
    results_path: Path = RESULTS_FILE,
    output_path: Path = SUMMARY_FILE,
) -> str:
    artifact = json.loads(results_path.read_text(encoding="utf-8"))
    summary = render_high_fidelity_no_data_summary(artifact)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(summary, encoding="utf-8")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Render the high-fidelity no-data markdown summary.")
    parser.add_argument("--input-json", type=Path, default=RESULTS_FILE)
    parser.add_argument("--output-summary", type=Path, default=SUMMARY_FILE)
    args = parser.parse_args()
    write_high_fidelity_no_data_report(
        results_path=args.input_json,
        output_path=args.output_summary,
    )


if __name__ == "__main__":
    main()
