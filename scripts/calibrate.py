#!/usr/bin/env python
"""Fit the served temperature on a calibration read and report the dev rows served at it.

    ./unsloth_uv/bin/python scripts/calibrate.py \
        --calibration /mnt/f/distill_jev_runs/main-cal/rows.json \
        --rows /mnt/f/distill_jev_runs/main-eval/rows.json --out /mnt/f/distill_jev_runs/main-eval \
        --write-run /mnt/f/distill_jev_runs/main

`--write-run` stores the temperature in `head.pt` (the pointer head applies it in eval mode,
as in Kev). The report also carries the out-of-fold estimate, so the in-sample fit can be
checked against rows it did not see.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch

from data import metrics
from data.suites import read_json, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", required=True, help="raw rows.json from the calibration split")
    ap.add_argument("--rows", required=True, help="raw rows.json to report served metrics on")
    ap.add_argument("--out", required=True, help="directory for calibration.json")
    ap.add_argument("--write-run", default="", help="checkpoint dir whose head.pt gets the temperature")
    a = ap.parse_args()
    calibration = read_json(a.calibration)
    rows = read_json(a.rows)
    temperature = metrics.fit_temperature(calibration, **metrics.TEMPERATURE_FIT)
    report = {"temperature": temperature, "fit": metrics.TEMPERATURE_FIT_METHOD,
              "calibration_n": len(metrics.scored_rows(calibration)), "rows_n": len(metrics.scored_rows(rows)),
              "raw": metrics.metrics(rows), "served": metrics.metrics(metrics.served_at(rows, temperature)),
              "cross_validation": metrics.cross_validated_temperature(calibration)}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "calibration.json", report)
    if a.write_run:
        path = Path(a.write_run) / "head.pt"
        head = torch.load(path, map_location="cpu", weights_only=False)
        head["temperature"] = float(temperature)
        torch.save(head, path)
        print(f"[calibrate] temperature {temperature:.3f} written to {path}", flush=True)
    print(json.dumps({"temperature": round(temperature, 4), "raw": {"acc": report["raw"]["acc"], "brier": report["raw"]["brier"], "ece": report["raw"]["ece"]},
                      "served": {"acc": report["served"]["acc"], "brier": report["served"]["brier"], "ece": report["served"]["ece"],
                                 "coverage_at_5pct_error": report["served"]["coverage_at_5pct_error"]},
                      "oof_ece": report["cross_validation"]["out_of_fold"]["ece"], "oof_separated": report["cross_validation"]["separated"]}, indent=2))


if __name__ == "__main__":
    main()
