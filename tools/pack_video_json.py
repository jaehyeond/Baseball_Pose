#!/usr/bin/env python3
"""
Pack per-frame 2D predictions (one JSON per frame) into a single
results_output_video.json that triangulate_video.py can read.

Input directory is expected to contain files like:
  00000.json, 00001.json, ...
Each file should be a JSON list of instances, where each instance has:
  - keypoints: Jx2 (or Jx3 with conf in the 3rd, only first 2 used later)
  - keypoint_scores (or scores): J

Output format:
  {
    "frames": [
      {"instances": [...]},  # from 00000.json
      {"instances": [...]},  # from 00001.json
      ...
    ]
  }

Usage:
  python tools/pack_video_json.py --src-dir D:\\mmpose\\results_2d_camera1
  # (optional) custom out path
  python tools/pack_video_json.py --src-dir <dir> --out-file <dir>\\results_output_video.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import List


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True, help="Directory with per-frame JSON files")
    ap.add_argument(
        "--out-file",
        default=None,
        help="Output JSON path (default: <src-dir>/results_output_video.json)",
    )
    ap.add_argument(
        "--pattern",
        default="*.json",
        help="Glob pattern for frame jsons (default: *.json)",
    )
    args = ap.parse_args()

    src = os.path.abspath(args.src_dir)
    out = args.out_file or os.path.join(src, "results_output_video.json")

    files: List[str] = sorted(
        f for f in glob.glob(os.path.join(src, args.pattern))
        if os.path.basename(f).lower() != "results_output_video.json"
    )
    if not files:
        raise FileNotFoundError(f"No frame jsons found in {src} with pattern {args.pattern}")

    frames = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Expect a list of instances per frame
        if isinstance(data, list):
            frames.append({"instances": data})
        elif isinstance(data, dict) and "instances" in data:
            frames.append({"instances": data["instances"]})
        else:
            raise ValueError(f"Unsupported JSON shape in {fp}. Expected list[instances] or {'instances': ...}.")

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"frames": frames}, f, ensure_ascii=False, indent=2)

    print(f"Wrote {out} with {len(frames)} frames from {src}")


if __name__ == "__main__":
    main()

