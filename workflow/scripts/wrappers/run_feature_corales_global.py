#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pandas as pd
from scripts.workflow.contracts import ensure_parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Stub rule: corales_global by region.")
    parser.add_argument("--region", required=True)
    parser.add_argument("--ref-grid", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    out = ensure_parent(args.output)
    df = pd.DataFrame(
        {
            "regionid": [args.region] * 3,
            "pixid": [1, 2, 3],
            "x": [0.5, 1.5, 2.5],
            "y": [9.5, 8.5, 7.5],
            "corals": [100.0, 90.0, 80.0],
        }
    )
    df.to_parquet(out, index=False)
    print(f"OK -> {out}")


if __name__ == "__main__":
    main()
