"""Rebuild the three documented R3 clips from private state snapshots."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"outputs/p5a-r3"))
import report_final


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("output",type=Path)
    a=p.parse_args()
    meta=report_final.read(ROOT/"outputs/p5a-r3/frozen_candidates.json")["candidates"]
    selection=report_final.read(ROOT/"outputs/p5a-r3/media-selection.json")
    rows={r["name"]:report_final.read(a.archive/f"{r['name']}.json") for r in meta}
    manifest=report_final.read(ROOT/"outputs/p5a-r3/test_manifest.json")["cases"]
    for group in rows.values():
        for row,case,base in zip(group,manifest,rows["bc"],strict=True):
            assert all(row[field]==case[field] for field in ("seed","orientation","xy"))
            assert row["initial_sha256"]==base["initial_sha256"]
    a.output.mkdir(parents=True)
    report_final.ROOT=a.output
    report_final.RUN=a.archive
    for label,case in selection.items():
        index=next(i for i,r in enumerate(manifest) if
                   (r["seed"],r["orientation"])==(case["seed"],case["orientation"]))
        report_final.render(label,index,case["strategies"],rows,{r["name"]:r for r in meta})


if __name__=="__main__":
    main()
