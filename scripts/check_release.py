"""Check tracked documentation links and the frozen comparison's release inputs."""
import argparse
from collections import Counter
import csv
from hashlib import file_digest
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\((<[^>]+>|[^\s)]+)(?:\s+\"[^\"]*\")?\)")
USER_PATH = re.compile(r"(?:/" r"home/[^/\s]+/|/" r"Users/[^/\s]+/|[A-Za-z]:\\Users\\[^\\\s]+\\)")
PUBLIC_MEDIA = {"docs/media/rgb-expert-demo.gif": 3 * 1024**2, "docs/media/rgb-expert-demo.mp4": 1024**2}
PRIVATE_SUFFIXES = {".pt", ".pth", ".zip", ".npz", ".stl", ".tar", ".tgz", ".gif", ".mp4", ".parquet"}
HEADING = re.compile(r"^#{1,6}\s+(.+?)\s*#*$", re.MULTILINE)
FILES = (
    "README.md", "docs/REPRODUCTION.md", "configs/assets.json", "scripts/fetch_cr3.py",
    "scripts/demo.py", "scripts/check_release.py", "scripts/test_release.py",
    "scripts/presentation.py", "outputs/p4a/summary.json", "outputs/p4c/summary.json",
    "outputs/p5a-r2/summary.json", "outputs/p5a-r3/summary.json",
    "outputs/p5a-r3/frozen_candidates.json", "outputs/p5a-r3/test_manifest.json",
    "outputs/p5a-r3/episodes.csv",
)


def tracked_files():
    result = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True)
    return {p for p in result.stdout.decode().split("\0") if p}


def headings(contents):
    duplicates = Counter()
    for heading in HEADING.findall(contents):
        slug = re.sub(r"[^\w\- ]", "", heading.lower()).replace(" ", "-")
        duplicate = duplicates[slug]
        duplicates[slug] += 1
        yield f"{slug}-{duplicate}" if duplicate else slug


def check_links(root, tracked):
    errors = []
    targets = set()
    for name in sorted(p for p in tracked if p.endswith(".md")):
        lines = (root / name).read_text().splitlines()
        fenced = False
        for number, line in enumerate(lines, 1):
            if line.lstrip().startswith(("```", "~~~")):
                fenced = not fenced
                continue
            if fenced:
                continue
            for raw in LINK.findall(line):
                url = urlsplit(raw.strip("<>"))
                if url.scheme or url.netloc:
                    continue
                path = (root / name).parent / unquote(url.path) if url.path else root / name
                try:
                    target = path.resolve().relative_to(root).as_posix()
                except ValueError:
                    errors.append(f"{name}:{number}: outside checkout: {raw}")
                    continue
                if not path.exists() or (
                        target not in tracked and not any(p.startswith(target + "/") for p in tracked)
                ):
                    errors.append(f"{name}:{number}: untracked target: {raw}")
                    continue
                if url.fragment and target.endswith(".md"):
                    if unquote(url.fragment) not in set(headings(path.read_text())):
                        errors.append(f"{name}:{number}: missing anchor: {raw}")
                targets.add(target)
    return errors, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", action="store_true", help="Use file inventory in a Git-free source archive")
    args = parser.parse_args()
    tracked = ({p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file()}
               if args.archive else tracked_files())
    errors, targets = check_links(ROOT, tracked)
    errors.extend(f"Required release file absent: {p}" for p in FILES
                  if p not in tracked or not (ROOT / p).is_file())
    for name in sorted(tracked):
        path = ROOT / name
        if not path.is_file():
            continue
        if (path.suffix.lower() in PRIVATE_SUFFIXES and name not in PUBLIC_MEDIA) or path.stat().st_size > PUBLIC_MEDIA.get(name, 1024**2):
            errors.append(f"Review tracked asset size/type: {name}")
        if path.suffix in {".py", ".sh", ".md", ".json", ".yml", ".txt"}:
            if USER_PATH.search(path.read_text()):
                errors.append(f"Local home directory embedded in {name}")
    r3 = ROOT / "outputs/p5a-r3"
    with (r3 / "episodes.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    candidates = json.loads((r3 / "frozen_candidates.json").read_text())["candidates"]
    cases = json.loads((r3 / "test_manifest.json").read_text())["cases"]
    keys = {(model["name"], case["seed"], case["orientation"]) for model in candidates for case in cases}
    actual = {(r["candidate"], int(r["seed"]), r["orientation"]) for r in rows}
    if len(rows) != 560 or len(keys) != 560 or actual != keys:
        errors.append("Frozen CSV does not cover the seven candidates and 80 scene pairs exactly once")
    with (r3 / "episodes.csv").open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    expected = next(line.split()[0] for line in (r3 / "SHA256SUMS").read_text().splitlines()
                    if line.endswith("  episodes.csv"))
    if digest != expected:
        errors.append("Frozen CSV SHA-256 differs from R3")
    for entry in json.loads((ROOT / "configs/assets.json").read_text())["assets"]:
        path = ROOT / entry["path"]
        if entry["acquisition"] != "Git checkout":
            continue
        if entry["path"] not in tracked or not path.is_file():
            errors.append(f"Missing tracked manifest asset: {entry['path']}")
        else:
            with path.open("rb") as stream:
                if file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
                    errors.append(f"Manifest digest mismatch: {entry['path']}")
    print(f"{len(targets)} internal link targets; {len(rows)} frozen rows; CSV SHA-256 {digest}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
