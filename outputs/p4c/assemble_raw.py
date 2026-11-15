"""Assemble disjoint recorded seed shards into a new raw dataset."""
import argparse
import json
from pathlib import Path


def assemble(out, shards):
    out.mkdir(parents=True)
    merged = []
    seen = set()
    schema = (shards[0] / "schema.json").read_text()
    for shard in shards:
        assert (shard / "schema.json").read_text() == schema
        for episode in json.loads((shard / "manifest.json").read_text()):
            case = episode["seed"], episode["orientation"]
            assert case not in seen
            seen.add(case)
            source = shard / episode["id"]
            identifier = f"episode-{len(merged):03d}"
            (out / identifier).symlink_to(source.resolve(), target_is_directory=True)
            merged.append(dict(episode, id=identifier))
    assert len(merged) == 38
    (out / "schema.json").write_text(schema)
    (out / "manifest.json").write_text(json.dumps(merged, indent=2)+"\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("output", type=Path)
    p.add_argument("shards", type=Path, nargs="+")
    a = p.parse_args()
    assemble(a.output, a.shards)
