"""Regression checks for release asset staging."""
import importlib.util
from pathlib import Path

import pytest


def test_release_restore_rejects_missing_and_mismatched_assets(tmp_path, monkeypatch):
    script = Path(__file__).resolve().parents[1] / "scripts/demo.py"
    spec = importlib.util.spec_from_file_location("release_demo", script)
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)

    source = tmp_path / "source"
    source.mkdir()
    source_asset = source / "model.xml"
    source_asset.write_bytes(b"valid asset")
    checkout = tmp_path / "checkout"
    monkeypatch.setattr(demo, "ROOT", checkout)
    monkeypatch.setattr(demo, "assets", lambda: [{
        "path": "model.xml", "required_for": ["quickstart"],
        "acquisition": "local source", "sha256": demo.checksum(source_asset),
    }])
    groups = {"quickstart"}
    with pytest.raises(SystemExit, match="Missing or mismatched"):
        demo.verify(groups)
    demo.restore(source, groups)
    demo.verify(groups)
    destination = checkout / source_asset.name
    destination.write_bytes(b"local user's asset")
    with pytest.raises(SystemExit, match="Refusing to overwrite"):
        demo.restore(source, groups)
    assert destination.read_bytes() == b"local user's asset"
