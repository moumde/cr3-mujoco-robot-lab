"""Check release documentation and GitHub anchor handling."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_release import check_links, headings


def test_markdown_validation(tmp_path):
    names = {"README.md", "docs/guide.md"}
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(
        "[read](docs/guide.md#quick-start-1)\n[missing](docs/lost.md)\n"
        "```md\n[example](example.md)\n```\n"
    )
    guide = "# Quick Start\n## Quick Start\n"
    (tmp_path / "docs/guide.md").write_text(guide)
    assert list(headings(guide)) == ["quick-start", "quick-start-1"]
    errors, destinations = check_links(tmp_path, names)
    assert len(errors) == 1 and "docs/lost.md" in errors[0]
    assert destinations == {"docs/guide.md"}
