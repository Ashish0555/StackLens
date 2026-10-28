"""Local diff review parsing and evidence-bound findings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from stacklens.review.service import ReviewReport, parse_unified_diff, review_diff

DIFF = """\
diff --git a/example.py b/example.py
index 0000000..1111111 100644
--- a/example.py
+++ b/example.py
@@ -1,2 +1,3 @@
 def answer():
+    return "changed"
     return "old"
"""


def test_parse_unified_diff_loads_changed_lines(tmp_path: Path) -> None:
    (tmp_path / "example.py").write_text('def answer():\n    return "changed"\n', encoding="utf-8")

    files = parse_unified_diff(DIFF, tmp_path)

    assert files[0].path == "example.py"
    assert files[0].changed_lines == [2]
    assert 'return "changed"' in files[0].content


def test_parse_rejects_path_traversal(tmp_path: Path) -> None:
    diff = DIFF.replace("example.py", "../outside.py")

    with pytest.raises(ValueError, match="Unsafe diff path"):
        parse_unified_diff(diff, tmp_path)


def test_review_rejects_finding_outside_changed_lines(tmp_path: Path) -> None:
    (tmp_path / "example.py").write_text('def answer():\n    return "changed"\n', encoding="utf-8")

    class FakeModel:
        def invoke(self, _prompt: str) -> ReviewReport:
            return ReviewReport.model_validate(
                {
                    "summary": "Potential issue.",
                    "findings": [
                        {
                            "path": "example.py",
                            "line": 1,
                            "severity": "major",
                            "confidence": 0.9,
                            "title": "Issue",
                            "explanation": "The changed behavior may be incorrect.",
                            "suggested_fix": "Review the implementation.",
                        }
                    ],
                }
            )

    with pytest.raises(ValueError, match="non-changed line"):
        review_diff(DIFF, tmp_path, model=FakeModel())


def test_review_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ReviewReport.model_validate({"summary": "x", "unexpected": True})
