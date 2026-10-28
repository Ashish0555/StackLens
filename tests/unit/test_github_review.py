"""GitHub pull-request review client tests."""

import base64
import json
from collections.abc import Iterator
from unittest.mock import patch

import pytest

from stacklens.review.github import GitHubAPIError, GitHubClient
from stacklens.review.service import ReviewReport


class FakeResponse:
    def __init__(self, value: object) -> None:
        self.value = value

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.value).encode()


def test_load_pull_request_fetches_patch_and_head_content() -> None:
    content = base64.b64encode(b"def answer():\n    return 42\n").decode()
    responses: Iterator[object] = iter(
        [
            {"title": "Improve answer", "head": {"sha": "abc123"}},
            [{"filename": "example.py", "patch": "@@ -1,1 +1,2 @@\n+    return 42\n"}],
            {"encoding": "base64", "content": content},
        ]
    )

    with patch(
        "stacklens.review.github.urlopen",
        side_effect=lambda _request, timeout: FakeResponse(next(responses)),
    ):
        pull_request = GitHubClient(
            token="token",
            api_url="https://api.github.com",
        ).load_pull_request("owner/repo", 7)

    assert pull_request.head_sha == "abc123"
    assert pull_request.files[0].path == "example.py"
    assert pull_request.files[0].changed_lines == [1]


def test_publish_review_sends_inline_comments() -> None:
    captured: list[dict] = []

    def fake_open(request, timeout):
        captured.append(json.loads(request.data))
        return FakeResponse({"id": 1})

    from stacklens.review.github import PullRequest
    from stacklens.review.service import ChangedFile, ReviewFinding, Severity

    pull_request = PullRequest(
        owner="owner",
        repository="repo",
        number=7,
        head_sha="abc123",
        title="Improve answer",
        files=[ChangedFile(path="example.py", diff="+x", content="x", changed_lines=[2])],
        skipped_files=[],
    )
    report = ReviewReport(
        summary="Please fix the issue.",
        findings=[
            ReviewFinding(
                path="example.py",
                line=2,
                severity=Severity.MAJOR,
                confidence=0.9,
                title="Unsafe change",
                explanation="The changed line is unsafe.",
                suggested_fix="Validate the input.",
            )
        ],
    )

    with patch("stacklens.review.github.urlopen", side_effect=fake_open):
        GitHubClient(token="token", api_url="https://api.github.com").publish_review(
            pull_request, report
        )

    assert captured[0]["commit_id"] == "abc123"
    assert captured[0]["comments"][0]["path"] == "example.py"
    assert captured[0]["comments"][0]["line"] == 2


def test_repository_format_is_validated() -> None:
    with pytest.raises(GitHubAPIError, match="OWNER/REPOSITORY"):
        GitHubClient(token="token", api_url="https://api.github.com").load_pull_request("bad", 1)
