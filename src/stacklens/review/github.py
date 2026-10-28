"""GitHub pull-request review integration."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urljoin
from urllib.request import Request, urlopen

from stacklens.config import get_settings
from stacklens.review.service import (
    ChangedFile,
    ReviewReport,
    changed_lines_from_patch,
    review_files,
)

_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_MAX_FILES = 100


class GitHubAPIError(RuntimeError):
    """A safe, user-facing GitHub API failure."""


@dataclass(frozen=True, slots=True)
class PullRequest:
    owner: str
    repository: str
    number: int
    head_sha: str
    title: str
    files: list[ChangedFile]


class GitHubClient:
    """Small GitHub REST client with explicit read and publish operations."""

    def __init__(self, *, token: str, api_url: str) -> None:
        self._token = token
        self._api_url = api_url.rstrip("/") + "/"

    def load_pull_request(self, repository: str, number: int) -> PullRequest:
        owner, repo = _split_repository(repository)
        pull = self._request(f"repos/{owner}/{repo}/pulls/{number}")
        files = self._request(f"repos/{owner}/{repo}/pulls/{number}/files?per_page={_MAX_FILES}")
        if not isinstance(files, list) or len(files) >= _MAX_FILES:
            raise GitHubAPIError("The pull request has too many files for this review.")
        changed: list[ChangedFile] = []
        for item in files:
            if not isinstance(item, dict):
                continue
            path = item.get("filename")
            patch = item.get("patch")
            if not isinstance(path, str) or not isinstance(patch, str):
                continue
            content = self._file_content(owner, repo, path, str(pull["head"]["sha"]))
            changed.append(
                ChangedFile(
                    path=path,
                    diff=patch,
                    content=content,
                    changed_lines=changed_lines_from_patch(patch),
                )
            )
        if not changed:
            raise GitHubAPIError("The pull request has no reviewable text patches.")
        return PullRequest(
            owner=owner,
            repository=repo,
            number=number,
            head_sha=str(pull["head"]["sha"]),
            title=str(pull.get("title", "")),
            files=changed,
        )

    def publish_review(self, pull_request: PullRequest, report: ReviewReport) -> None:
        comments = [
            {
                "path": finding.path,
                "line": finding.line,
                "side": "RIGHT",
                "body": (
                    f"**{finding.severity.value.upper()}** — {finding.title}\n\n"
                    f"{finding.explanation}\n\n"
                    f"Suggested fix: {finding.suggested_fix}\n\n"
                    f"Confidence: {finding.confidence:.0%}"
                ),
            }
            for finding in report.findings
        ]
        payload = {
            "commit_id": pull_request.head_sha,
            "body": report.summary,
            "event": "COMMENT",
            "comments": comments,
        }
        self._request(
            f"repos/{pull_request.owner}/{pull_request.repository}/pulls/"
            f"{pull_request.number}/reviews",
            method="POST",
            payload=payload,
        )

    def _file_content(self, owner: str, repo: str, path: str, ref: str) -> str:
        encoded_path = quote(path, safe="/")
        response = self._request(f"repos/{owner}/{repo}/contents/{encoded_path}?ref={quote(ref)}")
        if not isinstance(response, dict) or response.get("encoding") != "base64":
            raise GitHubAPIError(f"GitHub did not return base64 content for {path}.")
        try:
            return base64.b64decode(str(response["content"])).decode("utf-8", errors="replace")
        except (KeyError, ValueError) as error:
            raise GitHubAPIError(f"GitHub returned invalid content for {path}.") from error

    def _request(self, path: str, *, method: str = "GET", payload: dict | None = None):
        request = Request(
            urljoin(self._api_url, path),
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "stacklens-review",
                "X-GitHub-Api-Version": "2022-11-28",
                **({"Content-Type": "application/json"} if payload is not None else {}),
            },
            method=method,
        )
        try:
            with urlopen(request, timeout=30) as response:
                return json.loads(response.read())
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as error:
            status = getattr(error, "code", "unavailable")
            raise GitHubAPIError(f"GitHub API request failed ({status}).") from error


def build_github_client() -> GitHubClient:
    """Build a client from environment-backed settings."""
    settings = get_settings()
    token = settings.github_token.get_secret_value()
    if not token:
        raise GitHubAPIError("Set GITHUB_TOKEN in .env before using GitHub review.")
    return GitHubClient(token=token, api_url=settings.github_api_url)


def _split_repository(repository: str) -> tuple[str, str]:
    if not _REPOSITORY.fullmatch(repository):
        raise GitHubAPIError("Repository must use the OWNER/REPOSITORY format.")
    return tuple(repository.split("/", 1))  # type: ignore[return-value]


def review_pull_request(
    repository: str,
    number: int,
    *,
    publish: bool = False,
    model=None,
) -> tuple[PullRequest, ReviewReport]:
    """Review a GitHub pull request and optionally publish one review."""
    client = build_github_client()
    pull_request = client.load_pull_request(repository, number)
    report = review_files(pull_request.files, model=model)
    if publish:
        client.publish_review(pull_request, report)
    return pull_request, report
