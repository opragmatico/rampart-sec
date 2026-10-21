"""GitHub Issues as the external tracker. Enabled by RAMPART_TICKET_BACKEND=github plus
RAMPART_GITHUB_REPO=owner/name and GITHUB_TOKEN."""

from __future__ import annotations

from textwrap import dedent

from ..http import HttpClient
from .store import Ticket

ISSUE_TITLE = "[{priority}] {cve_id} on {system} (due {due_date})"
# Dedented once here, before the ticket's fields are filled in: a multi-line summary would otherwise
# defeat dedent's common-indent detection.
ISSUE_BODY = dedent(
    """\
    **CVE:** {cve_id}
    **System:** {system}
    **Priority:** {priority}
    **Due:** {due_date}
    **Owner team:** {owner_team}

    {summary}

    _Opened by Rampart ({id}); approved by an operator._"""
)


class GitHubIssues:
    def __init__(self, http: HttpClient, repo: str, token: str):
        self.http, self.repo, self.token = http, repo, token

    async def create(self, t: Ticket) -> str:
        fields = t.to_dict() | {"owner_team": t.owner_team or "n/a"}
        res = await self.http.post_json(
            f"https://api.github.com/repos/{self.repo}/issues",
            {
                "title": ISSUE_TITLE.format(**fields),
                "body": ISSUE_BODY.format(**fields),
                "labels": ["security", "vulnerability", t.priority.lower()],
            },
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"},
            timeout=30,
            retries=3,
        )
        return res.body["html_url"]
