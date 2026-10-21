"""GitHub Issues as the external tracker. Enabled by RAMPART_TICKET_BACKEND=github plus
RAMPART_GITHUB_REPO=owner/name and GITHUB_TOKEN."""

from __future__ import annotations

from ..http import HttpClient
from .store import Ticket


class GitHubIssues:
    def __init__(self, http: HttpClient, repo: str, token: str):
        self.http, self.repo, self.token = http, repo, token

    async def create(self, t: Ticket) -> str:
        body = (
            f"**CVE:** {t.cve_id}\n**System:** {t.system}\n**Priority:** {t.priority}\n"
            f"**Due:** {t.due_date}\n**Owner team:** {t.owner_team or 'n/a'}\n\n{t.summary}\n\n"
            f"_Opened by Rampart ({t.id}); approved by an operator._"
        )
        res = await self.http.post_json(
            f"https://api.github.com/repos/{self.repo}/issues",
            {
                "title": f"[{t.priority}] {t.cve_id} on {t.system} (due {t.due_date})",
                "body": body,
                "labels": ["security", "vulnerability", t.priority.lower()],
            },
            headers={"Authorization": f"Bearer {self.token}", "Accept": "application/vnd.github+json"},
            timeout=30,
            retries=3,
        )
        return res.body["html_url"]
