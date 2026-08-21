"""Pillar 3: Real-World Live Grounding & Dynamic Web Fallback.

When the internal knowledge base lacks information for real-world queries,
dynamically fetches verified live web summaries (via Wikipedia / DuckDuckGo API)
to ground the response and prevent knowledge-cutoff staleness.
"""

import requests
from dataclasses import dataclass
from rich.console import Console

console = Console()


@dataclass
class LiveGroundingResult:
    found: bool
    source_title: str
    content: str
    url: str


class WebGroundingRetriever:
    """Retrieves live real-world knowledge on-the-fly when local context is insufficient."""

    def search_live_grounding(self, query: str) -> LiveGroundingResult:
        """Fetch summary from Wikipedia REST API as zero-dependency live grounding."""
        try:
            # Clean query for search
            clean_q = query.replace("What is", "").replace("Who is", "").replace("?", "").strip()
            if not clean_q:
                return LiveGroundingResult(found=False, source_title="", content="", url="")

            url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(clean_q)}"
            resp = requests.get(url, timeout=3.5, headers={"User-Agent": "SelfHealingRAG/1.0"})

            if resp.status_code == 200:
                data = resp.json()
                extract = data.get("extract", "")
                title = data.get("title", clean_q)
                page_url = data.get("content_urls", {}).get("desktop", {}).get("page", "")

                if extract and len(extract) > 40:
                    console.print(f"[bold green]🌐 Live Grounding: Retrieved external context for '{title}'[/bold green]")
                    return LiveGroundingResult(
                        found=True,
                        source_title=f"Wikipedia: {title}",
                        content=extract,
                        url=page_url,
                    )
        except Exception as e:
            console.print(f"[dim]Live grounding fallback skipped ({e})[/dim]")

        return LiveGroundingResult(found=False, source_title="", content="", url="")


web_grounding = WebGroundingRetriever()
