"""Serving the Version 1.0 result page.

The page is a static file. The only thing done to it on the way out is substituting a fixed allow-list of
**project constants** — the canonical disclaimer and the uncertain-call advice — so that `page.html` cannot
drift from `src.predict.DISCLAIMER` and `src.uncertainty.ADVICE` the way a pasted copy would.

This is a string replacement over a closed set of module constants, not a template engine, so it adds no
dependency. **Nothing derived from a request is ever substituted** (protocol amendment 7, point 2): that is
the one way a static page could become an injection vector, so the substitution map is built from imported
constants only and takes no argument.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from src.predict import DISCLAIMER
from src.uncertainty import ADVICE

PAGE_PATH = Path(__file__).with_name("page.html")     # beside this module, so not CWD-dependent

# The closed substitution set. Keys are the tokens in page.html; values are imported constants.
SUBSTITUTIONS: dict[str, str] = {
    "{{DISCLAIMER}}": DISCLAIMER,
    "{{ADVICE}}": ADVICE,
}


@lru_cache(maxsize=1)
def render_page() -> str:
    """The page as served: the file on disk with the canonical constants substituted in.

    Cached because the result depends only on the file and two module constants, so it is the same for every
    request. A missing or unreadable file is left to raise: the page is shipped alongside this module, so its
    absence is a broken installation rather than a runtime condition worth degrading for.
    """
    html = PAGE_PATH.read_text(encoding="utf-8")
    for token, value in SUBSTITUTIONS.items():
        html = html.replace(token, value)
    return html


def unsubstituted_tokens(html: str) -> list[str]:
    """Any `{{TOKEN}}` left in the rendered page. Empty is the only acceptable answer.

    A token that survives rendering means the page is showing a placeholder where a disclaimer should be,
    which is worse than a missing style: a test asserts this is empty.
    """
    import re

    return re.findall(r"\{\{[A-Z_]+\}\}", html)
