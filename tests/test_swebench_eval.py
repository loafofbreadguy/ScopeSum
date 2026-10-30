"""Exclusion and patch-parsing behaviour for the SWE-bench Lite evaluation."""

from scopesum.data import exclusion_reason
from scopesum.eval_localize import parse_patch

SWEBENCH_REPOS = ["django", "sympy", "pytest"]


def test_exact_exclusion_only_matches_whole_repo_name():
    excl = ["fastapi", "requests", "rich", "httpx"]
    assert exclusion_reason("django/django", "x.py", "", excl, excluded_exact=SWEBENCH_REPOS) == "repo"
    assert exclusion_reason("someone/django", "x.py", "", excl, excluded_exact=SWEBENCH_REPOS) == "repo"
    assert exclusion_reason("me/django-blog", "x.py", "", excl, excluded_exact=SWEBENCH_REPOS) is None
    assert exclusion_reason("me/pytest-cov", "x.py", "", excl, excluded_exact=SWEBENCH_REPOS) is None


def test_swebench_style_patch():
    patch = (
        "diff --git a/django/db/models/query.py b/django/db/models/query.py\n"
        "--- a/django/db/models/query.py\n"
        "+++ b/django/db/models/query.py\n"
        "@@ -10,3 +10,3 @@ class QuerySet:\n"
        "     def a(self):\n"
        "-        return 1\n"
        "+        return 2\n"
        "     x = 1\n"
    )
    assert parse_patch(patch) == {"django/db/models/query.py": {11}}
