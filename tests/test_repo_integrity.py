"""Repo-integrity guards: the data package must never silently drop out of git.

Background: a bare `data/` pattern in .gitignore matched ANY directory named
data — including src/ssm_phylo/data/ — which silently excluded the entire
data package from commits (train.py, tests, and README all reference it).
These tests make that failure mode loud.
"""
import subprocess

import pytest

REPO_ROOT = subprocess.run(
    ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=False
).stdout.strip()


def _in_git_repo() -> bool:
    return bool(REPO_ROOT)


def test_data_package_is_tracked():
    if not _in_git_repo():
        pytest.skip("not inside a git repository")
    out = subprocess.run(
        ["git", "ls-files"], capture_output=True, text=True, check=True
    ).stdout
    for path in (
        "src/ssm_phylo/data/__init__.py",
        "src/ssm_phylo/data/simulation.py",
        "src/ssm_phylo/data/datasets.py",
    ):
        assert path in out.splitlines(), (
            f"{path} is NOT tracked by git — check .gitignore for a bare "
            "`data/` pattern swallowing the package"
        )


def test_no_bare_data_ignore():
    gitignore = (REPO_ROOT if _in_git_repo() else __file__ and ".")
    gitignore_path = __import__("pathlib").Path(gitignore) / ".gitignore"
    if not gitignore_path.exists():
        pytest.skip(".gitignore not found")
    lines = gitignore_path.read_text().splitlines()
    bare_data = [ln for ln in lines if ln.strip() == "data/"]
    assert not bare_data, "bare `data/` pattern present — it swallows src/ssm_phylo/data/"
    assert any(ln.strip() == "/data/" for ln in lines) or any(
        ln.strip() == "!src/ssm_phylo/data/" for ln in lines
    ), "missing root-anchored /data/ or the negation !src/ssm_phylo/data/"
