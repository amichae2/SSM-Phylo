"""Import contract: the package must import without `kernels` installed.

`kernels` (HuggingFace's fused kernels) is an optional speedup auto-detected
by transformers 5.x. If it is absent at runtime, the pipeline falls back to
transformers' eager Mamba. This test simulates absence by stubbing sys.modules
BEFORE importing ssm_phylo.
"""

import sys


def test_import_without_kernels(monkeypatch):
    """Importing ssm_phylo must not require the `kernels` package."""
    monkeypatch.setitem(sys.modules, "kernels", None)

    import ssm_phylo

    assert ssm_phylo.__version__
    assert "kernels" not in sys.modules or sys.modules["kernels"] is None


def test_submodules_import_without_kernels(monkeypatch):
    """Every module imports cleanly with `kernels` stubbed out."""
    monkeypatch.setitem(sys.modules, "kernels", None)

    import importlib

    for mod in [
        "ssm_phylo",
        "ssm_phylo.data",
        "ssm_phylo.data.simulation",
        "ssm_phylo.data.datasets",
        "ssm_phylo.models",
        "ssm_phylo.models.encoder",
        "ssm_phylo.models.head",
        "ssm_phylo.models.losses",
        "ssm_phylo.losses",
        "ssm_phylo.train",
        "ssm_phylo.infer",
        "ssm_phylo.build_tree",
        "ssm_phylo.evaluate",
    ]:
        importlib.import_module(mod)


def test_placeholder_mains_return_zero():
    """Phase-0 stubs: every main() prints its purpose and returns 0.

    train/infer/build_tree/evaluate are real CLIs since Phases 3-4 (their
    main() parses argv), so they are excluded here.
    """
    from ssm_phylo import losses

    assert losses.main() == 0
