"""Import contract: the package must import without mamba-ssm installed.

mamba-ssm is an optional fused-kernel dependency. If it is absent at runtime
(imports wrapped in try/except), the pipeline falls back to HuggingFace
transformers' eager Mamba. This test simulates absence by stubbing
sys.modules BEFORE importing ssm_phylo.
"""

import sys


def test_import_without_mamba_ssm(monkeypatch):
    """Importing ssm_phylo must not require the mamba_ssm package."""
    monkeypatch.setitem(sys.modules, "mamba_ssm", None)
    monkeypatch.setitem(sys.modules, "mamba_ssm.utils", None)
    monkeypatch.setitem(sys.modules, "causal_conv1d", None)

    import ssm_phylo

    assert ssm_phylo.__version__
    assert "mamba_ssm" not in sys.modules or sys.modules["mamba_ssm"] is None


def test_submodules_import_without_mamba_ssm(monkeypatch):
    """Every placeholder module imports cleanly with mamba-ssm stubbed out."""
    monkeypatch.setitem(sys.modules, "mamba_ssm", None)

    import importlib

    for mod in [
        "ssm_phylo",
        "ssm_phylo.data",
        "ssm_phylo.data.simulation",
        "ssm_phylo.data.datasets",
        "ssm_phylo.models",
        "ssm_phylo.models.checkpoint_compat",
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

    train is a real CLI since Phase 3 (its main() parses argv), so it is
    excluded here.
    """
    from ssm_phylo import build_tree, evaluate, infer, losses

    for mod in [losses, infer, build_tree, evaluate]:
        assert mod.main() == 0
