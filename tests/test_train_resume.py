"""Phase 3 training tests: SIGTERM/resume protocol, atomicity, non-blocking push.

Runs the real `python -m ssm_phylo.train --smoke` entrypoint as a subprocess
with a local tmp dir standing in for the Drive mirror (CKPT_DIR) and a scratch
LOCAL_CKPT_DIR. No GPU, no Drive, no fused kernels needed.
"""
import csv
import os
import subprocess
import sys
import time

SMOKE_ARGS = ["--smoke", "--smoke-step-delay", "0.05", "--max-steps", "100"]


def _env(tmp_path, extra=None):
    env = os.environ.copy()
    env.update(
        {
            "LOCAL_CKPT_DIR": str(tmp_path / "local_ckpts"),
            "LOCAL_DATA_DIR": str(tmp_path / "local_data"),
            "CKPT_DIR": str(tmp_path / "drive_ckpts"),
        }
    )
    if extra:
        env.update(extra)
    return env


def _run(env, *extra, timeout=300):
    proc = subprocess.Popen(
        [sys.executable, "-m", "ssm_phylo.train", *SMOKE_ARGS, *extra],
        env=env, cwd=str(_repo_root()), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    out, _ = proc.communicate(timeout=timeout)
    return proc.returncode, out


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _wait_for(path, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(0.1)
    return False


def _read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _no_partial_files(*dirs):
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            assert not name.startswith(".tmp-"), f"partial file left behind: {d}/{name}"
            assert not name.endswith(".part"), f"partial file left behind: {d}/{name}"
            assert not name.endswith(".tmp"), f"partial file left behind: {d}/{name}"


# --------------------------------------------------------------------------- #
def test_smoke_clean_run_checkpoints_and_mirrors(tmp_path):
    env = _env(tmp_path)
    rc, out = _run(env)
    assert rc == 0, out[-3000:]
    local = tmp_path / "local_ckpts"
    drive = tmp_path / "drive_ckpts"
    assert os.path.exists(local / "latest.pt")
    assert os.path.exists(drive / "latest.pt")          # mirrored
    assert os.path.exists(local / "best.pt")
    assert os.path.exists(drive / "best.pt")
    numbered_drive = [n for n in os.listdir(drive) if n.startswith("ckpt-") and n.endswith(".pt")]
    assert len(numbered_drive) <= 2                      # drive keeps <= 2 numbered
    _no_partial_files(str(local), str(drive))
    # smoke trains >= 3 steps
    steps = [int(r["step"]) for r in _read_csv(tmp_path / "local_data" / "metrics.csv") if r["step"]]
    assert max(steps) >= 3


def test_sigterm_saves_interrupted_and_resume_continues_steps(tmp_path):
    env = _env(tmp_path)
    proc = subprocess.Popen(
        [sys.executable, "-m", "ssm_phylo.train", *SMOKE_ARGS],
        env=env, cwd=str(_repo_root()), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True,
    )
    local = tmp_path / "local_ckpts"
    drive = tmp_path / "drive_ckpts"
    assert _wait_for(str(local / "ckpt-1.pt")), "train never wrote ckpt-1.pt"
    time.sleep(0.3)
    proc.send_signal(subprocess.signal.SIGTERM)
    out, _ = proc.communicate(timeout=180)
    assert proc.returncode == 0, f"train exited {proc.returncode}: {out[-3000:]}"
    # emergency checkpoint saved locally AND pushed to the Drive mirror
    assert os.path.exists(local / "ckpt-interrupted.pt")
    assert os.path.exists(drive / "ckpt-interrupted.pt")
    assert os.path.exists(local / "ckpt-interrupted.meta.json")
    _no_partial_files(str(local), str(drive))

    # resume: global_step continues, never restarts
    rc, out2 = _run(env, "--resume", "latest", "--max-steps", "20")
    assert rc == 0, out2[-3000:]
    assert "RESUMED from step" in out2, out2[-3000:]
    assert "pulled from Drive" in out2, out2[-3000:]
    import re as _re

    resumed_step = int(_re.search(r"RESUMED from step (\d+)", out2).group(1))
    steps = [int(r["step"]) for r in _read_csv(tmp_path / "local_data" / "metrics.csv") if r["step"]]
    assert len(steps) == len(set(steps)), "steps must never repeat (no restart/double-count)"
    assert max(steps) == 20, f"expected continuation to step 20, got {max(steps)}"
    # every continuation step (resumed_step+1 .. 20) must be present; a restarted
    # run would either repeat steps or never exceed resumed_step
    assert set(range(resumed_step + 1, 21)) <= set(steps), (
        f"missing continuation steps after resume from {resumed_step}"
    )
    _no_partial_files(str(local), str(drive))


def test_background_push_never_blocks_training(tmp_path):
    push_log = tmp_path / "push_count.log"
    slow_push = tmp_path / "slow_push.sh"
    slow_push.write_text(
        f"#!/bin/bash\nsleep 5\necho x >> {push_log}\nexit 0\n"
    )
    os.chmod(slow_push, 0o755)
    env = _env(tmp_path, {"SSM_PHYLO_PUSH_CMD": str(slow_push)})
    t0 = time.time()
    rc, out = _run(env, "--max-steps", "4", "--smoke-step-delay", "0.2")
    wall = time.time() - t0
    assert rc == 0, out[-3000:]
    n_pushes = len(push_log.read_text().splitlines()) if push_log.exists() else 0
    assert n_pushes == 1, f"expected exactly 1 push (skip-if-running), got {n_pushes}"
    # 4 saves at 0.2s/step + one 5s push wait + build/sim overhead; blocking
    # per-save would be ~20s+ (4 x 5s) — 15s proves pushes never block
    assert wall < 15, f"training blocked on push (wall={wall:.1f}s)"


def test_help_exits_zero():
    proc = subprocess.run(
        [sys.executable, "-m", "ssm_phylo.train", "--help"],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0
    for flag in ("--resume", "--no-drive-sync", "--precision", "--ckpt-dir",
                 "--smoke", "--log", "--max-steps", "--grad-checkpointing"):
        assert flag in proc.stdout
