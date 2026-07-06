"""safe_kill must kill the target but NEVER its own caller/ancestors — the self-match property that
raw `pkill -f` lacks (a kill-then-relaunch one-liner whose command line contains the pattern kills its
own shell before the relaunch runs). Exercised against real processes.

NOTE on liveness: we never probe our own children with os.kill(pid, 0) — a SIGKILLed-but-unreaped
child is a ZOMBIE that still answers signal-0 as 'alive'. We reap via Popen.wait() and read the
returncode (negative == killed by that signal) instead.
"""
import os
import signal
import subprocess

import pytest

SAFE_KILL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "phase0", "safe_kill.sh")


def _target(marker):
    # a process whose argv IS the marker (exec -a sets argv[0]); pgrep -f matches it
    return subprocess.Popen(["bash", "-c", f"exec -a {marker} sleep 30"])


def _reap(p, timeout=5):
    """Return the child's exit status (None if still running), reaping the zombie so a killed child
    doesn't linger as 'alive'."""
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return None


def test_kills_matching_target():
    marker = "SAFEKILL_TARGET_zX7"
    p = _target(marker)
    try:
        assert p.poll() is None
        subprocess.run(["bash", SAFE_KILL, marker], capture_output=True, text=True)
        rc = _reap(p)
        assert rc is not None and rc < 0, f"target not killed (rc={rc})"
    finally:
        if p.poll() is None:
            p.kill(); p.wait()


def test_does_not_kill_its_own_caller():
    """THE property: a caller whose OWN command line contains the pattern must survive, while a
    separate real target with the same pattern still dies. Reproduces the exact footgun."""
    marker = "SAFEKILL_SELFMATCH_q9W"
    target = _target(marker)
    try:
        # a shell whose argv literally contains `marker` (like `ssh box "...marker..."`) calls
        # safe_kill marker then prints SURVIVED. Raw `pkill -f` would kill this shell first.
        script = f"echo START_{marker}; bash {SAFE_KILL} {marker}; echo SURVIVED_{marker}"
        out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, timeout=15)
        assert f"SURVIVED_{marker}" in out.stdout, f"caller self-killed; stdout={out.stdout!r}"
        rc = _reap(target)
        assert rc is not None and rc < 0, "separate target survived (over-exclusion)"
    finally:
        if target.poll() is None:
            target.kill(); target.wait()


def test_nothing_to_kill_is_success():
    out = subprocess.run(["bash", SAFE_KILL, "SAFEKILL_NO_SUCH_PROC_abc123"], capture_output=True, text=True)
    assert out.returncode == 0
    assert "0 proc" in out.stdout


def test_signal_flag_routes_the_signal():
    """-s selects the signal; default is KILL. (Message reflects the routed signal.)"""
    m = "SAFEKILL_NOPROC_sig_v2"
    term = subprocess.run(["bash", SAFE_KILL, m, "-s", "TERM"], capture_output=True, text=True)
    kill = subprocess.run(["bash", SAFE_KILL, m], capture_output=True, text=True)
    assert "SIGTERM" in term.stdout and "SIGKILL" in kill.stdout


def test_term_is_catchable_not_a_hard_kill():
    """A process that traps TERM and exits cleanly proves -s TERM sends a real, catchable SIGTERM
    (not SIGKILL). Uses a wait-loop shell (no exec, so the trap stays installed) whose cmdline
    carries the marker."""
    m = "SAFEKILL_TRAP_m4R"
    # bash keeps the TERM trap (no exec); the marker rides in a trailing comment
    p = subprocess.Popen(["bash", "-c", f"trap 'exit 42' TERM; while true; do sleep 0.2; done  # {m}"])
    try:
        import time
        time.sleep(0.4)
        subprocess.run(["bash", SAFE_KILL, m, "-s", "TERM"], capture_output=True, text=True)
        rc = _reap(p)
        assert rc == 42, f"TERM not delivered as a catchable signal (rc={rc})"
    finally:
        if p.poll() is None:
            p.kill(); p.wait()


def test_missing_pattern_errors():
    out = subprocess.run(["bash", SAFE_KILL], capture_output=True, text=True)
    assert out.returncode == 2 and "usage" in out.stderr
