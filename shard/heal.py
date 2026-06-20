"""mid-generation recovery decision logic.

recovery is stop, swap, replay, continue. the caller localizes the failed stage,
swaps in a warm spare for the same layer block, resets every live stage's kv
state, replays the committed prefix, checks that replay predicts the known driver
token, then resumes decoding.

this module is pure decision logic: no torch, sockets, or network work lives
here. the phase0 engine and libp2p sidecar call these helpers and perform the
real node operations.
"""

from dataclasses import dataclass
from typing import Protocol


class HealError(Exception):
    """recovery cannot proceed safely."""


@dataclass(frozen=True)
class StageReport:
    stage_id: int
    reachable: bool
    last_recv_chunk: int | None
    last_fwd_chunk: int | None


@dataclass(frozen=True)
class FailureLocation:
    stage_id: int
    reason: str


@dataclass(frozen=True)
class ReplayPlan:
    replay_prefix: list[int]
    expected_next: int | None
    resume_pos: int


@dataclass(frozen=True)
class RecoveryOutcome:
    failed_stage: int
    reason: str
    resume_pos: int
    cur: int | None


class SwarmOps(Protocol):
    def probe(self) -> list[StageReport]: ...
    def has_spare(self, stage_id: int) -> bool: ...
    def activate_spare(self, stage_id: int) -> str: ...
    def rewire(self, predecessor_stage_id: int, new_endpoint: str) -> None: ...
    def reset_all(self) -> None: ...
    def replay(self, prefix: list[int]) -> int | None: ...
    def resume(self, resume_pos: int, cur: int | None) -> None: ...


def locate_failure(reports: list[StageReport]) -> FailureLocation | None:
    """find the first clear failed stage from control and verify progress reports."""
    for report in reports:
        if not report.reachable:
            return FailureLocation(report.stage_id, "unreachable")

    for upstream, downstream in zip(reports, reports[1:]):
        sent = upstream.last_fwd_chunk
        received = downstream.last_recv_chunk
        if sent is not None and (received is None or received < sent):
            return FailureLocation(downstream.stage_id, "gap")

    return None


def plan_replay(prompt_ids: list[int], committed_out: list[int]) -> ReplayPlan:
    """build the prefix replay that must reproduce the committed driver token."""
    if not prompt_ids:
        raise HealError("empty prompt")

    if not committed_out:
        return ReplayPlan(
            replay_prefix=list(prompt_ids),
            expected_next=None,
            resume_pos=len(prompt_ids),
        )

    return ReplayPlan(
        replay_prefix=list(prompt_ids) + list(committed_out[:-1]),
        expected_next=committed_out[-1],
        resume_pos=len(prompt_ids) + len(committed_out) - 1,
    )


def run_recovery(
    ops: SwarmOps,
    prompt_ids: list[int],
    committed_out: list[int],
) -> RecoveryOutcome:
    """run the ordered recovery decisions against injected swarm operations."""
    reports = ops.probe()
    loc = locate_failure(reports)
    if loc is None:
        raise HealError("no failure localized")

    if not ops.has_spare(loc.stage_id):
        raise HealError(f"no warm spare for stage {loc.stage_id}")

    new_ep = ops.activate_spare(loc.stage_id)
    predecessor = loc.stage_id - 1
    ops.rewire(predecessor, new_ep)
    ops.reset_all()

    plan = plan_replay(prompt_ids, committed_out)
    got = ops.replay(plan.replay_prefix)
    if plan.expected_next is not None and got != plan.expected_next:
        raise HealError(
            f"replay mismatch: got {got!r} expected {plan.expected_next!r}"
        )

    cur = plan.expected_next if plan.expected_next is not None else got
    ops.resume(plan.resume_pos, cur)
    return RecoveryOutcome(
        failed_stage=loc.stage_id,
        reason=loc.reason,
        resume_pos=plan.resume_pos,
        cur=cur,
    )
