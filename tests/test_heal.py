import unittest

from shard.heal import (
    FailureLocation,
    HealError,
    RecoveryOutcome,
    StageReport,
    locate_failure,
    plan_replay,
    run_recovery,
)


class RecordingOps:
    def __init__(
        self,
        reports,
        *,
        spare=True,
        replay_result=None,
        endpoint="spare-stage",
    ):
        self.reports = reports
        self.spare = spare
        self.replay_result = replay_result
        self.endpoint = endpoint
        self.calls = []

    def probe(self):
        self.calls.append(("probe",))
        return self.reports

    def has_spare(self, stage_id):
        self.calls.append(("has_spare", stage_id))
        return self.spare

    def activate_spare(self, stage_id):
        self.calls.append(("activate_spare", stage_id))
        return self.endpoint

    def rewire(self, predecessor_stage_id, new_endpoint):
        self.calls.append(("rewire", predecessor_stage_id, new_endpoint))

    def reset_all(self):
        self.calls.append(("reset_all",))

    def replay(self, prefix):
        self.calls.append(("replay", list(prefix)))
        return self.replay_result

    def resume(self, resume_pos, cur):
        self.calls.append(("resume", resume_pos, cur))


class LocateFailureTests(unittest.TestCase):
    def test_first_unreachable_wins(self):
        reports = [
            StageReport(2, True, 0, 0),
            StageReport(7, False, 0, 0),
            StageReport(8, False, None, None),
        ]

        self.assertEqual(locate_failure(reports), FailureLocation(7, "unreachable"))

    def test_gap_detection_when_downstream_lags(self):
        reports = [
            StageReport(0, True, 1, 4),
            StageReport(1, True, 2, 2),
        ]

        self.assertEqual(locate_failure(reports), FailureLocation(1, "gap"))

    def test_gap_detection_when_downstream_received_none(self):
        reports = [
            StageReport(3, True, None, 5),
            StageReport(4, True, None, None),
        ]

        self.assertEqual(locate_failure(reports), FailureLocation(4, "gap"))

    def test_all_consistent_returns_none(self):
        reports = [
            StageReport(0, True, None, 0),
            StageReport(1, True, 0, 3),
            StageReport(2, True, 3, None),
        ]

        self.assertIsNone(locate_failure(reports))

    def test_head_unreachable(self):
        reports = [
            StageReport(0, False, None, None),
            StageReport(1, True, None, None),
        ]

        self.assertEqual(locate_failure(reports), FailureLocation(0, "unreachable"))


class PlanReplayTests(unittest.TestCase):
    def test_normal_multi_token_output(self):
        plan = plan_replay([10, 11], [20, 21, 22])

        self.assertEqual(plan.replay_prefix, [10, 11, 20, 21])
        self.assertEqual(plan.expected_next, 22)
        self.assertEqual(plan.resume_pos, 4)

    def test_single_token_output(self):
        plan = plan_replay([10, 11], [20])

        self.assertEqual(plan.replay_prefix, [10, 11])
        self.assertEqual(plan.expected_next, 20)
        self.assertEqual(plan.resume_pos, 2)

    def test_empty_output(self):
        plan = plan_replay([10, 11], [])

        self.assertEqual(plan.replay_prefix, [10, 11])
        self.assertIsNone(plan.expected_next)
        self.assertEqual(plan.resume_pos, 2)

    def test_empty_prompt_raises(self):
        with self.assertRaisesRegex(HealError, "empty prompt"):
            plan_replay([], [20])


class RunRecoveryTests(unittest.TestCase):
    def test_happy_path_gap_at_middle_stage(self):
        reports = [
            StageReport(0, True, None, 2),
            StageReport(1, True, 1, 1),
            StageReport(2, True, 1, None),
        ]
        ops = RecordingOps(reports, replay_result=31, endpoint="new-stage-1")

        out = run_recovery(ops, [10, 11], [30, 31])

        self.assertEqual(out, RecoveryOutcome(1, "gap", 3, 31))
        self.assertEqual(
            ops.calls,
            [
                ("probe",),
                ("has_spare", 1),
                ("activate_spare", 1),
                ("rewire", 0, "new-stage-1"),
                ("reset_all",),
                ("replay", [10, 11, 30]),
                ("resume", 3, 31),
            ],
        )

    def test_no_spare_raises_without_resume(self):
        reports = [StageReport(0, False, None, None)]
        ops = RecordingOps(reports, spare=False)

        with self.assertRaisesRegex(HealError, "no warm spare for stage 0"):
            run_recovery(ops, [10], [])

        self.assertNotIn("resume", [call[0] for call in ops.calls])

    def test_replay_mismatch_raises_without_resume(self):
        reports = [
            StageReport(0, True, None, 4),
            StageReport(1, True, 3, None),
        ]
        ops = RecordingOps(reports, replay_result=99)

        with self.assertRaisesRegex(HealError, "replay mismatch: got 99 expected 20"):
            run_recovery(ops, [10], [20])

        self.assertNotIn("resume", [call[0] for call in ops.calls])

    def test_head_death_rewires_from_coordinator(self):
        reports = [StageReport(0, False, None, None)]
        ops = RecordingOps(reports, replay_result=None, endpoint="new-head")

        run_recovery(ops, [10], [])

        self.assertIn(("rewire", -1, "new-head"), ops.calls)


if __name__ == "__main__":
    unittest.main()
