"""The account of a tune: it says what was measured, tried and left out, and one thing to do next, from the result alone."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

from strata_cli import tune_commands
from strata_control import autotune, tune_report
from strata_control.tune_session import StartFailed
from tests.tuning.test_autotune import make_ctx, run_tuner, speed_with


def result_of(model, **kw) -> dict:
    t, s, r = run_tuner(model, **kw)
    return r.to_dict()


class WhyNotAWinTest(unittest.TestCase):
    def test_a_gain_below_the_bar_names_both_numbers(self):
        why = autotune.why_not_a_win([1.01, 1.02], [1.0], threshold=1.05, rounds=2)
        self.assertEqual(why, "+1.5% against your settings; it needed +5.0% to count")

    def test_a_decode_gain_that_costs_prompt_speed_says_so(self):
        why = autotune.why_not_a_win([1.08, 1.08], [0.8, 0.8], threshold=1.03, rounds=2)
        self.assertIn("prompts got 20% slower", why)

    def test_a_gain_that_is_not_in_every_round_says_so(self):
        why = autotune.why_not_a_win([1.2, 0.99], [1.0], threshold=1.03, rounds=2)
        self.assertIn("faster in 1 of 2 rounds", why)

    def test_too_few_rounds(self):
        self.assertIn("1 of 2 rounds", autotune.why_not_a_win([1.2], [], threshold=1.03, rounds=2))


class ReportTest(unittest.TestCase):
    def test_a_win_is_reported_with_its_numbers_and_the_next_step(self):
        r = result_of(speed_with({("--kv-resident", "65536"): 1.10}), session_kw={"noise": 0.001})
        text = tune_report.render(r, preset_name="My preset")
        self.assertEqual(r["code"], "faster")
        self.assertIn("Your settings: 60.0 tok/s answers", text)
        self.assertIn("To count, a setting had to be at least", text)
        self.assertIn("What was tried:", text)
        self.assertRegex(text, r"Keep more of the conversation in graphics memory\s+faster")
        self.assertIn("'My preset' is applied at every start", text)

    def test_a_setting_that_did_not_count_says_why(self):
        r = result_of(speed_with({("--kv-resident", "65536"): 1.01}), session_kw={"noise": 0.001})
        rows = {x["id"]: x for x in tune_report.arm_rows(r)}
        self.assertEqual(rows["kv_resident_64k"]["verdict"], "no real gain")
        self.assertIn("needed", rows["kv_resident_64k"]["detail"])
        self.assertIn("Nothing to change", tune_report.render(r))

    def test_settings_left_out_for_answer_differences_say_which_flag_unlocks_them(self):
        r = result_of(speed_with({}))
        text = tune_report.render(r)
        self.assertIn("--allow-small-differences", text)
        self.assertIn("Not tried:", text)
        self.assertNotIn("tick", text)
        self.assertTrue(tune_report.unlockable(r).get("placement"))

    def test_allowing_small_differences_removes_the_hint(self):
        r = result_of(speed_with({}), ctx=make_ctx(allow_placement=True))
        self.assertNotIn("--allow-small-differences", tune_report.render(r))
        self.assertFalse(tune_report.unlockable(r).get("placement"))

    def test_a_setting_left_out_for_time_is_named(self):
        r = {"code": "already_fastest", "skipped": [{"id": "trim_stage", "label": "Trim stage weights on two cards", "reason": "x", "kind": "time"}]}
        self.assertIn("Left out to stay inside the time you chose: Trim stage weights on two cards", tune_report.render(r))

    def test_every_ending_has_a_next_step(self):
        endings = [
            {"code": "faster"}, {"code": "already_fastest"}, {"code": "stopped"},
            {"code": "busy_pc", "reason_code": "noisy"}, {"code": "busy_pc", "reason_code": "drift"},
            {"code": "busy_pc", "reason_code": "not_tried"},
            {"code": "failed", "reason_code": "no_memory"}, {"code": "failed", "reason_code": "start_failed"},
            {"code": "failed", "reason_code": "tuner_error"}]
        for e in endings:
            self.assertTrue(tune_report.next_step(e), e)

    def test_a_baseline_that_ran_out_of_memory_gets_the_memory_advice(self):
        r = result_of(speed_with({}), session_kw={"fail": lambda a, e: StartFailed("OOM", "out of memory")})
        text = tune_report.render(r)
        self.assertIn("raise --vram-reserve-mib", text)
        self.assertNotIn("What was tried", text)

    def test_nothing_is_invented_when_figures_are_missing(self):
        text = tune_report.render({"code": "stopped", "base": {}, "noise": {}, "arms": [], "skipped": []})
        self.assertNotIn("tok/s", text)
        self.assertNotIn("None", text)

    def test_markdown_has_no_paths_or_keys(self):
        r = result_of(speed_with({("--kv-resident", "65536"): 1.10}), session_kw={"noise": 0.001})
        r["trials"].append({"args": ["--native", "/home/someone/secret/model.gguf"], "api_key": "k"})
        md = tune_report.markdown(r)
        self.assertIn("| Setting | Verdict | Detail |", md)
        self.assertNotIn("/home/", md)
        self.assertNotIn("model.gguf", md)

    def test_the_result_the_app_reads_keeps_its_fields(self):
        r = result_of(speed_with({}))
        for key in ("outcome", "code", "reason_code", "values", "message", "arms", "skipped", "noise", "settings", "answers"):
            self.assertIn(key, r)
        self.assertIn("threshold_pct", r["noise"])


class SavedReportTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        patcher = mock.patch.dict(os.environ, {"STRATA_PRESET_DIR": self.dir.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.dir.cleanup)

    def test_save_load_and_list(self):
        r = result_of(speed_with({("--kv-resident", "65536"): 1.10}), session_kw={"noise": 0.001})
        path = tune_report.save(r, target="my-profile", model="qwen3.8 iq3", when=1_800_000_000)
        self.assertIsNotNone(path)
        self.assertEqual(oct(os.stat(path).st_mode & 0o777), "0o600")
        saved = json.loads(path.read_text())
        self.assertNotIn("trials", saved["result"])
        self.assertEqual(tune_report.load()[0], path)
        self.assertEqual(tune_report.load("iq3")[1]["schema"], tune_report.SCHEMA)
        self.assertEqual(tune_report.list_reports(), [path])
        with self.assertRaises(FileNotFoundError):
            tune_report.load("nothing-like-this")

    def test_a_report_that_cannot_be_written_does_not_fail_the_tune(self):
        with mock.patch.object(tune_report.os, "replace", side_effect=OSError("disk full")):
            self.assertIsNone(tune_report.save({"code": "faster"}, target="t"))

    def test_the_presets_list_ignores_the_reports_folder(self):
        from strata_control import presets
        tune_report.save({"code": "faster"}, target="t")
        self.assertEqual(presets.list_presets(), [])


class ProgressTest(unittest.TestCase):
    def test_the_progress_line_says_where_the_tune_is_and_what_it_measured(self):
        text = tune_commands._progress_detail({"runs_done": 2, "runs_planned": 11, "speeds": [
            {"label": "base", "tok_s": 88.5}, {"label": "Larger prompt chunks", "tok_s": 91.0}]})
        self.assertEqual(text, " - run 3 of about 11; your settings 88.5 tok/s; last: Larger prompt chunks 91.0 tok/s")
        self.assertEqual(tune_commands._progress_detail({}), "")


if __name__ == "__main__":
    unittest.main()


class PlanTest(unittest.TestCase):
    def test_the_plan_is_what_the_tuner_then_does(self):
        for mode in ("quick", "thorough"):
            opts = autotune.Options(mode=mode)
            plan = autotune.plan(make_ctx(), opts)
            t, s, r = run_tuner(speed_with({}), opts=autotune.Options(mode=mode))
            self.assertEqual([x["id"] for x in plan["will_try"]], [a["id"] for a in r.arms], mode)

    def test_a_plan_starts_no_engine_and_counts_the_runs(self):
        plan = autotune.plan(make_ctx(), autotune.Options())
        self.assertEqual(plan["runs"], plan["rounds"] * (len(plan["will_try"]) + 1) + 1)
        self.assertEqual(plan["classes"], ["A", "C"])

    def test_a_budget_that_fits_nothing_says_so_rather_than_nothing_is_worth_trying(self):
        plan = autotune.plan(make_ctx(), autotune.Options(budget_s=200.0))
        self.assertEqual(plan["will_try"], [])
        text = tune_report.render_plan(plan)
        self.assertIn("this budget fits no other setting", text)
        self.assertNotIn("worth trying", text)

    def test_the_text_names_what_will_be_tried_and_left_out(self):
        text = tune_report.render_plan(autotune.plan(make_ctx(), autotune.Options()), {"model": "iq3 test", "gpus": "A card (12 GB)"})
        for want in ("Tune plan for iq3 test", "nothing is started or changed", "Graphics: A card (12 GB)", "Will try:",
                     "Keep more of the conversation", "Will not try:", "--allow-small-differences", "without --plan"):
            self.assertIn(want, text)
