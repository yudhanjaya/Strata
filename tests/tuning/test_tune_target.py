"""`strata tune iq3_s`: a model or part of a saved setup's name is enough to say what to tune."""

from __future__ import annotations

import unittest

from strata_cli import cli, tune_target
from strata_cli.tune_target import TargetError, choose, rewrite_argv

PROFILES = ["launch-iq3_s-128k", "launch-iq3_s-dual", "launch-ist-das-2-00bpw-128k", "launch-swift-iq2_xs-128k"]
MODELS = {"iq3_s": "/i/strata-iq3_s.json", "swift-iq2_xs": "/i/strata-swift-iq2_xs.json",
          "deepseek-v4.1-flash-ist-das-2.00bpw": "/i/strata-deepseek.json", "iq3_xxs": "/i/strata-iq3_xxs.json"}
LAST_USED = {"launch-iq3_s-128k": 20.0, "launch-iq3_s-dual": 10.0}


def pick(q, profiles=PROFILES, models=MODELS):
    return choose(q, profiles, models, newest=lambda p: LAST_USED.get(p, 0.0))


class ChooseTest(unittest.TestCase):
    def test_an_exact_saved_profile_is_used_as_it_is(self):
        c = pick("launch-iq3_s-dual")
        self.assertEqual((c.ref, c.kind, c.note), ("launch-iq3_s-dual", "profile", ""))

    def test_a_profile_without_the_launch_prefix_is_found(self):
        self.assertEqual(pick("iq3_s-dual").ref, "launch-iq3_s-dual")

    def test_a_model_with_saved_setups_gets_the_one_used_last_and_names_the_others(self):
        c = pick("iq3_s")
        self.assertEqual((c.ref, c.kind), ("launch-iq3_s-128k", "profile"))
        self.assertIn("launch-iq3_s-dual", c.note)
        self.assertEqual(c.others, ["launch-iq3_s-dual"])

    def test_the_model_name_is_matched_loosely(self):
        self.assertEqual(pick("IQ3S").ref, "launch-iq3_s-128k")
        self.assertEqual(pick("Swift-IQ2_XS").ref, "launch-swift-iq2_xs-128k")

    def test_a_model_never_takes_another_models_setups(self):
        c = pick("iq3_xxs")                      # its own run configuration, not iq3_s's saved setups
        self.assertEqual((c.ref, c.kind), ("/i/strata-iq3_xxs.json", "model"))
        self.assertIn("installed settings", c.note)

    def test_a_part_of_a_name_works_when_it_is_unique(self):
        self.assertEqual(pick("swift").ref, "launch-swift-iq2_xs-128k")      # the model and its own setup are one answer
        self.assertEqual(pick("xxs").ref, "/i/strata-iq3_xxs.json")
        self.assertEqual(pick("dual").ref, "launch-iq3_s-dual")

    def test_a_setup_that_does_not_start_with_its_models_name_is_not_guessed(self):
        with self.assertRaises(TargetError) as e:
            pick("ist-das")
        self.assertIn("launch-ist-das-2-00bpw-128k", str(e.exception))
        self.assertIn("deepseek-v4.1-flash-ist-das-2.00bpw", str(e.exception))

    def test_a_part_that_fits_several_is_refused_with_the_names(self):
        with self.assertRaises(TargetError) as e:
            pick("128k")
        self.assertIn("launch-iq3_s-128k", str(e.exception))
        self.assertIn("launch-swift-iq2_xs-128k", str(e.exception))

    def test_an_unknown_name_lists_what_exists_and_the_closest(self):
        with self.assertRaises(TargetError) as e:
            pick("iq3_xs")
        text = str(e.exception)
        self.assertIn("no saved setup or installed model is called 'iq3_xs'", text)
        self.assertIn("Did you mean", text)
        self.assertIn("iq3_xxs", text)

    def test_nothing_installed_points_at_launch(self):
        with self.assertRaises(TargetError) as e:
            choose("iq3_s", [], {})
        self.assertIn("strata launch", str(e.exception))

    def test_an_empty_name_is_refused(self):
        with self.assertRaises(TargetError):
            pick("  ")


class RewriteTest(unittest.TestCase):
    def test_a_plain_word_after_tune_is_a_name_to_tune(self):
        self.assertEqual(rewrite_argv(["tune", "iq3_s", "--mode", "thorough"]), ["tune", "auto", "iq3_s", "--mode", "thorough"])

    def test_every_existing_form_is_left_alone(self):
        for argv in (["tune", "auto", "x"], ["tune", "report"], ["tune", "plan", "--engine", "e"], ["tune", "--help"],
                     ["tune"], ["launch", "iq3_s"], ["presets", "list"], []):
            self.assertEqual(rewrite_argv(argv), argv)

    def test_the_subcommand_list_matches_the_parser(self):
        import argparse
        tune = next(a for a in cli.parser()._subparsers._group_actions[0].choices["tune"]._subparsers._group_actions)
        self.assertEqual(set(tune.choices), set(tune_target.SUBCOMMANDS))


if __name__ == "__main__":
    unittest.main()


class ResolveTargetTest(unittest.TestCase):
    """`tune auto` keeps taking a profile or a path exactly as before; only a name that is neither is looked up."""

    def test_the_lookup_order(self):
        from unittest import mock
        from strata_cli import tune_commands
        from strata_control import profiles

        saved = [{"id": "launch-iq3_s-128k"}, {"id": "broken", "error": "cannot read profile (ValueError)"}]
        found = tune_target.Choice("launch-iq3_s-128k", "launch-iq3_s-128k", "profile", "a note")

        def resolve(ref):
            if ref in ("launch-iq3_s-128k", "broken"):
                if ref == "broken":
                    raise profiles.ProfileError("cannot read profile (ValueError)", 5)
                return None, None, None
            if ref.endswith(".json"):
                return None, None, None
            raise profiles.ProfileError(f"profile {ref!r} is not saved; use strata profile create", 5)

        with mock.patch.object(profiles, "resolve", resolve), mock.patch.object(profiles, "list_profiles", lambda: saved), \
                mock.patch.object(tune_target, "resolve", return_value=found) as lookup:
            self.assertEqual(tune_commands._resolve_target("launch-iq3_s-128k"), ("launch-iq3_s-128k", ""))
            self.assertEqual(tune_commands._resolve_target("/some/run.json"), ("/some/run.json", ""))
            lookup.assert_not_called()
            self.assertEqual(tune_commands._resolve_target("iq3_s"), ("launch-iq3_s-128k", "a note"))
            lookup.assert_called_once_with("iq3_s")
            with self.assertRaises(profiles.ProfileError):               # a saved profile that cannot be read is not guessed around
                tune_commands._resolve_target("broken")
