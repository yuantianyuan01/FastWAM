import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from experiments.libero.summarize_results import calculate_trial_weighted_success_rate, summarize_results


class CalculateTrialWeightedSuccessRateTest(unittest.TestCase):
    def test_weights_each_suite_by_its_number_of_trials(self):
        suite_stats = {
            "small_suite": {"total_trials": 1, "total_successes": 1},
            "large_suite": {"total_trials": 9, "total_successes": 0},
        }

        self.assertEqual(calculate_trial_weighted_success_rate(suite_stats), 10.0)

    def test_returns_zero_without_trials(self):
        suite_stats = {
            "empty_suite": {"total_trials": 0, "total_successes": 0},
        }

        self.assertEqual(calculate_trial_weighted_success_rate(suite_stats), 0.0)

    def test_summary_json_uses_trial_weighted_rate(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            fixtures = {
                "libero_spatial": {"total_episodes": 1, "successes": 1},
                "libero_goal": {"total_episodes": 9, "successes": 0},
            }
            for suite, result in fixtures.items():
                suite_dir = output_dir / suite
                suite_dir.mkdir()
                result.update({"duration": 1.0, "task_description": "test task"})
                result_file = suite_dir / "gpu0_task0_results.json"
                result_file.write_text(json.dumps(result), encoding="utf-8")

            with redirect_stdout(io.StringIO()):
                summarize_results(output_dir)

            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["overall"]["average_success_rate"], 50.0)
            self.assertEqual(summary["overall"]["trial_weighted_success_rate"], 10.0)


if __name__ == "__main__":
    unittest.main()
