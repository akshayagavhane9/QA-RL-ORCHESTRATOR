"""
Log & Flakiness Analyzer Tool for QA-RL Orchestrator.

This tool keeps track of:
- historical outcomes of tests
- estimated failure probabilities
- estimated flakiness for each test

The CI environment and agents can use this information to:
- compute risk scores
- prioritize certain tests
- detect flaky tests
"""

from typing import Dict, Any, List


class LogAnalyzer:
    """
    Simple in-memory log analyzer.

    It stores aggregated statistics per test, such as:
    - total runs
    - real failures
    - flaky failures

    From this, we can compute:
    - failure rate
    - flakiness rate
    - a simple "risk score"
    """

    def __init__(self, num_tests: int) -> None:
        self.num_tests = num_tests

        # Stats per test index
        self.total_runs: List[int] = [0 for _ in range(num_tests)]
        self.real_failures: List[int] = [0 for _ in range(num_tests)]
        self.flaky_failures: List[int] = [0 for _ in range(num_tests)]

    def update_with_result(
        self,
        test_index: int,
        bug_found: bool,
        is_flaky_failure: bool,
    ) -> None:
        """
        Update statistics based on the outcome of a single test run.
        """
        if test_index < 0 or test_index >= self.num_tests:
            return

        self.total_runs[test_index] += 1

        if bug_found and not is_flaky_failure:
            self.real_failures[test_index] += 1
        elif is_flaky_failure:
            self.flaky_failures[test_index] += 1

    def get_test_stats(self, test_index: int) -> Dict[str, Any]:
        """
        Get raw stats for a specific test.
        """
        if test_index < 0 or test_index >= self.num_tests:
            return {}

        return {
            "total_runs": self.total_runs[test_index],
            "real_failures": self.real_failures[test_index],
            "flaky_failures": self.flaky_failures[test_index],
        }

    def get_test_risk_summary(self) -> Dict[int, Dict[str, float]]:
        """
        Return a dictionary mapping test_index -> summary stats:
        - failure_rate
        - flakiness_rate
        - simple risk_score (placeholder for now)

        This can later be used in the state representation for the DQN.
        """
        summary: Dict[int, Dict[str, float]] = {}

        for i in range(self.num_tests):
            runs = self.total_runs[i]
            if runs == 0:
                failure_rate = 0.0
                flakiness_rate = 0.0
            else:
                failure_rate = self.real_failures[i] / runs
                flakiness_rate = self.flaky_failures[i] / runs

            # Placeholder risk score: just use failure_rate for now
            risk_score = failure_rate

            summary[i] = {
                "failure_rate": failure_rate,
                "flakiness_rate": flakiness_rate,
                "risk_score": risk_score,
            }

        return summary
