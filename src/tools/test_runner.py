"""
Test Runner Tool for QA-RL Orchestrator.

This tool simulates the execution of a test case in the CI environment.

In a real system, this would:
- call test frameworks
- run Selenium/API tests
- collect logs, screenshots, etc.

Here, we only simulate:
- duration
- whether the test passes or fails
- whether a failure is real or flaky
"""

from typing import Dict, Any, Tuple


class TestRunner:
    """
    Simulated test runner.

    The environment will call this to "execute" a test and get:
    - duration
    - pass/fail outcome
    - whether the failure was flaky or real
    """

    def __init__(self) -> None:
        # You can store global/simulation-level settings here if needed.
        pass

    def run_test(
        self,
        test_id: int,
        test_metadata: Dict[int, Dict[str, Any]],
        commit_context: Dict[str, Any],
    ) -> Tuple[float, bool, bool]:
        """
        Simulate running a single test.

        Args:
            test_id: index of the test to run
            test_metadata: dictionary containing properties for each test
            commit_context: information about the current commit (e.g., changed module, bug location)

        Returns:
            duration: simulated test duration (seconds)
            bug_found: True if this test caught a REAL bug
            is_flaky_failure: True if this failure was flaky (false positive)
        """
        import random  # local import is fine here

        meta = test_metadata.get(test_id)
        if meta is None:
            # Unknown test -> no-op
            return 0.0, False, False

        base_duration = float(meta.get("base_duration", 5.0))
        base_fail_prob = float(meta.get("base_fail_prob", 0.2))
        flakiness_prob = float(meta.get("flakiness_prob", 0.02))
        module = meta.get("module", "unknown")

        # --- Simulate duration (small noise around base_duration) ---
        noise_factor = random.uniform(-0.2, 0.2)  # +/- 20%
        duration = base_duration * (1.0 + noise_factor)
        duration = max(0.5, duration)  # minimum duration

        # --- Determine if a real bug is present & relevant to this test ---
        real_bug_present = bool(commit_context.get("real_bug_present", False))
        real_bug_module = commit_context.get("real_bug_module")

        effective_bug_fail_prob = 0.0
        if real_bug_present:
            if real_bug_module == module:
                # This test directly exercises the buggy module
                effective_bug_fail_prob = base_fail_prob
            else:
                # Small chance it indirectly catches cross-module issues
                effective_bug_fail_prob = base_fail_prob * 0.1

        # --- Simulate outcomes ---
        bug_found = False
        is_flaky_failure = False

        # First, check if we caught a real bug
        if effective_bug_fail_prob > 0.0 and random.random() < effective_bug_fail_prob:
            bug_found = True
            is_flaky_failure = False
        else:
            # No real bug detected; maybe a flaky failure
            if random.random() < flakiness_prob:
                bug_found = False
                is_flaky_failure = True

        return duration, bug_found, is_flaky_failure
