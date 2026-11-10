"""Exercise the launcher's actual lifecycle block without SSH, Docker or GPUs."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "flare_cluster.sh"


class HcaLifecycleTests(unittest.TestCase):
    def run_lifecycle(self, scenario):
        source = SCRIPT.read_text()
        block = source[source.index('RUN_ID="'):source.index('\nstart() {')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "device").mkdir()
            (root / "unrelated").mkdir()
            harness = r'''
set -euo pipefail
DATA=$TEST_ROOT
RANKS=2
RDMA_HCAS=("first,second" "first,second")
say() { printf '%s\n' "$*"; }
fail() { printf '%s\n' "$*" >&2; exit 1; }
node() {
  local rank=$1; shift
  if [[ "$1" == bash ]]; then
    if [[ "$SCENARIO" == remote && "$rank" == 1 ]]; then return 42; fi
    if [[ "$SCENARIO" == signal && "$rank" == 1 ]]; then kill -TERM $$; return 0; fi
    TEST_RANK=$rank "$@" "$rank"
  else
    "$@"
  fi
}
readlink() {
  if [[ "$SCENARIO" == resolve && "$TEST_RANK" == 1 && "$2" == */first ]]; then return 1; fi
  if [[ "$SCENARIO" == missing && "$TEST_RANK" == 1 && "$2" == */first ]]; then
    printf '%s/missing\n' "$TEST_ROOT"; return 0
  fi
  printf '%s/device\n' "$TEST_ROOT"
}
ln() {
  printf '%s:%s\n' "$TEST_RANK" "${3##*/}" >> "$TEST_ROOT/attempts"
  if [[ "$SCENARIO" == link && "$TEST_RANK" == 1 && "$3" == */first ]]; then return 1; fi
  # Simulate independent filesystems for the two nodes in one test directory.
  if [[ "$TEST_RANK" == 1 ]]; then return 0; fi
  command ln "$@"
}
export -f readlink ln
'''
            env = dict(os.environ, TEST_ROOT=tmp, SCENARIO=scenario)
            result = subprocess.run(["bash", "-c", harness + block], env=env,
                                    text=True, capture_output=True)
            self.assertEqual(list(root.glob("runtime/*-ib")), [], result.stderr)
            self.assertTrue((root / "unrelated").is_dir())
            attempts = (root / "attempts").read_text() if (root / "attempts").exists() else ""
            return result, attempts

    def test_success_cleans_up(self):
        result, attempts = self.run_lifecycle("success")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("1:second", attempts)

    def test_partial_initialization_failure_cleans_up(self):
        result, _ = self.run_lifecycle("remote")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("rank 1", result.stderr)

    def test_signal_during_initialization_cleans_up(self):
        result, _ = self.run_lifecycle("signal")
        self.assertEqual(result.returncode, 143, result.stderr)

    def test_first_device_failure_is_not_masked(self):
        for scenario in ("resolve", "missing", "link"):
            with self.subTest(scenario=scenario):
                result, attempts = self.run_lifecycle(scenario)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("rank 1", result.stderr)
                self.assertIn("HCA first", result.stderr)
                self.assertNotIn("1:second", attempts)


if __name__ == "__main__":
    unittest.main()
