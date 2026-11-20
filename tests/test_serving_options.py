"""Check the cluster launcher's opt-in serving flags without GPUs or SSH."""

import shlex
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "flare_cluster.sh"
BASE = [
    "bash", str(SCRIPT), "pilcothink/flare_spark:pre-0.2", "192.0.2.1",
    "--workers", "worker", "--model", "test-model", "--state-dir", "/tmp/flare-test",
    "--dry-run",
]
OPTIONS = {
    "--scheduler-policy": "throughput",
    "--scheduler-terminal-value": "0.5",
    "--kv-cache-limit": "context",
    "--embedding-table-storage": "ram",
    "--embedding-table-cache-size": "1G",
}


class ServingOptionsTests(unittest.TestCase):
    def test_opt_in_flags_go_only_to_both_serving_ranks(self):
        for opts in ({}, OPTIONS):
            with self.subTest(opts=opts):
                flags = [part for pair in opts.items() for part in pair]
                result = subprocess.run(BASE + flags, capture_output=True, text=True, check=True)
                commands = [shlex.split(line) for line in result.stdout.splitlines() if line.startswith("  ")]
                self.assertEqual(len(commands), 4)
                for cmd in commands:
                    is_serving = "/usr/local/bin/flare" in cmd
                    for flag in OPTIONS:
                        if is_serving and flag in opts:
                            self.assertEqual(cmd[cmd.index(flag) + 1], opts[flag])
                        else:
                            self.assertNotIn(flag, cmd)

    def test_unqualified_topology_and_cache_flags_are_not_advertised(self):
        for flag in ("--roce-chain", "--roce-ring", "--prefix-cache-memory", "--prefix-cache-disk", "--prefix-cache-pool"):
            with self.subTest(flag=flag):
                result = subprocess.run(BASE + [flag, "example"], capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn(f"Unknown option: {flag}", result.stderr)
