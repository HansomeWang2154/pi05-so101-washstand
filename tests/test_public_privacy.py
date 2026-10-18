"""Repository sharing checks use isolated local object databases."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
CHECKER = ROOT / "scripts" / "check_public.py"
CAREER_TEXT = "\u7b80\u5386 / in" + "ternship / job-" + "search"
PERSONAL_PATH = "/ho" + "me/test-person/work/project"


class PublicSharingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.repo = Path(self.temp.name) / "repo"
        self.repo.mkdir()
        self.git("init", "-q", "--initial-branch=main")
        self.git("config", "user.name", "Test Maintainer")
        self.git("config", "user.email", "test@example.invalid")

    def tearDown(self):
        self.temp.cleanup()

    def git(self, *args):
        return subprocess.run(["git", "-C", str(self.repo), *args], check=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.decode().strip()

    def write(self, path, text):
        target = self.repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")

    def commit(self, text="Technical change"):
        self.git("add", "-A")
        self.git("commit", "-q", "-m", text)
        return self.git("rev-parse", "HEAD")

    def check(self, *args, stdin=None, expected=0, repo=None):
        result = subprocess.run(
            [sys.executable, str(CHECKER), "--repo", str(repo or self.repo), *args],
            input=stdin, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=self.repo,
        )
        self.assertEqual(result.returncode, expected, result.stdout + result.stderr)
        report = json.loads(result.stderr if expected == 2 else result.stdout)
        self.assertNotIn(CAREER_TEXT, result.stdout + result.stderr)
        self.assertNotIn(PERSONAL_PATH, result.stdout + result.stderr)
        return report

    def test_index_reads_staged_objects_and_allows_resume_and_portable_paths(self):
        self.write("README.md", "resume training; OpenCV; $HOME/work; /ho" + "me/<user>/work")
        self.git("add", "README.md")
        self.write("README.md", CAREER_TEXT)
        report = self.check()
        self.assertTrue(report["passed"])
        self.assertEqual(report["checked"]["unique_blobs"], 1)

    def test_index_rejects_staged_context_without_echoing_body(self):
        self.write("README.md", CAREER_TEXT)
        self.git("add", "README.md")
        report = self.check("--index", expected=1)
        self.assertEqual(report["findings"][0]["category"], "career_context")
        self.assertEqual(report["finding_count"], 3)

    def test_personal_home_locations_are_rejected(self):
        self.write("README.md", PERSONAL_PATH + "\n/Use" + "rs/another-person/work")
        self.git("add", "README.md")
        report = self.check(expected=1)
        self.assertEqual(report["findings"][0]["category"], "personal_home_path")
        self.assertEqual(report["finding_count"], 2)

    def test_json_keys_and_values_are_decoded_but_python_source_is_not(self):
        self.write("config.json", json.dumps({"\u5b9e\u4e60": {"root": PERSONAL_PATH}}))
        self.write("rules.py", 'PATTERN = r"\\u7b80\\u5386"\n')
        self.git("add", "-A")
        report = self.check(expected=1)
        self.assertEqual({row["path"] for row in report["findings"]}, {"config.json"})
        self.assertEqual({row["category"] for row in report["findings"]},
                         {"career_context", "personal_home_path"})

    def test_force_added_private_files_are_rejected(self):
        self.write(".gitignore", ".pri" + "vate/\nconfigs/lo" + "cal.json\n.env\n")
        names = [".pri" + "vate/note.md", "configs/lo" + "cal.json", ".env"]
        for name in names:
            self.write(name, "{}")
        self.git("add", ".gitignore")
        self.git("add", "-f", *names)
        report = self.check(expected=1)
        self.assertEqual({row["path"] for row in report["findings"]}, set(names))
        self.assertTrue(all(row["category"] == "private_file" for row in report["findings"]))

    def test_example_environment_file_is_allowed(self):
        self.write(".env.example", "MODEL_NAME=base")
        self.git("add", "-A")
        self.assertTrue(self.check()["passed"])

    def test_clean_tip_does_not_hide_old_content_or_old_private_paths(self):
        self.write("README.md", CAREER_TEXT)
        self.write(".pri" + "vate/note.md", "local notes")
        self.commit()
        self.git("rm", "-q", ".pri" + "vate/note.md")
        self.write("README.md", "Technical project")
        self.commit()
        self.assertTrue(self.check("--revision", "main")["passed"])
        report = self.check("--history", "main", expected=1)
        self.assertEqual(report["checked"]["commits"], 2)
        self.assertEqual({row["category"] for row in report["findings"]},
                         {"career_context", "private_file"})

    def test_old_commit_messages_are_inspected(self):
        self.write("README.md", "Technical project")
        self.commit("Record " + CAREER_TEXT)
        self.write("README.md", "Updated technical project")
        self.commit()
        report = self.check("--history", "main", expected=1)
        self.assertTrue(all(row["scope"] == "commit_message" for row in report["findings"]))

    def test_old_escaped_json_is_inspected_after_tip_cleanup(self):
        self.write("config.json", json.dumps({"notes": "\u5b9e\u4e60"}))
        self.commit()
        self.write("config.json", "{}")
        self.commit()
        self.check("--revision", "main")
        report = self.check("--history", "main", expected=1)
        self.assertEqual(report["findings"][0]["category"], "career_context")

    def test_json_encoding_does_not_hide_decoded_values(self):
        target = self.repo / "config.json"
        target.write_bytes(json.dumps({"notes": "\u5b9e\u4e60"}).encode("utf-16"))
        self.git("add", "-A")
        self.assertEqual(self.check(expected=1)["finding_count"], 1)

    def test_binary_assets_do_not_match_coincidental_text_bytes(self):
        marker = b"90" + b"h"
        (self.repo / "numeric.stl").write_bytes(b"\0\x01 " + marker + b" \0\x02")
        (self.repo / "opaque.stl").write_bytes(b"\xff " + marker + b" \xfe")
        self.git("add", "-A")
        report = self.check()
        self.assertTrue(report["passed"])
        self.assertEqual(report["checked"]["unique_blobs"], 2)

    def test_history_deduplicates_blobs_but_checks_every_name(self):
        self.write("safe.md", "Technical project")
        self.commit()
        self.git("mv", "safe.md", "\u7b80\u5386.md")
        self.commit()
        report = self.check("--history", "main", expected=1)
        self.assertEqual(report["checked"]["unique_blobs"], 1)
        self.assertEqual(report["checked"]["unique_paths"], 2)
        self.assertEqual(report["findings"][0]["scope"], "path")

    def test_schedule_context_and_numeric_json_budget_are_rejected(self):
        self.write("README.md", "\u4e09\u5468 90" + "h 2026-11-" + "05")
        self.write("config.json", json.dumps({"active_hours_budget": 90}))
        self.git("add", "-A")
        report = self.check(expected=1)
        self.assertEqual(report["finding_count"], 4)
        self.assertTrue(all(row["category"] == "personal_schedule" for row in report["findings"]))

    def test_pre_push_inspects_all_reachable_history_and_multiple_updates(self):
        self.write("README.md", CAREER_TEXT)
        old = self.commit()
        self.write("README.md", "Technical project")
        new = self.commit()
        self.check("--revision", new)
        lines = (f"refs/heads/main {new} refs/heads/main {old}\n"
                 f"refs/heads/backup {new} refs/heads/backup {'0' * 40}\n")
        report = self.check("--pre-push", stdin=lines, expected=1)
        self.assertEqual(report["checked"]["refs"], 1)
        self.assertEqual(report["checked"]["commits"], 2)

    def test_pre_push_deletion_has_no_new_objects(self):
        self.write("README.md", CAREER_TEXT)
        old = self.commit()
        report = self.check("--pre-push", stdin=f"(delete) {'0' * 40} refs/heads/main {old}\n")
        self.assertEqual(report["checked"]["commits"], 0)

    def test_pre_push_checks_the_second_distinct_ref(self):
        self.write("README.md", "Technical project")
        clean = self.commit()
        self.write("README.md", CAREER_TEXT)
        unsafe = self.commit()
        lines = (f"refs/heads/clean {clean} refs/heads/clean {'0' * 40}\n"
                 f"refs/heads/other {unsafe} refs/heads/other {'0' * 40}\n")
        report = self.check("--pre-push", stdin=lines, expected=1)
        self.assertEqual(report["checked"]["refs"], 2)
        self.assertEqual(report["checked"]["commits"], 2)

    def test_malformed_hook_input_and_invalid_ref_fail_closed(self):
        self.check("--pre-push", stdin="unexpected input\n", expected=2)
        self.check("--history", "absent", expected=2)

    def test_bare_repository_is_supported(self):
        self.write("README.md", "Technical project")
        self.commit()
        bare = Path(self.temp.name) / "candidate.git"
        subprocess.run(["git", "clone", "--bare", "-q", str(self.repo), str(bare)],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertTrue(self.check("--history", "main", repo=bare)["passed"])

    def test_checker_and_test_sources_are_shareable(self):
        self.write("scripts/check_public.py", CHECKER.read_text())
        self.write("tests/test_public_privacy.py", Path(__file__).read_text())
        self.git("add", "-A")
        self.assertTrue(self.check()["passed"])


if __name__ == "__main__":
    unittest.main()
