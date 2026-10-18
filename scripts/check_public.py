#!/usr/bin/env python3
"""Inspect staged or reachable Git objects before sharing a repository.

Only Git objects are read. Working files, remotes and device resources are untouched.
Diagnostics contain object paths, categories and counts, never matching body text.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys


CAREER = re.compile(
    r"\u7b80\u5386|\u5b9e\u4e60|\u6c42\u804c|\u6295\u9012|"
    r"\u9762\u8bd5|\u62db\u8058|\u5e94\u8058|\u627e\u5de5\u4f5c|"
    r"\b(?:in" + r"ternships?|job[-_\s]*(?:search|hunt(?:ing)?|application|interview)|"
    r"curriculum\s+vitae|r\u00e9sum\u00e9|cover[-_\s]*letter)\b",
    re.IGNORECASE,
)
HOME_PATH = re.compile(r"/(?:ho" + r"me|Use" + r"rs)/([^/\s\"'`\\]+)(?:/|(?=$))")
SCHEDULE = re.compile(
    r"\u4e09\u5468|\b90\s*h\b|90\s*\u5c0f\u65f6|"
    r"\u6bcf\u5468\s*(?:\u7ea6\s*)?30\s*(?:h\b|\u5c0f\u65f6)|"
    r"2026-" + r"11-" + r"05|11[-\uff5e\u301c]05|11\s*\u6708\s*5\s*\u65e5",
    re.IGNORECASE,
)
PRIVATE_COMPONENTS = {".pri" + "vate", ".ssh", ".aws"}
PRIVATE_EXACT = {"configs/lo" + "cal.json"}


class CheckError(RuntimeError):
    """A check could not be completed; do not allow a partial pass."""


def git(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode:
        raise CheckError(f"Git read failed ({args[0]}, exit {result.returncode})")
    return result.stdout


class ObjectReader:
    """Stream each selected object without repeatedly spawning Git."""

    def __init__(self, repo: Path):
        self.process = subprocess.Popen(
            ["git", "-C", str(repo), "cat-file", "--batch"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )

    def read(self, oid: str) -> tuple[str, bytes]:
        if not re.fullmatch(r"[0-9a-f]{40,64}", oid):
            raise CheckError("Invalid object identifier")
        self.process.stdin.write((oid + "\n").encode("ascii"))
        self.process.stdin.flush()
        header = self.process.stdout.readline().split()
        if len(header) != 3 or not header[2].isdigit():
            raise CheckError("Git object is unavailable")
        size = int(header[2])
        payload = self.process.stdout.read(size)
        if len(payload) != size or self.process.stdout.read(1) != b"\n":
            raise CheckError("Incomplete Git object read")
        return header[1].decode("ascii"), payload

    def close(self):
        self.process.stdin.close()
        self.process.stdout.close()
        self.process.wait()


def text_counts(text: str) -> Counter:
    counts = Counter()
    counts["career_context"] = len(CAREER.findall(text))
    counts["personal_schedule"] = len(SCHEDULE.findall(text))
    for match in HOME_PATH.finditer(text):
        user = match.group(1)
        # Explicit documentation placeholders and shell variables are portable.
        if not (user.startswith(("<", "$", "{")) or user.endswith(">")):
            counts["personal_home_path"] += 1
    return +counts


def decoded_json_counts(value) -> Counter:
    counts = Counter()
    if isinstance(value, str):
        counts.update(text_counts(value))
    elif isinstance(value, dict):
        for key, child in value.items():
            counts.update(text_counts(key))
            counts.update(decoded_json_counts(child))
            if key == "active_hours_budget" and child == 90:
                counts["personal_schedule"] += 1
    elif isinstance(value, list):
        for child in value:
            counts.update(decoded_json_counts(child))
    return counts


def blob_counts(payload: bytes, is_json: bool) -> Counter:
    if is_json:
        try:
            # The JSON decoder also recognizes UTF-16/32 and a leading BOM.
            return decoded_json_counts(json.loads(payload))
        except (ValueError, RecursionError):
            pass
    # Opaque assets can contain coincidental character sequences in numeric data.
    # Their Git paths are checked independently; body rules apply to text only.
    if b"\0" in payload:
        return Counter()
    try:
        text = payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return Counter()
    return text_counts(text)


def private_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    name = parts[-1] if parts else ""
    return (
        bool(PRIVATE_COMPONENTS.intersection(parts))
        or normalized in PRIVATE_EXACT
        or name == ".env"
        or (name.startswith(".env.") and name != ".env.example")
        or name.endswith((".local.json", ".local.txt", ".pem", ".key"))
    )


class Scanner:
    def __init__(self, repo: Path):
        self.repo = repo
        self.findings = defaultdict(Counter)
        self.paths = set()
        self.blob_paths = defaultdict(set)
        self.commits = set()
        self.trees = set()

    def add_counts(self, scope: str, path: str, counts: Counter):
        self.findings[(scope, path)].update(counts)

    def add_path(self, path: str):
        if path not in self.paths:
            self.paths.add(path)
            self.add_counts("path", path, text_counts(path))
            if private_path(path):
                self.add_counts("path", path, Counter(private_file=1))

    def read_tree(self, tree: str):
        if tree in self.trees:
            return
        self.trees.add(tree)
        for entry in git(self.repo, "ls-tree", "-r", "-z", "--full-tree", tree).split(b"\0"):
            if not entry:
                continue
            info, name = entry.split(b"\t", 1)
            _mode, kind, oid = info.split()
            path = name.decode("utf-8", errors="surrogateescape")
            self.add_path(path)
            if kind == b"blob":
                self.blob_paths[oid.decode("ascii")].add(path)

    def index(self):
        for entry in git(self.repo, "ls-files", "--stage", "-z").split(b"\0"):
            if not entry:
                continue
            info, name = entry.split(b"\t", 1)
            mode, oid, stage = info.split()
            path = name.decode("utf-8", errors="surrogateescape")
            self.add_path(path)
            if stage != b"0":
                self.add_counts("path", path, Counter(unmerged_index=1))
            if mode != b"160000" and oid.strip(b"0"):
                self.blob_paths[oid.decode("ascii")].add(path)

    def revision(self, ref: str):
        tree = git(self.repo, "rev-parse", "--verify", "--end-of-options",
                   ref + "^{tree}").decode("ascii").strip()
        self.read_tree(tree)

    def history(self, refs: list[str], reader: ObjectReader):
        resolved = []
        for ref in refs:
            resolved.append(git(self.repo, "rev-parse", "--verify", "--end-of-options",
                                ref + "^{commit}").decode("ascii").strip())
        if not resolved:
            return
        for line in git(self.repo, "rev-list", *resolved, "--").splitlines():
            oid = line.decode("ascii")
            if oid in self.commits:
                continue
            self.commits.add(oid)
            kind, payload = reader.read(oid)
            if kind != "commit":
                raise CheckError("Expected a commit object")
            headers, separator, message = payload.partition(b"\n\n")
            tree_lines = [row[5:] for row in headers.splitlines() if row.startswith(b"tree ")]
            if not separator or len(tree_lines) != 1:
                raise CheckError("Malformed commit object")
            self.add_counts("commit_message", "commit:" + oid,
                            text_counts(message.decode("utf-8", errors="replace")))
            self.read_tree(tree_lines[0].decode("ascii"))

    def blobs(self, reader: ObjectReader):
        for oid, paths in self.blob_paths.items():
            kind, payload = reader.read(oid)
            if kind != "blob":
                raise CheckError("Expected a blob object")
            # A blob can have both JSON and non-JSON names across history.
            for is_json in {path.lower().endswith(".json") for path in paths}:
                counts = blob_counts(payload, is_json)
                if counts:
                    for path in sorted(paths):
                        if path.lower().endswith(".json") == is_json:
                            self.add_counts("content", path, counts)

    def result(self, mode: str, ref_count: int) -> dict:
        findings = [
            {"scope": scope, "path": path, "category": category, "count": count}
            for (scope, path), counts in sorted(self.findings.items())
            for category, count in sorted(counts.items()) if count
        ]
        return {
            "passed": not findings,
            "mode": mode,
            "checked": {"refs": ref_count, "commits": len(self.commits),
                        "trees": len(self.trees), "unique_blobs": len(self.blob_paths),
                        "unique_paths": len(self.paths)},
            "finding_count": sum(item["count"] for item in findings),
            "findings": findings,
        }


def push_refs(stream) -> list[str]:
    refs = []
    for line in stream:
        columns = line.split()
        if not columns:
            continue
        if len(columns) != 4 or not re.fullmatch(r"[0-9a-fA-F]{40,64}", columns[1]):
            raise CheckError("Invalid ref update input")
        oid = columns[1].lower()
        if oid.strip("0"):
            refs.append(oid)
    return list(dict.fromkeys(refs))


def default_repo() -> Path:
    here = Path.cwd()
    probe = subprocess.run(["git", "-C", str(here), "rev-parse", "--git-dir"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return here if probe.returncode == 0 else Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="Git checkout or bare repository")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--index", action="store_true", help="Inspect staged objects (default)")
    modes.add_argument("--revision", metavar="REF", help="Inspect one revision tree")
    modes.add_argument("--history", metavar="REF", help="Inspect all reachable commits and trees")
    modes.add_argument("--pre-push", action="store_true", help="Read ref updates from standard input")
    args = parser.parse_args(argv)
    reader = None
    try:
        repo = args.repo if args.repo is not None else default_repo()
        git(repo, "rev-parse", "--git-dir")
        scanner = Scanner(repo)
        reader = ObjectReader(repo)
        if args.pre_push:
            mode, refs = "pre_push", push_refs(sys.stdin)
            scanner.history(refs, reader)
        elif args.history is not None:
            mode, refs = "history", [args.history]
            scanner.history(refs, reader)
        elif args.revision is not None:
            mode, refs = "revision", [args.revision]
            scanner.revision(args.revision)
        else:
            mode, refs = "index", []
            scanner.index()
        scanner.blobs(reader)
        result = scanner.result(mode, len(refs))
        print(json.dumps(result, ensure_ascii=True, indent=2))
        return 0 if result["passed"] else 1
    except (CheckError, OSError, UnicodeError) as exc:
        # Git's raw diagnostics can echo body text or machine-specific paths.
        message = str(exc) if isinstance(exc, CheckError) else type(exc).__name__
        print(json.dumps({"passed": False, "error": message}, ensure_ascii=True), file=sys.stderr)
        return 2
    finally:
        if reader is not None:
            reader.close()


if __name__ == "__main__":
    raise SystemExit(main())
