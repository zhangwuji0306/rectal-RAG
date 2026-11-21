"""Shared helpers for the code release boundary (D-R4).

The release set is defined by an explicit allowlist (``docs/发布范围-allowlist.txt``).
A verified checkout scans and exports the Git index blobs after checking the
working tree against them. Git operations here are read-only.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ALLOWLIST = ROOT / "docs" / "发布范围-allowlist.txt"

# --------------------------------------------------------------------------- #
# allowlist
# --------------------------------------------------------------------------- #


def allowlist_globs() -> list[str]:
    """Return the globs listed inside the fenced block of the allowlist."""
    return parse_allowlist(ALLOWLIST.read_text(encoding="utf-8"))


def parse_allowlist(text: str) -> list[str]:
    blocks = text.split("```")
    if len(blocks) < 3:
        raise AssertionError("allowlist has no fenced glob block")
    return [
        line.strip()
        for line in blocks[1].splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def glob_regex(pattern: str) -> re.Pattern[str]:
    """Translate a path glob into a path-aware regex.

    ``*`` and ``?`` never cross a path separator, a ``**`` segment matches zero
    or more whole segments, and a trailing ``/**`` matches everything below the
    directory (the behaviour the allowlist relies on for fixture trees).
    """
    parts = pattern.split("/")
    out: list[str] = ["^"]
    for index, part in enumerate(parts):
        last = index == len(parts) - 1
        if part == "**":
            # A ``**`` segment consumes its own trailing separator, so it may
            # match zero segments; a trailing ``/**`` takes everything below.
            out.append(".*" if last else "(?:[^/]+/)*")
            continue
        translated = re.escape(part)
        translated = translated.replace(r"\*\*", ".*")
        translated = translated.replace(r"\*", "[^/]*")
        translated = translated.replace(r"\?", "[^/]")
        out.append(translated)
        if not last:
            out.append("/")
    return re.compile("".join(out) + "$")


def matches_allowlist(relative: str, globs: list[str] | None = None) -> bool:
    globs = allowlist_globs() if globs is None else globs
    return any(glob_regex(glob).match(relative) for glob in globs)


def _glob_for_pathlib(pattern: str) -> str:
    return pattern + "/*" if pattern.endswith("/**") else pattern


def worktree_files(globs: list[str] | None = None) -> list[str]:
    """Every file on disk selected by the allowlist globs."""
    globs = allowlist_globs() if globs is None else globs
    found: set[str] = set()
    for pattern in globs:
        for path in ROOT.glob(_glob_for_pathlib(pattern)):
            if path.is_file():
                found.add(path.relative_to(ROOT).as_posix())
    return sorted(found)


# --------------------------------------------------------------------------- #
# read-only git access
# --------------------------------------------------------------------------- #


def git_available() -> bool:
    return (ROOT / ".git").exists()


def git(*args: str) -> str:
    return git_bytes(*args).decode("utf-8", "replace")


def git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", "-c", "core.quotepath=false", *args],
        cwd=ROOT, capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "git " + " ".join(args) + " failed: "
            + completed.stderr.decode("utf-8", "replace").strip()
        )
    return completed.stdout


def tracked_files() -> list[str]:
    return [name for name in git("ls-files").splitlines() if name]


def head_tree() -> list[str]:
    return [name for name in git("ls-tree", "-r", "--name-only", "HEAD").splitlines() if name]


def tags() -> list[str]:
    return [name for name in git("tag").splitlines() if name]


def staged_deletions() -> list[str]:
    """Paths with a staged removal in the index (read from ``git status``)."""
    out: list[str] = []
    for line in git("status", "--porcelain").splitlines():
        if line.startswith("D  ") or line.startswith("D "):
            name = line[3:].strip()
            if name.startswith('"') and name.endswith('"'):
                name = name[1:-1]
            out.append(name)
    return out


def history_entries() -> list[tuple[str, str]]:
    """``(commit, path)`` pairs for every path present in every reachable commit."""
    raw = git("log", "--all", "--name-only", "--pretty=format:%x00%H")
    entries: list[tuple[str, str]] = []
    for chunk in raw.split("\0"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        lines = [line for line in chunk.splitlines() if line]
        commit, paths = lines[0], lines[1:]
        entries.extend((commit, path) for path in paths)
    return entries


def history_additions() -> list[tuple[str, str]]:
    """``(commit, path)`` pairs for every path ever added by a reachable commit."""
    raw = git("log", "--all", "--diff-filter=A", "--name-only", "--pretty=format:%x00%H")
    entries: list[tuple[str, str]] = []
    for chunk in raw.split("\0"):
        chunk = chunk.strip("\n")
        if not chunk:
            continue
        lines = [line for line in chunk.splitlines() if line]
        commit, paths = lines[0], lines[1:]
        entries.extend((commit, path) for path in paths)
    return entries


def recorded_remote_refs() -> dict[str, str]:
    """Remote-tracking refs as recorded locally by the last fetch.

    Read straight from ``.git`` so that inspecting the record needs no network
    access and no additional Git command.
    """
    refs: dict[str, str] = {}
    ref_root = ROOT / ".git" / "refs"
    if ref_root.is_dir():
        for path in sorted(ref_root.rglob("*")):
            if path.is_file():
                name = path.relative_to(ROOT / ".git").as_posix()
                refs[name] = path.read_text(encoding="utf-8").strip()
    packed = ROOT / ".git" / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line or line.startswith(("#", "^")):
                continue
            sha, _, name = line.partition(" ")
            refs.setdefault(name.strip(), sha.strip())
    return refs


# --------------------------------------------------------------------------- #
# release set
# --------------------------------------------------------------------------- #


def release_files(source: str = "worktree") -> list[str]:
    """Allowlisted files, either as tracked in Git or as present on disk."""
    if source == "tracked":
        return [name for name in tracked_files() if matches_allowlist(name)]
    if source == "worktree":
        return worktree_files()
    raise ValueError(f"unknown release source: {source}")


def build_release_dir(target: Path, source: str = "worktree") -> list[str]:
    """Copy the release set into ``target`` and return the relative paths."""
    copied: list[str] = []
    for name in release_files(source):
        origin = ROOT / name
        if not origin.is_file() or origin.resolve() != origin:
            raise ValueError("release source is missing or uses a linked path")
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, destination)
        copied.append(name)
    return copied


def tree_size(target: Path) -> int:
    return sum(path.stat().st_size for path in target.rglob("*") if path.is_file())


# --------------------------------------------------------------------------- #
# forbidden content
# --------------------------------------------------------------------------- #

# Categories that may never be distributed.  Each entry is a label plus a
# path-aware pattern; the labels are used in failure messages.
FORBIDDEN_PATHS: tuple[tuple[str, str], ...] = (
    ("raw_fulltext_xml", r"^corpus_raw/"),
    ("corpus_markdown", r"^(直肠癌文献爬取/)?converted/.*\.md$"),
    ("mac_corpus_copy", r"^(直肠癌文献爬取/)?mucinous/"),
    ("normalized_corpus", r"^(直肠癌文献爬取/)?normalized/"),
    ("legacy_pdf_corpus", r"^(直肠癌文献爬取/)?(papers|pdfs_merged|flash-failed)/"),
    ("vector_store", r"(^|/)chroma_db"),
    ("production_database", r"\.sqlite"),
    ("bulk_metadata_export", r"\.nbib$"),
    ("ledger_manifest", r"(^|/)manifest[^/]*\.json$"),
    ("b3_state", r"^index/b3-[^/]*\.json$"),
    ("model_cache", r"(^|/)\.model-cache/"),
    ("browser_driver", r"drivers?/|chromedriver|geckodriver"),
    ("local_backup", r"(^|/)[^/]*-backups/|^backups/|^archive/"),
    ("zotero_metadata", r"\.zotero-|\.zvec"),
    ("runtime_reports", r"^直肠癌文献爬取/reports/"),
    ("virtualenv", r"^\.venv/"),
    ("release_artifacts", r"^dist/"),
    ("env_file", r"^\.env$|^\.env\.(?!example$)"),
)
FORBIDDEN_RES = tuple((label, re.compile(pattern)) for label, pattern in FORBIDDEN_PATHS)

# Explicit exemptions: a synthetic test fixture that the allowlist names and the
# ignore rules un-ignore.  Everything else matching a forbidden category fails.
EXEMPT_FILES = {"tests/fixtures/a7_article.pdf", ".env.example"}
FIXTURE_PDF = "tests/fixtures/a7_article.pdf"
FIXTURE_PDF_MAX_BYTES = 64 * 1024

# Files whose contents must not contain local secret material or machine paths.
SECRET_PATTERNS: tuple[tuple[str, str], ...] = (
    ("configured_token", r"""\b(?:MINERU_TOKEN|NCBI_API_KEY)["']?\s*[:=]\s*["']?(?!your|YOUR|MY|<|\$\{|xxx|placeholder|PLACEHOLDER)[A-Za-z0-9_.\-]{20,}"""),
    ("openai_style_key", r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}\b"),
    ("github_token", r"\b(gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b"),
    ("aws_access_key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("slack_token", r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b"),
    ("private_key_block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("assigned_secret", r"""(?i)\b(api[_-]?key|apikey|secret[_-]?key|access[_-]?token|client[_-]?secret|passwd|password)["']?\s*[:=]\s*["'][A-Za-z0-9/+_\-]{24,}["']"""),
)
SECRET_RES = tuple((label, re.compile(pattern)) for label, pattern in SECRET_PATTERNS)
PRIVATE_PATH_RE = re.compile(r"\b[A-Za-z]:[\\/](?:Users|writing-rag)[\\/]|/(?:home|Users)/[^/\s]+/")
TEXT_SUFFIXES = {
    ".py", ".md", ".json", ".jsonl", ".txt", ".csv", ".ps1", ".sh",
    ".yml", ".yaml", ".toml", ".cfg", ".ini", ".example", ".lock", ".gitignore", ".xml",
}
DATA_SUFFIXES = {".jsonl", ".csv", ".json", ".xml"}


def forbidden_hits(relative: str) -> list[str]:
    if relative in EXEMPT_FILES:
        return []
    return [label for label, regex in FORBIDDEN_RES if regex.search(relative)]


def is_text(path: Path) -> bool:
    if path.suffix.lower() in TEXT_SUFFIXES:
        return True
    return path.suffix == "" and path.name in {".gitignore", ".gitattributes"}


def secret_hits(text: str) -> list[tuple[str, int, str]]:
    hits: list[tuple[str, int, str]] = []
    for label, regex in SECRET_RES:
        for match in regex.finditer(text):
            line = text[: match.start()].count("\n") + 1
            hits.append((label, line, "[REDACTED]"))
    return hits


def private_path_hits(text: str) -> list[tuple[int, str]]:
    hits: list[tuple[int, str]] = []
    for match in PRIVATE_PATH_RE.finditer(text):
        line = text[: match.start()].count("\n") + 1
        hits.append((line, "[REDACTED]"))
    return hits


def content_rules(relative: str, data: bytes) -> list[str]:
    path = Path(relative)
    if relative == FIXTURE_PDF:
        if len(data) > FIXTURE_PDF_MAX_BYTES or not data.startswith(b"%PDF-"):
            return ["invalid_synthetic_pdf"]
        return []
    if not is_text(path):
        return ["unreviewed_file_type"]
    try:
        content = data.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
    except UnicodeError:
        return ["invalid_utf8"]
    rules = [f"{label} at line {line}" for label, line, _ in secret_hits(content)]
    if path.suffix.lower() in DATA_SUFFIXES:
        rules.extend(f"private_absolute_path at line {line}"
                     for line, _ in private_path_hits(content))
    return rules


def scan_release(target: Path, files: list[str]) -> list[str]:
    """Check each candidate file. Diagnostics contain rule IDs and line numbers only."""
    globs = allowlist_globs()
    findings = []
    for number, relative in enumerate(sorted(files), 1):
        path = target / relative
        rules = forbidden_hits(relative)
        if not matches_allowlist(relative, globs):
            rules.append("outside_allowlist")
        if not path.is_file() or path.resolve() != path:
            rules.append("missing_or_linked_file")
        if not rules:
            rules = content_rules(relative, path.read_bytes())
        findings.extend(f"file #{number}: {rule}" for rule in rules)
    return findings


class CheckoutBoundaryError(ValueError):
    """A checkout mismatch whose message is safe to print."""


def verified_checkout() -> tuple[int, dict[str, tuple[str, bytes]], list[str]]:
    """Verify a checkout against one snapshot of stage-0 index entries."""
    entries = {}
    invalid = 0
    for row in git_bytes("ls-files", "--stage", "-z").decode("utf-8").split("\0"):
        if not row:
            continue
        metadata, relative = row.split("\t", 1)
        mode, oid, stage = metadata.split()
        if mode not in {"100644", "100755"} or stage != "0":
            invalid += 1
        entries[relative] = (mode, oid)
    if invalid:
        raise CheckoutBoundaryError(f"{invalid} non-regular or unmerged index entries")

    allowlist_name = ALLOWLIST.relative_to(ROOT).as_posix()
    if allowlist_name not in entries:
        raise CheckoutBoundaryError("allowlist missing from the Git index")
    allowlist_data = git_bytes("cat-file", "blob", entries[allowlist_name][1])
    globs = parse_allowlist(allowlist_data.decode("utf-8"))
    files = worktree_files(globs)
    unexpected = entries.keys() - set(files)
    untracked = set(files) - entries.keys()
    if unexpected or untracked:
        raise CheckoutBoundaryError(f"{len(unexpected)} unexpected tracked, "
                                    f"{len(untracked)} untracked selected")

    candidate = {}
    findings = []
    for number, relative in enumerate(files, 1):
        path = ROOT / relative
        rules = forbidden_hits(relative)
        if not matches_allowlist(relative, globs):
            rules.append("outside_allowlist")
        if not path.is_file() or path.resolve() != path:
            rules.append("missing_or_linked_file")
        if not rules:
            mode, oid = entries[relative]
            data = allowlist_data if relative == allowlist_name else git_bytes("cat-file", "blob", oid)
            worktree_data = path.read_bytes()
            # Git can materialize text blobs with CRLF on Windows. Release
            # bytes still come from the captured index, including for output.
            if worktree_data != data and worktree_data != git_bytes(
                "cat-file", "--filters", f"--path={relative}", oid
            ):
                rules.append("worktree_content_differs_from_index")
            rules.extend(content_rules(relative, data))
            candidate[relative] = (mode, data)
        findings.extend(f"file #{number}: {rule}" for rule in rules)
    return len(files), candidate, findings


# --------------------------------------------------------------------------- #
# .gitignore coverage
# --------------------------------------------------------------------------- #


def gitignore_patterns() -> list[str]:
    lines = (ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def _ignore_pattern_regex(pattern: str) -> re.Pattern[str] | None:
    body = pattern.lstrip("/")
    if body.endswith("/"):
        body = body + "**"
    if body.startswith("**/"):
        body = body[3:]
    if body.endswith("/**"):
        return glob_regex(body)
    if "/" not in body:
        return glob_regex("**/" + body)
    return glob_regex(body)


def is_ignored(relative: str, patterns: list[str] | None = None) -> bool:
    """Last-match-wins evaluation of the repository ignore rules."""
    patterns = gitignore_patterns() if patterns is None else patterns
    ignored = False
    for pattern in patterns:
        negated = pattern.startswith("!")
        body = pattern[1:] if negated else pattern
        regex = _ignore_pattern_regex(body)
        if regex is None:
            continue
        if regex.match(relative) or regex.match(_basename_probe(relative)):
            ignored = not negated
    return ignored


def _basename_probe(relative: str) -> str:
    return relative


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan or assemble the allowlisted code distribution")
    parser.add_argument("--output", type=Path, help="assemble and scan a new directory")
    parser.add_argument("--verify-checkout", action="store_true",
                        help="verify the working tree and scan the allowlisted Git index blobs")
    args = parser.parse_args()
    if args.verify_checkout:
        if not git_available():
            print("NOT-RUN: a Git checkout is required")
            return 2
        try:
            count, candidate, findings = verified_checkout()
        except CheckoutBoundaryError as exc:
            print(f"FAIL checkout boundary: {exc}")
            return 1
        except (ValueError, AssertionError, RuntimeError, OSError):
            # Git and filesystem errors may contain candidate path names.
            print("FAIL checkout boundary: cannot read the Git candidate")
            return 1
    else:
        files = release_files()
        count = len(files)
        findings = scan_release(ROOT, files)
    if not count:
        print("FAIL: no files selected")
        return 1
    if args.output and not findings:
        target = args.output.resolve()
        target.mkdir(parents=True, exist_ok=False)
        if args.verify_checkout:
            for name, (mode, data) in candidate.items():
                destination = target / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(data)
                if mode == "100755":
                    destination.chmod(0o755)
        else:
            copied = build_release_dir(target)
            if copied != files:
                print("FAIL: selected paths changed during assembly")
                return 1
            findings = scan_release(target, copied)
    for finding in findings:
        print(finding)
    print(f"{'FAIL' if findings else 'PASS'} release scan: {count} files, "
          f"{len(findings)} findings (values redacted)")
    return int(bool(findings))


if __name__ == "__main__":
    raise SystemExit(main())
