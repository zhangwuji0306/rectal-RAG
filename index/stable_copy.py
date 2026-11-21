"""Windows stable-copy controller. No implicit production paths or CLI defaults.

Call only with an explicitly approved CopySpec. Failed runs remain quarantined;
this module never resumes, deletes, publishes, or evaluates a copy.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import uuid
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path

try:
    from .file_lock import FileLock, writer_lock_path
except ImportError:
    from file_lock import FileLock, writer_lock_path


class CopyRefused(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise CopyRefused(message)


def absolute(value):
    # Only standard local drive paths are needed by this contract. Inspect the
    # spelling before abspath can erase traversal or Win32 alias components.
    raw = os.fspath(value)
    require(isinstance(raw, str) and re.match(r"^[A-Za-z]:[\\/]", raw),
            "standard local drive absolute paths required")
    for component in re.split(r"[\\/]", raw[3:]):
        require(component not in {".."} and not component.endswith((".", " ")),
                "Win32 alias component refused")
        require(not re.search(r'[<>:"|?*~\x00-\x1f]', component),
                "Win32 alias component refused")
        stem = component.split(".", 1)[0].rstrip(" ").upper()
        require(not re.fullmatch(r"CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³]", stem),
                "DOS device component refused")
    return Path(os.path.abspath(raw))  # lexical only; never resolve links


def check_long_name(path):
    """Query one name only after its parent passed the root-to-leaf checks.

    FindFirstFile returns the long directory-entry name, even for arbitrary
    short names without '~'. It does not open file contents or resolve links.
    """
    if path == Path(path.anchor):
        return
    from ctypes import wintypes
    class FindData(ctypes.Structure):
        _fields_ = [("attributes", wintypes.DWORD),
                    ("created", wintypes.FILETIME), ("accessed", wintypes.FILETIME),
                    ("written", wintypes.FILETIME), ("size_high", wintypes.DWORD),
                    ("size_low", wintypes.DWORD), ("reserved0", wintypes.DWORD),
                    ("reserved1", wintypes.DWORD), ("name", wintypes.WCHAR * 260),
                    ("alternate", wintypes.WCHAR * 14)]
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.FindFirstFileW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(FindData)]
    api.FindFirstFileW.restype = wintypes.HANDLE
    api.FindClose.argtypes = [wintypes.HANDLE]
    api.FindClose.restype = wintypes.BOOL
    data = FindData()
    handle = api.FindFirstFileW(str(path), ctypes.byref(data))
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        require(data.name.casefold() == path.name.casefold(), "short-name alias refused")
    finally:
        if not api.FindClose(handle):
            raise ctypes.WinError(ctypes.get_last_error())


def volume_path(path):
    """Use the local volume identity; refuse SUBST/network DOS drive aliases."""
    path = absolute(path)
    from ctypes import wintypes
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.QueryDosDeviceW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
    api.QueryDosDeviceW.restype = wintypes.DWORD
    target = ctypes.create_unicode_buffer(32768)
    if not api.QueryDosDeviceW(path.drive, target, len(target)):
        raise ctypes.WinError(ctypes.get_last_error())
    require(re.fullmatch(r"\\Device\\HarddiskVolume[0-9]+", target.value),
            "nonstandard local volume mapping refused")
    return target.value + str(path)[2:]


def under(path, root):
    p, r = volume_path(path).casefold(), volume_path(root).casefold()
    return p == r or p.startswith(r.rstrip("\\/") + os.sep)


def overlap(a, b):
    return under(a, b) or under(b, a)


def check_node(path, role):
    path = absolute(path)
    info = path.lstat()
    require(not stat.S_ISLNK(info.st_mode) and not (
        getattr(info, "st_file_attributes", 0) & 0x400), "reparse point refused")
    check_long_name(path)
    require((stat.S_ISDIR(info.st_mode) if role == "Container" else
             stat.S_ISREG(info.st_mode)), "path role mismatch")
    if role == "Leaf":
        require(info.st_nlink == 1, "hardlinked source refused")
    return info


def check_chain(path, role, *, missing_leaf=False, observe=lambda *_: None):
    path = absolute(path)
    parts = path.parts
    node = Path(parts[0])
    for i in range(len(parts)):
        if i:
            node /= parts[i]
        last = i == len(parts) - 1
        observe(node, role if last else "Container")
        if last and missing_leaf:
            try:
                node.lstat()
            except FileNotFoundError:
                return
            raise CopyRefused("RunRoot already exists")
        check_node(node, role if last else "Container")


def tree(root, observe=lambda *_: None, *, inventory=True):
    """Inspect immediate children before enqueueing directories; never rglob."""
    check_chain(root, "Container", observe=observe)
    queue, dirs, files = deque([root]), ["."], {}
    while queue:
        parent = queue.popleft()
        check_node(parent, "Container")
        with os.scandir(parent) as entries:
            children = sorted((Path(e.path) for e in entries), key=str)
        for child in children:
            info = child.lstat()
            role = "Container" if stat.S_ISDIR(info.st_mode) else "Leaf"
            observe(child, role)
            info = check_node(child, role)
            name = child.relative_to(root).as_posix()
            if role == "Container":
                dirs.append(name)
                queue.append(child)
            else:
                files[name] = (info.st_size, info.st_mtime_ns) if inventory else None
    return {"dirs": sorted(dirs), "files": dict(sorted(files.items())) if inventory else sorted(files)}


@dataclass(frozen=True)
class CopySpec:
    snapshot_root: Path  # approved containment scope, not a copied tree
    eval_root: Path  # approved evaluation containment scope
    db: Path
    model: Path
    inputs: dict  # manifest, ready, journal, plan, qrels, split, freeze: Leaf
    code_leaves: tuple
    python: Path
    temp_root: Path
    run_root: Path
    publish_lock: Path
    protected_roots: tuple  # complete approved roots; no content/tree scans
    reserve_bytes: int


INPUT_NAMES = {"manifest": "manifest.json", "ready": "ready.json",
               "journal": "journal.json", "plan": "plan.json",
               "qrels": "queries.jsonl", "split": "split.json",
               "freeze": "test_freeze.md"}
REQUIRED_CODE = {"stable_copy.py", "file_lock.py", "rag_core.py"}
PRODUCTION_ROOTS = (Path(r"E:\writing-rag\index"), Path(r"E:\writing-rag\.model-cache"))


def gates(spec, observe=lambda *_: None):
    require(os.name == "nt", "Windows protection required")
    require(set(spec.inputs) == set(INPUT_NAMES), "all seven inputs required")
    require(bool(spec.protected_roots), "production root list required")
    # Validate every supplied spelling before filesystem inspection or locks.
    for p in (spec.snapshot_root, spec.eval_root, spec.db, spec.model,
              *spec.inputs.values(), *spec.code_leaves, spec.python,
              spec.temp_root, spec.run_root, spec.publish_lock,
              *spec.protected_roots, *PRODUCTION_ROOTS):
        absolute(p)
    for p in (spec.snapshot_root, spec.eval_root, spec.db, spec.model,
              *spec.inputs.values(), *spec.code_leaves, spec.python,
              spec.temp_root, spec.run_root, spec.publish_lock,
              *spec.protected_roots, *PRODUCTION_ROOTS):
        volume_path(p)
    require(REQUIRED_CODE <= {Path(p).name for p in spec.code_leaves},
            "explicit required code Leaves missing")
    require(type(spec.reserve_bytes) is int and spec.reserve_bytes >= 0,
            "explicit capacity reserve required")
    containers = (spec.snapshot_root, spec.eval_root, spec.db, spec.model,
                  spec.temp_root)
    leaves = (*spec.inputs.values(), *spec.code_leaves, spec.python,
              spec.db / ".publish-identity.json")
    # FileLock truncates its file on acquisition, before source protection.
    # Reject volume-normalized collisions first, including missing tree locks.
    locks = tuple(absolute(p) for p in (spec.publish_lock, writer_lock_path(spec.db)))
    require(not overlap(*locks), "lock paths overlap")
    for lock in locks:
        require(all(not overlap(lock, source) for source in
                    (spec.db, spec.model, *leaves)), "lock overlaps source")
    for p in containers:
        check_chain(p, "Container", observe=observe)
    for p in leaves:
        check_chain(p, "Leaf", observe=observe)
    # Supplied protection roots may describe absent future subtrees. Check all
    # existing ancestors, stopping at the first absent node, without resolve.
    # Fixed production literals remain string-only: they have no caller alias.
    for p in spec.protected_roots:
        node = Path(absolute(p).anchor)
        for part in absolute(p).parts[1:]:
            check_node(node, "Container")
            node /= part
            try:
                info = node.lstat()
            except FileNotFoundError:
                break
            check_node(node, "Container" if stat.S_ISDIR(info.st_mode) else "Leaf")
    for p in locks:
        check_chain(p.parent, "Container", observe=observe)
        # Existing lock files are Leaves; missing lock files are created by FileLock.
        try:
            check_node(p, "Leaf")
        except FileNotFoundError:
            pass
    check_chain(spec.run_root, "Container", missing_leaf=True, observe=observe)
    require(absolute(spec.run_root).parent == absolute(spec.temp_root),
            "RunRoot must be a direct TempRoot child")
    for p in (spec.db, spec.model):
        require(under(p, spec.snapshot_root), "tree outside approved snapshot root")
    for k, p in spec.inputs.items():
        root = spec.eval_root if k in {"qrels", "split", "freeze"} else spec.snapshot_root
        require(under(p, root), "input outside approved root")
    # Scope roots describe where approved inputs may live. Only the complete
    # copied trees and explicit resources exclude outputs; a common workspace
    # ancestor may also contain the separate TempRoot/RunRoot subtree.
    sources = (spec.db, spec.model, *leaves, *locks)
    require(bool(spec.protected_roots), "production root list required")
    protected = (*spec.protected_roots, *PRODUCTION_ROOTS)
    for p in protected:
        absolute(p)  # syntax check; fixed production literals are never opened
    targets = [spec.run_root / p for p in
               ("db", "model", "inputs", "tmp", "cache", "logs", "results", "owner.json")]
    targets += [spec.run_root / "inputs" / n for n in INPUT_NAMES.values()]
    for p in (spec.run_root, *targets):
        require(all(not overlap(p, s) for s in (*sources, *protected)),
                "target overlaps input/production root")
    # The precreation scan records layout only; the single inventory baseline is
    # taken later, after both complete trees passed and protection is held.
    db = tree(spec.db, observe, inventory=False)
    model = tree(spec.model, observe, inventory=False)
    require("chroma.sqlite3" in db["files"], "complete DB primary required")
    require(bool(model["files"]), "empty model tree")
    return {"db": db, "model": model}


class Protection:
    """Exclusive read handles for files; deny-delete directory handles pin names.

    FILE_SHARE=0 refuses existing readers/writers with data access and prevents
    new opens, including SQLite clients. Copy and hashes use these same handles.
    Directory handles pin directory names; additions are caught by inventories.
    """
    def __init__(self):
        require(os.name == "nt", "Windows protection required")
        import msvcrt
        self.msvcrt = msvcrt
        from ctypes import wintypes
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD,
            wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
        self.api.CreateFileW.restype = wintypes.HANDLE
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api.CloseHandle.restype = wintypes.BOOL
        self.streams, self.dirs = {}, []

    def hold(self, path, role):
        key = absolute(path)
        if key in self.streams or key in [p for p, _ in self.dirs]:
            return
        check_chain(key, role)
        # Directory metadata access pins names without excluding our file opens.
        access, share = (0x80, 3) if role == "Container" else (0x80000000, 0)
        h = self.api.CreateFileW(str(key), access, share, None, 3,
                                 0x02200000, None)  # BACKUP_SEMANTICS|OPEN_REPARSE_POINT
        if h == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())
        if role == "Container":
            self.dirs.append((key, h))
        else:
            try:
                fd = self.msvcrt.open_osfhandle(h, os.O_RDONLY | os.O_BINARY)
            except BaseException:
                self.api.CloseHandle(h)
                raise
            try:
                self.streams[key] = os.fdopen(fd, "rb")
            except BaseException:
                os.close(fd)
                raise
        check_chain(key, role)

    def close(self):
        errors = []
        for stream in self.streams.values():
            try:
                stream.close()
            except BaseException as exc:
                errors.append(exc)
        self.streams.clear()
        for _, h in reversed(self.dirs):
            if not self.api.CloseHandle(h):
                errors.append(ctypes.WinError(ctypes.get_last_error()))
        self.dirs.clear()
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def digest(self, path):
        stream = self.streams[absolute(path)]
        stream.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
        return digest.hexdigest()

    def copy(self, source, target):
        stream = self.streams[absolute(source)]
        stream.seek(0)
        with target.open("xb") as out:
            shutil.copyfileobj(stream, out, 1024 * 1024)
        info = os.fstat(stream.fileno())
        os.utime(target, ns=(info.st_atime_ns, info.st_mtime_ns))


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def snapshot(spec, protection):
    return {"db": tree(spec.db), "model": tree(spec.model), "sha": {
        **{k: protection.digest(p) for k, p in spec.inputs.items()},
        "identity": protection.digest(spec.db / ".publish-identity.json")}}


def creation_identity(root):
    """Require Windows ACL owner to equal this controller's current token user."""
    from ctypes import wintypes as w
    api = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    api.GetNamedSecurityInfoW.argtypes = [w.LPWSTR, w.DWORD, w.DWORD,
        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
    api.GetNamedSecurityInfoW.restype = w.DWORD
    api.OpenProcessToken.argtypes = [w.HANDLE, w.DWORD, ctypes.POINTER(w.HANDLE)]
    api.GetTokenInformation.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p,
        w.DWORD, ctypes.POINTER(w.DWORD)]
    api.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    kernel.GetCurrentProcess.restype = w.HANDLE
    kernel.CloseHandle.argtypes = [w.HANDLE]
    kernel.LocalFree.argtypes = [ctypes.c_void_p]
    owner, descriptor, token = ctypes.c_void_p(), ctypes.c_void_p(), w.HANDLE()
    error = api.GetNamedSecurityInfoW(str(root), 1, 1, ctypes.byref(owner), None,
                                     None, None, ctypes.byref(descriptor))
    if error:
        raise ctypes.WinError(error)
    try:
        if not api.OpenProcessToken(kernel.GetCurrentProcess(), 8, ctypes.byref(token)):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            needed = w.DWORD()
            api.GetTokenInformation(token, 1, None, 0, ctypes.byref(needed))
            require(needed.value > 0, "token identity unavailable")
            buffer = ctypes.create_string_buffer(needed.value)
            if not api.GetTokenInformation(token, 1, buffer, needed, ctypes.byref(needed)):
                raise ctypes.WinError(ctypes.get_last_error())
            user_sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
            require(bool(api.EqualSid(owner, user_sid)), "RunRoot owner differs from controller user")
        finally:
            kernel.CloseHandle(token)
    finally:
        kernel.LocalFree(descriptor)
    info = root.stat()
    return {"device": info.st_dev, "inode": info.st_ino, "owner_matches_token": True}


def stable_copy(spec, *, validate=None, observe=lambda *_: None, checkpoint=lambda *_: None):
    """Hold all resources in this process, release, then copied-only validation.

    validate accepts only RunRoot. Test checkpoints cannot waive any gate.
    No production invocation is performed by importing this module.
    """
    initial = gates(spec, observe)
    with ExitStack() as resources:
        resources.enter_context(FileLock(spec.publish_lock))
        observe("publisher", "held")
        resources.enter_context(FileLock(writer_lock_path(spec.db)))
        observe("writer", "held")
        protection = resources.enter_context(Protection())
        # Pin every existing ancestor before taking descendant protection handles.
        paths = [spec.db, spec.model, *spec.inputs.values()]
        for root in paths:
            for p in reversed(absolute(root).parents):
                protection.hold(p, "Container")
        for kind, root in (("db", spec.db), ("model", spec.model)):
            for name in initial[kind]["dirs"]:
                protection.hold(root / name, "Container")
            for name in initial[kind]["files"]:
                protection.hold(root / name, "Leaf")
        for p in spec.inputs.values():
            protection.hold(p, "Leaf")
        checkpoint("protected", protection)
        # Recheck all creation gates while locks/handles are held.
        require(gates(spec, observe) == initial, "source changed during protection setup")
        before = snapshot(spec, protection)  # the single pre-copy baseline
        checkpoint("source_before", before)
        required_bytes = sum(v[0] for k in ("db", "model") for v in before[k]["files"].values())
        required_bytes += sum(p.stat().st_size for p in spec.inputs.values())
        require(shutil.disk_usage(spec.temp_root).free >= required_bytes + spec.reserve_bytes,
                "insufficient free space")
        spec.run_root.mkdir()  # no parents/exist_ok: partial/existing runs are refused
        check_chain(spec.run_root, "Container")
        identity = creation_identity(spec.run_root)
        token = uuid.uuid4().hex
        owner = {"token": token, "pid": os.getpid(), "root": str(absolute(spec.run_root)), **identity}
        with (spec.run_root / "owner.json").open("x", encoding="utf-8") as f:
            json.dump(owner, f)
        for name in ("db", "model", "inputs", "tmp", "cache", "logs", "results"):
            (spec.run_root / name).mkdir()
            check_chain(spec.run_root / name, "Container")
        for kind, root in (("db", spec.db), ("model", spec.model)):
            for name in before[kind]["dirs"]:
                if name != ".":
                    (spec.run_root / kind / name).mkdir()
            for name in before[kind]["files"]:
                protection.copy(root / name, spec.run_root / kind / name)
        for k, p in spec.inputs.items():
            protection.copy(p, spec.run_root / "inputs" / INPUT_NAMES[k])
        checkpoint("copied", protection)
        require(snapshot(spec, protection) == before, "source_after differs from source_before")
        for kind in ("db", "model"):
            require(tree(spec.run_root / kind) == before[kind], "target inventory differs from baseline")
        target_sha = {k: sha(spec.run_root / "inputs" / name) for k, name in INPUT_NAMES.items()}
        target_sha["identity"] = sha(spec.run_root / "db" / ".publish-identity.json")
        require(target_sha == before["sha"], "target key SHA differs from baseline")
        checkpoint("before_final", protection)
        require(snapshot(spec, protection) == before, "final source recheck differs from source_before")
        observe("final_source_recheck", "PASS")
    observe("source_resources", "released")
    # Only this new root is passed onward; no source callbacks are invoked here.
    result = (validate or validate_copy)(spec.run_root)
    return {"source_before": before, "run_root": str(spec.run_root), "validation": result}


def chunking(manifest, plan, ready):
    required = {"max_tokens": 800, "overlap_tokens": 80, "min_tokens": 16}
    a, b, c = (x.get("chunking") for x in (manifest, plan, ready))
    require(isinstance(a, dict) and a == b, "manifest/plan chunking mismatch")
    for item in (a, b, c):
        require(isinstance(item, dict) and set(required) <= set(item) <= set(required) | {"tokenizer"},
                "unknown/missing chunking field")
        require(all(type(item[k]) is int and item[k] == v for k, v in required.items()),
                "chunking parameters mismatch")
        require("tokenizer" not in item or item["tokenizer"] == "local-model-tokenizer",
                "tokenizer mismatch")
    require(c == a or ("tokenizer" not in c and c == {k: v for k, v in a.items() if k != "tokenizer"}),
            "ready may omit only tokenizer")


def pair(manifest, plan, ready, journal, identity, manifest_sha):
    chunking(manifest, plan, ready)
    require(all(x == manifest_sha for x in (ready.get("manifest_sha256"),
        journal.get("staging_manifest_sha256"), identity.get("manifest_sha256"))), "manifest binding mismatch")
    require(ready.get("status") == "ready" and journal.get("state") == "completed"
            and journal.get("result_pair") == "new", "pair not ready/completed/new")
    require(type(identity.get("schema_version")) is int and identity["schema_version"] == 1,
            "identity schema mismatch")
    require(isinstance(journal.get("pair_id"), str) and bool(journal["pair_id"])
            and journal["pair_id"] == identity.get("pair_id"), "pair identity mismatch")
    for key, expected in (("model", "BAAI/bge-m3"), ("embedding_text_version", "passage-v2")):
        require(all(x.get(key) == expected for x in (manifest, plan, ready)), "model/version mismatch")
    fp = manifest.get("model_fingerprint")
    require(isinstance(fp, str) and re.fullmatch("[0-9a-f]{64}", fp)
            and fp == plan.get("model_fingerprint") == ready.get("model_fingerprint"), "model fingerprint mismatch")
    files, items = manifest.get("files"), plan.get("sources")
    require(isinstance(files, dict) and bool(files) and isinstance(items, list) and bool(items), "empty sources")
    names = [x["source"] for x in items]
    require(len(names) == len(set(names)) and set(names) == set(files), "source set mismatch")
    total, canonical = 0, 0
    for item in items:
        name, digest, count = item["source"], item["sha256"], item["estimated_chunks"]
        e = files[name]
        require(isinstance(digest, str) and re.fullmatch("[0-9a-f]{64}", digest), "source SHA invalid")
        require(type(count) is int and count >= 0 and e.get("sha256") == digest
                and e.get("chunks") == count and type(e.get("chunks")) is int, "source SHA/chunks mismatch")
        require(e.get("embedding_text_version") == "passage-v2", "source version mismatch")
        c = item["canonical_source"]
        require(type(item.get("is_canonical")) is bool and item["is_canonical"] == (c == name), "canonical flag mismatch")
        if c == name:
            canonical += 1
            require(count > 0 and not e.get("dup_of"), "canonical role mismatch")
        else:
            require(c in files and not files[c].get("dup_of") and files[c].get("chunks", 0) > 0
                    and files[c]["sha256"] == digest and count == 0 and e.get("dup_of") == c,
                    "alias/dup_of mismatch")
        total += count
    corpus = json.dumps(sorted([n, files[n]["sha256"]] for n in names), ensure_ascii=False, separators=(",", ":"))
    fingerprint = hashlib.sha256(corpus.encode()).hexdigest()
    require(fingerprint == manifest.get("sources_fingerprint") == plan.get("sources_fingerprint")
            == ready.get("corpus_fingerprint"), "sources fingerprint mismatch")
    fields = ((manifest, "total_chunks", total), (plan, "planned_chunks", total),
              (plan, "canonical_source_count", canonical), (ready, "total_chunks", total),
              (ready, "source_entries", len(names)), (ready, "corpus_source_count", len(names)),
              (ready, "chunked_sources", canonical))
    require(all(type(d.get(k)) is int and d[k] == v for d, k, v in fields), "aggregate mismatch")
    return {"entries": len(names), "sources": canonical, "chunks": total, "model_fingerprint": fp}


def freeze(qrels, split, frozen):
    require(isinstance(qrels, list) and all(isinstance(r, dict) for r in qrels), "qrels invalid")
    ids = [r.get("query_id") for r in qrels]
    require(all(isinstance(i, str) and i for i in ids), "query IDs invalid")
    require(len(ids) == len(set(ids)) == 100, "qrels count/identity mismatch")
    dev = [r for r in qrels if r.get("split") == "dev"]
    test = [r for r in qrels if r.get("split") == "test"]
    require(len(dev) == 70 and len(test) == 30, "split counts mismatch")
    positive = 0
    for r in qrels:
        js = r.get("judgements")
        require(isinstance(js, list) and len(js) == 3 and all(isinstance(j, dict) and type(j.get("relevance")) is int
                and 0 <= j["relevance"] <= 3 for j in js), "judgements invalid")
        positive += r["split"] == "dev" and any(j["relevance"] >= 2 for j in js)
    require(positive == 60, "dev positive/no-positive mismatch")
    for obj in (split, frozen):
        require(isinstance(obj, dict), "freeze object invalid")
        test_ids = obj.get("test_query_ids")
        require(isinstance(test_ids, list) and all(isinstance(i, str) and i for i in test_ids)
                and len(test_ids) == len(set(test_ids)) == 30
                and set(test_ids) == {r["query_id"] for r in test}, "locked test IDs mismatch")
        gains, counts = obj.get("ndcg_gain"), obj.get("split_counts")
        require(obj.get("status") == "locked" and "locked" not in obj and "gains" not in obj
                and obj.get("frozen_at") == "2026-09-21T23:48:43+08:00"
                and type(obj.get("positive_threshold")) is int and obj["positive_threshold"] == 2
                and isinstance(gains, dict) and gains == {"0": 0, "1": 0, "2": 2, "3": 3}
                and all(type(v) is int for v in gains.values())
                and isinstance(counts, dict) and counts == {"dev": 70, "test": 30}
                and all(type(v) is int for v in counts.values()), "freeze contract mismatch")
    return {"queries": 100, "dev": 70, "test": 30, "dev_positive": 60, "dev_no_positive": 10}


def validate_copy(root, *, fingerprint=None, preflight=None, expected_counts=(2499, 2492, 79239)):
    """Bind environment before importing rag_core; validate only copied inputs."""
    root = absolute(root)
    previous = {k: os.environ.get(k) for k in ("TEMP", "TMP", "HF_HOME", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME")}
    previous_temp = tempfile.tempdir
    try:
        for k in previous:
            os.environ[k] = str(root / ("tmp" if k in {"TEMP", "TMP"} else "cache"))
        tempfile.tempdir = None
        require(absolute(tempfile.gettempdir()) == root / "tmp", "TEMP fell back outside RunRoot")
        if fingerprint is None or preflight is None:
            try:
                from . import rag_core
            except ImportError:
                import rag_core
            fingerprint = fingerprint or rag_core.model_fingerprint
            preflight = preflight or rag_core.index_preflight
        return _validate_copy(root, fingerprint=fingerprint, preflight=preflight, expected_counts=expected_counts)
    finally:
        tempfile.tempdir = previous_temp
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _validate_copy(root, *, fingerprint, preflight, expected_counts):
    """Callbacks must use explicit copied paths (rag_core functions in real use)."""
    root = absolute(root)
    def read(name):
        return json.loads((root / "inputs" / INPUT_NAMES[name]).read_text(encoding="utf-8-sig"))
    manifest, plan, ready, journal = (read(k) for k in ("manifest", "plan", "ready", "journal"))
    identity = json.loads((root / "db" / ".publish-identity.json").read_text(encoding="utf-8-sig"))
    summary = pair(manifest, plan, ready, journal, identity, sha(root / "inputs" / "manifest.json"))
    require((summary["entries"], summary["sources"], summary["chunks"]) == expected_counts,
            "copied pair differs from approved counts")
    qrels = [json.loads(line) for line in (root / "inputs" / "queries.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    blocks = re.findall(r"```json\s*(.*?)```", (root / "inputs" / "test_freeze.md").read_text(encoding="utf-8-sig"), re.S)
    require(len(blocks) == 1, "exactly one machine-readable freeze block required")
    frozen = freeze(qrels, read("split"), json.loads(blocks[0]))
    require(fingerprint(root / "model") == summary["model_fingerprint"], "copied model fingerprint mismatch")
    # The existing fingerprint uses size/path for files >1 MiB; no byte identity claim.
    pf = preflight(db_dir=root / "db", model_dir=root / "model", manifest_path=root / "inputs" / "manifest.json", collection_name="papers", device="cpu")
    for key, path in (("db_dir", root / "db"), ("model_path", root / "model"), ("manifest_path", root / "inputs" / "manifest.json")):
        require(absolute(pf[key]) == path, "preflight returned other paths")
    require(pf.get("collection") == "papers" and pf.get("collection_count") == summary["chunks"]
            and pf.get("source_count") == summary["sources"] and pf.get("manifest_file_count") == summary["entries"]
            and pf.get("manifest_total_chunks") == summary["chunks"]
            and pf.get("model_fingerprint") == summary["model_fingerprint"], "preflight summary mismatch")
    return {"pair": summary, "freeze": frozen, "preflight": pf}
