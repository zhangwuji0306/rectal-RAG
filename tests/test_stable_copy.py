"""Synthetic-only Windows stable-copy validation; no production data imports."""
import copy
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch
from index import stable_copy as sc


def write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


def fixture(base):
    source = base / "workspace"
    ev, temp = source / "evaluation", source / ".claude" / "tmp"
    for p in (source, ev, temp):
        p.mkdir(parents=True)
    db, model = source / "index" / "chroma_db", source / ".model-cache" / "model"
    db.mkdir(parents=True)
    model.mkdir(parents=True)
    with sqlite3.connect(db / "chroma.sqlite3") as con:
        con.execute("create table fixture(x)")
        con.execute("insert into fixture values(1)")
    con.close()
    (db / "hnsw").mkdir()
    (db / "hnsw" / "data.bin").write_bytes(b"synthetic-hnsw")
    (model / "config.json").write_bytes(b"synthetic-model")
    (model / "empty").mkdir()
    digest = "a" * 64
    files = {"PMID_fixture.md": {"sha256": digest, "chunks": 2, "embedding_text_version": "passage-v2"},
             "alias.md": {"sha256": digest, "chunks": 0, "dup_of": "PMID_fixture.md", "embedding_text_version": "passage-v2"}}
    items = [{"source": n, "sha256": digest, "estimated_chunks": e["chunks"],
              "canonical_source": e.get("dup_of", n), "is_canonical": not bool(e.get("dup_of"))} for n, e in files.items()]
    fp = hashlib.sha256(json.dumps(sorted([n, digest] for n in files), separators=(",", ":")).encode()).hexdigest()
    common = {"model": "BAAI/bge-m3", "model_fingerprint": "b" * 64, "embedding_text_version": "passage-v2",
              "chunking": {"max_tokens": 800, "overlap_tokens": 80, "min_tokens": 16, "tokenizer": "local-model-tokenizer"}}
    manifest = {**copy.deepcopy(common), "files": files, "total_chunks": 2, "sources_fingerprint": fp}
    plan = {**copy.deepcopy(common), "sources": items, "planned_chunks": 2, "canonical_source_count": 1, "sources_fingerprint": fp}
    inputs = {k: (ev if k in {"qrels", "split", "freeze"} else db.parent) / n for k, n in sc.INPUT_NAMES.items()}
    write(inputs["manifest"], manifest)
    digest_manifest = sc.sha(inputs["manifest"])
    ready = {**copy.deepcopy(common), "status": "ready", "manifest_sha256": digest_manifest,
             "total_chunks": 2, "source_entries": 2, "chunked_sources": 1, "corpus_source_count": 2, "corpus_fingerprint": fp}
    del ready["chunking"]["tokenizer"]
    journal = {"state": "completed", "result_pair": "new", "pair_id": "fixture-pair", "staging_manifest_sha256": digest_manifest}
    identity = {"schema_version": 1, "pair_id": "fixture-pair", "manifest_sha256": digest_manifest}
    write(db / ".publish-identity.json", identity)
    for k, obj in (("plan", plan), ("ready", ready), ("journal", journal)):
        write(inputs[k], obj)
    rows = [{"query_id": f"synthetic-{i}", "split": "dev" if i < 70 else "test",
             "judgements": [{"relevance": 2 if i < 60 else 0}, {"relevance": 0}, {"relevance": 1}]} for i in range(100)]
    inputs["qrels"].write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    frozen = {"test_query_ids": [r["query_id"] for r in rows[70:]], "status": "locked",
              "frozen_at": "2026-09-21T23:48:43+08:00", "positive_threshold": 2,
              "ndcg_gain": {"0": 0, "1": 0, "2": 2, "3": 3}, "split_counts": {"dev": 70, "test": 30}}
    write(inputs["split"], frozen)
    inputs["freeze"].write_text("```json\n" + json.dumps(frozen) + "\n```\n", encoding="utf-8")
    code = source / "code"
    code.mkdir()
    leaves = tuple(code / n for n in sorted(sc.REQUIRED_CODE))
    for p in leaves:
        p.write_bytes(b"synthetic code Leaf")
    python = source / ".venv" / "Scripts" / "python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"synthetic interpreter Leaf")
    spec = sc.CopySpec(source, ev, db, model, inputs, leaves, python, temp, temp / "unique-run",
        db.parent / "publish.lock", (db.parent, model.parent, ev, python.parent.parent, code), 1024)
    return spec, (manifest, plan, ready, journal, identity, digest_manifest), (rows, frozen)


class SchemaTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.spec, self.pair, self.freeze = fixture(Path(self.tmp.name))

    def test_chunking_six_cases(self):
        sc.chunking(*self.pair[:3])
        for case in range(5):
            a, b, c = copy.deepcopy(self.pair[:3])
            if case == 0:
                c["chunking"]["overlap_tokens"] = 81
            elif case == 1:
                c["chunking"]["extra"] = 1
            elif case == 2:
                for x in (a, b, c):
                    x["chunking"]["extra"] = 1
            elif case == 3:
                a["chunking"]["tokenizer"] = b["chunking"]["tokenizer"] = "other"
            else:
                b["chunking"]["min_tokens"] = 17
            with self.subTest(case=case), self.assertRaises(sc.CopyRefused):
                sc.chunking(a, b, c)

    def test_pair_bindings_alias_aggregates(self):
        self.assertEqual(sc.pair(*self.pair)["sources"], 1)
        for index, key, value in ((2, "manifest_sha256", "bad"), (3, "state", "ready"),
                                  (4, "pair_id", "other"), (2, "source_entries", 3), (0, "sources_fingerprint", "bad")):
            values = copy.deepcopy(self.pair)
            values[index][key] = value
            with self.subTest(key=key), self.assertRaises(sc.CopyRefused):
                sc.pair(*values)
        values = copy.deepcopy(self.pair)
        values[0]["files"]["alias.md"]["dup_of"] = "absent"
        with self.assertRaises(sc.CopyRefused):
            sc.pair(*values)

    def test_freeze_exact_ids_values(self):
        rows, frozen = self.freeze
        sc.freeze(rows, frozen, frozen)
        for key, value in (("status", "unlocked"), ("positive_threshold", 1), ("ndcg_gain", {}),
                           ("test_query_ids", frozen["test_query_ids"][:-1])):
            with self.subTest(key=key), self.assertRaises(sc.CopyRefused):
                sc.freeze(rows, frozen, {**frozen, key: value})

    def assert_freeze_refused(self, bad):
        rows, frozen = self.freeze
        for side in ("split", "freeze"):
            with self.subTest(side=side), self.assertRaises(sc.CopyRefused):
                sc.freeze(rows, bad if side == "split" else frozen,
                          bad if side == "freeze" else frozen)

    def test_freeze_required_fields_and_old_schema_refused(self):
        _, frozen = self.freeze
        for key in frozen:
            bad = copy.deepcopy(frozen)
            del bad[key]
            with self.subTest(missing=key):
                self.assert_freeze_refused(bad)
        old = copy.deepcopy(frozen)
        old["locked"] = True
        old["gains"] = old.pop("ndcg_gain")
        del old["status"]
        self.assert_freeze_refused(old)
        for key, value in (("locked", True), ("gains", frozen["ndcg_gain"])):
            with self.subTest(legacy_extra=key):
                self.assert_freeze_refused({**frozen, key: value})

    def test_freeze_status_threshold_time_types(self):
        _, frozen = self.freeze
        for key, values in (("status", ("ready", "LOCKED", True, None)),
                            ("positive_threshold", (1, 3, True, 2.0, "2", None)),
                            ("frozen_at", ("2026-09-21T23:48:44+08:00", "2026-09-21T15:48:43Z", None))):
            for value in values:
                with self.subTest(key=key, value=value):
                    self.assert_freeze_refused({**frozen, key: value})

    def test_freeze_gain_exact_keys_values_types(self):
        _, frozen = self.freeze
        valid = frozen["ndcg_gain"]
        cases = [None, [], {}, {int(k): v for k, v in valid.items()},
                 {**valid, "4": 4}]
        for key in valid:
            cases.extend(({k: v for k, v in valid.items() if k != key},
                          {**valid, key: 1}, {**valid, key: float(valid[key])},
                          {**valid, key: str(valid[key])}, {**valid, key: False}))
        for gain in cases:
            with self.subTest(gain=gain):
                self.assert_freeze_refused({**frozen, "ndcg_gain": gain})

    def test_freeze_split_counts_exact_keys_values_types(self):
        _, frozen = self.freeze
        for counts in (None, [], {}, {"dev": 70}, {"dev": 69, "test": 31},
                       {"dev": 70.0, "test": 30}, {"dev": 70, "test": "30"},
                       {"dev": 70, "test": True}, {"dev": 70, "test": 30, "extra": 0}):
            with self.subTest(counts=counts):
                self.assert_freeze_refused({**frozen, "split_counts": counts})

    def test_freeze_test_ids_exact_unique_strings(self):
        rows, frozen = self.freeze
        ids = frozen["test_query_ids"]
        sc.freeze(rows, {**frozen, "test_query_ids": list(reversed(ids))}, frozen)
        for bad_ids in (None, tuple(ids), ids[:-1], ids + ["extra"],
                        ids[:-1] + [ids[0]], ids[:-1] + ["other"],
                        ids[:-1] + [None], ids[:-1] + [[]], ids[:-1] + [""]):
            with self.subTest(ids=bad_ids):
                self.assert_freeze_refused({**frozen, "test_query_ids": bad_ids})

    def test_freeze_qrels_counts_split_positive_and_shapes(self):
        rows, frozen = self.freeze
        for case in ("count", "duplicate", "id", "split", "positive", "judgements", "relevance"):
            bad = copy.deepcopy(rows)
            if case == "count":
                bad.pop()
            elif case == "duplicate":
                bad[-1]["query_id"] = bad[0]["query_id"]
            elif case == "id":
                bad[0]["query_id"] = []
            elif case == "split":
                bad[0]["split"] = "test"
            elif case == "positive":
                bad[0]["judgements"][0]["relevance"] = 0
            elif case == "judgements":
                bad[0]["judgements"][0] = None
            else:
                bad[0]["judgements"][0]["relevance"] = True
            with self.subTest(case=case), self.assertRaises(sc.CopyRefused):
                sc.freeze(bad, frozen, frozen)
        for bad in (None, [], "locked"):
            with self.subTest(object=bad):
                self.assert_freeze_refused(bad)

    def test_invalid_copied_freeze_stops_before_fingerprint_preflight(self):
        for p in (self.spec.run_root / "inputs", self.spec.run_root / "db",
                  self.spec.run_root / "tmp", self.spec.run_root / "cache"):
            p.mkdir(parents=True)
        for source in self.spec.inputs.values():
            (self.spec.run_root / "inputs" / source.name).write_bytes(source.read_bytes())
        identity = self.spec.db / ".publish-identity.json"
        (self.spec.run_root / "db" / identity.name).write_bytes(identity.read_bytes())
        _, frozen = self.freeze
        for key in ("split", "freeze"):
            path = self.spec.run_root / "inputs" / sc.INPUT_NAMES[key]
            original = path.read_bytes()
            bad = {**frozen, "status": "unlocked"}
            if key == "split":
                write(path, bad)
            else:
                path.write_text("```json\n" + json.dumps(bad) + "\n```\n", encoding="utf-8")
            with self.subTest(side=key), patch.object(sc, "pair", wraps=sc.pair) as pair:
                fingerprint, preflight = Mock(), Mock()
                with self.assertRaisesRegex(sc.CopyRefused, "freeze contract mismatch"):
                    sc.validate_copy(self.spec.run_root, fingerprint=fingerprint,
                                     preflight=preflight, expected_counts=(2, 1, 2))
                pair.assert_called_once()
                fingerprint.assert_not_called()
                preflight.assert_not_called()
            path.write_bytes(original)


@unittest.skipUnless(os.name == "nt", "Windows share semantics required")
class ControllerTests(unittest.TestCase):
    setUp = SchemaTests.setUp

    def assert_alias_refused(self, spec):
        before = self.source_state()
        locks = (self.spec.publish_lock, sc.writer_lock_path(self.spec.db))
        existed = {p: p.exists() for p in locks}
        events = []
        with patch.object(sc, "FileLock") as lock, \
                patch.object(sc, "Protection") as protection, \
                patch.object(sc, "snapshot") as baseline, \
                patch.object(Path, "mkdir") as mkdir, \
                patch.object(sc, "validate_copy") as validator:
            with self.assertRaises(sc.CopyRefused):
                sc.stable_copy(spec, validate=validator,
                               checkpoint=lambda *args: events.append(args))
            for mock in (lock, protection, baseline, mkdir, validator):
                mock.assert_not_called()
        self.assertEqual(events, [])
        self.assertEqual(before, self.source_state())
        self.assertEqual(existed, {p: p.exists() for p in locks})
        self.assertFalse(self.spec.run_root.exists())
        self.assertFalse(spec.run_root.exists())

    def test_extended_model_output_alias_refused(self):
        alias = Path("\\\\?\\" + str(self.spec.model))
        self.assertTrue(os.path.samefile(alias, self.spec.model))
        run = alias / "new-output"
        self.assert_alias_refused(replace(self.spec, temp_root=alias, run_root=run))
        self.assertFalse(run.exists())

    def test_extended_manifest_lock_alias_refused(self):
        source = self.spec.inputs["manifest"]
        alias = Path("\\\\?\\" + str(source))
        self.assertTrue(os.path.samefile(alias, source))
        before = (alias.read_bytes(), alias.stat().st_mtime_ns)
        self.assert_alias_refused(replace(self.spec, publish_lock=alias))
        self.assertEqual(before, (alias.read_bytes(), alias.stat().st_mtime_ns))
        # Alias already denotes the input; no distinct lock was created.
        self.assertTrue(os.path.samefile(alias, source))

    def test_all_spec_and_protected_paths_reject_namespaces(self):
        for field in ("snapshot_root", "eval_root", "db", "model", "python",
                      "temp_root", "run_root", "publish_lock"):
            with self.subTest(field=field):
                self.assert_alias_refused(replace(self.spec,
                    **{field: Path("\\\\?\\" + str(getattr(self.spec, field)))}))
        for key, source in self.spec.inputs.items():
            with self.subTest(input=key):
                self.assert_alias_refused(replace(self.spec,
                    inputs={**self.spec.inputs, key: Path("\\\\?\\" + str(source))}))
        self.assert_alias_refused(replace(self.spec, code_leaves=(
            Path("\\\\?\\" + str(self.spec.code_leaves[0])), *self.spec.code_leaves[1:])))
        self.assert_alias_refused(replace(self.spec, protected_roots=(
            Path("\\\\?\\" + str(self.spec.temp_root)),)))
        with patch.object(sc, "PRODUCTION_ROOTS", (Path("\\\\?\\" + str(self.spec.temp_root)),)):
            self.assert_alias_refused(self.spec)

    def test_win32_component_aliases_refused(self):
        for name in ("model.", "model ", "config.json:stream", "MODEL~1",
                     "NUL", "con.txt", "COM1.log", "LPT²", "CONOUT$", "bad\x01name",
                     "bad?name", ".."):
            with self.subTest(name=name):
                self.assert_alias_refused(replace(self.spec,
                    publish_lock=self.spec.temp_root / name))
        for raw in (r"\\.\E:\device", r"\??\E:\device", r"\\server\share\lock",
                    r"E:relative", r"\rooted"):
            with self.subTest(raw=raw):
                self.assert_alias_refused(replace(self.spec, publish_lock=Path(raw)))

    def test_mixed_separators_and_dot_normalize_same_resource(self):
        source = self.spec.inputs["manifest"]
        mixed = str(source).replace("\\", "/")
        self.assertEqual(sc.absolute(mixed), source)
        self.assertEqual(sc.absolute(source.parent / "." / source.name), source)
        self.assertTrue(sc.overlap(mixed.upper(), source))
        self.assert_lock_collision_refused(replace(self.spec, publish_lock=Path(mixed)))

    def test_arbitrary_short_name_rejected_after_parent_check(self):
        # Volumes may disable 8.3 generation. Emulate only the one Win32 name
        # response, including an arbitrary short alias without a tilde.
        from unittest.mock import Mock
        api = Mock()
        def find(_, data):
            data._obj.name = "canonical-long-name"
            return 123
        api.FindFirstFileW.side_effect = find
        api.FindClose.return_value = True
        visited = []
        original = sc.check_node
        def node(path, role):
            visited.append(path)
            if path == self.spec.model:
                with patch.object(sc.ctypes, "WinDLL", return_value=api):
                    return original(path, role)
            return original(path, role)
        with patch.object(sc, "check_node", node):
            self.assert_alias_refused(self.spec)
        self.assertIn(self.spec.model.parent, visited)
        self.assertLess(visited.index(self.spec.model.parent), visited.index(self.spec.model))
        api.FindClose.assert_called_once_with(123)

    def test_existing_protected_root_long_name_checked(self):
        protected = self.spec.snapshot_root / "protected-long-name"
        protected.mkdir()
        original = sc.check_long_name
        def check(path):
            if path == protected:
                raise sc.CopyRefused("short-name alias refused")
            return original(path)
        with patch.object(sc, "check_long_name", check):
            self.assert_alias_refused(replace(self.spec, protected_roots=(protected,)))

    def test_dos_drive_alias_mapping_refused_or_compared_by_volume(self):
        from unittest.mock import Mock
        api = Mock()
        mapping = [r"\??\E:\substituted"]
        def query(_, buffer, size):
            buffer.value = mapping[0]
            return len(mapping[0]) + 2
        api.QueryDosDeviceW.side_effect = query
        with patch.object(sc.ctypes, "WinDLL", return_value=api):
            self.assert_alias_refused(self.spec)
            mapping[0] = r"\Device\HarddiskVolume123"
            # Different letters for the same local volume compare identically.
            self.assertTrue(sc.overlap(r"C:\same\child", r"E:\same"))
            self.assertFalse(sc.overlap(r"C:\same-extra", r"E:\same"))

    def source_state(self):
        paths = [p for root in (self.spec.db, self.spec.model)
                 for p in root.rglob("*") if p.is_file()]
        paths += [*self.spec.inputs.values(), *self.spec.code_leaves, self.spec.python]
        return {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}

    def assert_lock_collision_refused(self, spec, message="lock overlaps source"):
        before = self.source_state()
        locks = (spec.publish_lock, sc.writer_lock_path(spec.db))
        lock_exists = {p: p.exists() for p in locks}
        with patch.object(sc, "FileLock", wraps=sc.FileLock) as lock, \
                patch.object(sc, "validate_copy") as validator:
            with self.assertRaisesRegex(sc.CopyRefused, message):
                sc.stable_copy(spec, validate=validator)
            lock.assert_not_called()
            validator.assert_not_called()
        self.assertEqual(before, self.source_state())
        self.assertEqual(lock_exists, {p: p.exists() for p in locks})
        self.assertFalse(spec.run_root.exists())

    def test_publisher_lock_source_leaf_collisions(self):
        sources = (*self.spec.inputs.values(), *self.spec.code_leaves, self.spec.python,
                   self.spec.db / ".publish-identity.json", self.spec.db / "chroma.sqlite3",
                   self.spec.model / "config.json")
        for source in sources:
            with self.subTest(source=source.name):
                self.assert_lock_collision_refused(replace(self.spec, publish_lock=source))

    def test_publisher_lock_tree_overlaps(self):
        for root in (self.spec.db, self.spec.model):
            for lock in (root, root / "new.lock", root.parent):
                with self.subTest(root=root.name, lock=lock):
                    self.assert_lock_collision_refused(replace(self.spec, publish_lock=lock),
                                                      "lock (paths overlap|overlaps source)")

    def test_writer_lock_source_leaf_collisions(self):
        writer = sc.writer_lock_path(self.spec.db)
        writer.write_bytes(b"synthetic source at writer sidecar")
        for key in self.spec.inputs:
            with self.subTest(input=key):
                inputs = {**self.spec.inputs, key: writer}
                self.assert_lock_collision_refused(replace(self.spec, inputs=inputs))
        # Retain the required named Leaves so the collision gate is reached.
        self.assert_lock_collision_refused(replace(self.spec,
            code_leaves=(*self.spec.code_leaves, writer)))
        self.assert_lock_collision_refused(replace(self.spec, python=writer))

    def test_writer_lock_model_tree_overlap(self):
        self.assert_lock_collision_refused(replace(self.spec, model=self.spec.snapshot_root))

    def test_publisher_writer_lock_collision_lexical_casefold(self):
        writer = sc.writer_lock_path(self.spec.db)
        writer.write_bytes(b"existing lock bytes")
        for lock in (writer, writer.parent / "." / writer.name.upper()):
            with self.subTest(lock=lock):
                self.assert_lock_collision_refused(replace(self.spec, publish_lock=lock),
                                                  "lock paths overlap")

    def reader(self, sqlite=False):
        code = ("import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); c.execute('select * from fixture').fetchall(); c.close()"
                if sqlite else "import sys; f=open(sys.argv[1],'rb'); f.read(1); f.close()")
        return subprocess.run([sys.executable, "-B", "-c", code, str(self.spec.db / "chroma.sqlite3")],
                              capture_output=True).returncode

    def released(self):
        self.assertEqual(self.reader(), 0)
        self.assertEqual(self.reader(sqlite=True), 0)
        with sc.FileLock(self.spec.publish_lock), sc.FileLock(sc.writer_lock_path(self.spec.db)):
            pass

    def test_root_first_reparse_stops_before_child(self):
        checked, original = [], sc.check_node
        def node(p, role):
            checked.append(p)
            if p == self.spec.db.parent:
                raise sc.CopyRefused("simulated ancestor reparse")
            return original(p, role)
        with patch.object(sc, "check_node", node), self.assertRaises(sc.CopyRefused):
            sc.check_chain(self.spec.db / "chroma.sqlite3", "Leaf")
        self.assertEqual(checked[0], Path(self.spec.db.anchor))
        self.assertNotIn(self.spec.db, checked)

    def test_gates_before_create_existing_root_refused(self):
        for bad in (replace(self.spec, python=self.spec.db),
                    replace(self.spec, protected_roots=(self.spec.temp_root,))):
            with self.assertRaises(sc.CopyRefused):
                sc.stable_copy(bad, validate=lambda _: self.fail("validation reached"))
            self.assertFalse(self.spec.run_root.exists())
        self.spec.run_root.mkdir()
        (self.spec.run_root / "retain").write_bytes(b"old")
        with self.assertRaises(sc.CopyRefused):
            sc.stable_copy(self.spec, validate=lambda _: None)
        self.assertEqual((self.spec.run_root / "retain").read_bytes(), b"old")

    def test_output_inside_source_or_protected_tree_refused_before_lock(self):
        roots = (self.spec.db, self.spec.model, self.spec.db.parent,
                 self.spec.eval_root, self.spec.python.parent.parent,
                 self.spec.code_leaves[0].parent, *self.spec.protected_roots)
        for root in dict.fromkeys(roots):
            with self.subTest(root=root):
                self.assert_lock_collision_refused(
                    replace(self.spec, temp_root=root, run_root=root / "new-output"),
                    "target overlaps input/production root")

    def test_output_exact_resource_overlap_gate(self):
        # Existing Leaves normally stop at the new-root check first. Bypass
        # only that check here to independently exercise the overlap gate.
        resources = (*self.spec.inputs.values(), *self.spec.code_leaves,
                     self.spec.python, self.spec.db / ".publish-identity.json",
                     self.spec.publish_lock, sc.writer_lock_path(self.spec.db))
        original = sc.check_chain
        def chain(path, role, **kwargs):
            if not kwargs.get("missing_leaf"):
                return original(path, role, **kwargs)
        for resource in resources:
            with self.subTest(resource=resource):
                before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
                          for p in resources if p.exists()}
                source_before = self.source_state()
                existed = resource.exists()
                spec = replace(self.spec, temp_root=resource.parent, run_root=resource)
                with patch.object(sc, "check_chain", chain), \
                        patch.object(sc, "FileLock") as lock, \
                        patch.object(sc, "validate_copy") as validator, \
                        patch.object(Path, "mkdir") as mkdir:
                    with self.assertRaisesRegex(sc.CopyRefused, "target overlaps input/production root"):
                        sc.stable_copy(spec, validate=validator)
                    lock.assert_not_called()
                    validator.assert_not_called()
                    mkdir.assert_not_called()
                self.assertEqual(resource.exists(), existed)
                self.assertEqual(before, {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in before})
                self.assertEqual(source_before, self.source_state())
                self.assertFalse(self.spec.run_root.exists())

    def test_output_boundary_and_scope_roots(self):
        temp = self.spec.snapshot_root / "index-extra"
        temp.mkdir()
        spec = replace(self.spec, eval_root=self.spec.snapshot_root,
                       temp_root=temp, run_root=temp / "unique-output")
        # A shared scope root and a lexical prefix of protected index are safe.
        sc.gates(spec)
        self.assertFalse(spec.run_root.exists())
        for resource in (self.spec.db, self.spec.model, *self.spec.inputs.values(),
                         self.spec.python, *self.spec.code_leaves,
                         self.spec.publish_lock, sc.writer_lock_path(self.spec.db),
                         *self.spec.protected_roots):
            self.assertTrue(sc.overlap(resource, Path(str(resource).upper())))
            self.assertTrue(sc.overlap(resource, resource.parent))
            self.assertTrue(sc.overlap(resource, resource / "child"))
            self.assertFalse(sc.overlap(resource, resource.with_name(resource.name + "-extra")))

    def test_queue_checks_before_enqueue(self):
        original, scan, visited = sc.check_node, sc.os.scandir, []
        def node(p, role):
            if p == self.spec.db / "hnsw":
                raise sc.CopyRefused("simulated child reparse")
            return original(p, role)
        def record(p):
            visited.append(Path(p))
            return scan(p)
        with patch.object(sc, "check_node", node), patch.object(sc.os, "scandir", record), self.assertRaises(sc.CopyRefused):
            sc.tree(self.spec.db)
        self.assertNotIn(self.spec.db / "hnsw", visited)

    def test_real_junction_refused_before_create(self):
        link = self.spec.db / "junction"
        result = subprocess.run(["cmd.exe", "/c", "mklink", "/J", str(link), str(self.spec.model)], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr.decode(errors="replace"))
        try:
            with self.assertRaises(sc.CopyRefused):
                sc.stable_copy(self.spec, validate=lambda _: self.fail("junction followed"))
            self.assertFalse(self.spec.run_root.exists())
        finally:
            os.rmdir(link)

    def test_writer_acquisition_failure_releases_publisher(self):
        with sc.FileLock(sc.writer_lock_path(self.spec.db)):
            with self.assertRaises(RuntimeError):
                sc.stable_copy(self.spec, validate=lambda _: self.fail("validation reached"))
            with sc.FileLock(self.spec.publish_lock):
                pass
        self.assertFalse(self.spec.run_root.exists())

    def test_exception_after_protection_releases_every_handle(self):
        def checkpoint(stage, _):
            if stage == "protected":
                raise RuntimeError("synthetic protected-window exception")
        with self.assertRaisesRegex(RuntimeError, "protected-window"):
            sc.stable_copy(self.spec, checkpoint=checkpoint, validate=lambda _: self.fail("validation reached"))
        self.assertFalse(self.spec.run_root.exists())
        self.released()

    def test_validation_failure_occurs_after_release(self):
        def validate(_):
            self.released()
            raise RuntimeError("synthetic validator exception")
        with self.assertRaisesRegex(RuntimeError, "validator exception"):
            sc.stable_copy(self.spec, validate=validate)
        self.released()

    def test_full_window_baseline_copy_release_copied_only_validation(self):
        self.assertTrue(sc.under(self.spec.run_root, self.spec.snapshot_root))
        self.assertEqual(self.spec.run_root.parent, self.spec.temp_root)
        events, baselines = [], []
        before_sha = sc.sha(self.spec.inputs["manifest"])
        def checkpoint(stage, value):
            events.append(stage)
            if stage == "source_before":
                baselines.append(copy.deepcopy(value))
                self.assertFalse(self.spec.run_root.exists())
                self.assertEqual(value["sha"]["manifest"], before_sha)
                self.assertEqual(value["db"]["files"]["chroma.sqlite3"][1], (self.spec.db / "chroma.sqlite3").stat().st_mtime_ns)
            else:
                self.assertTrue(value.streams)
                self.assertNotEqual(self.reader(), 0)
                self.assertNotEqual(self.reader(sqlite=True), 0)
        def validate(root):
            self.released()
            self.assertEqual(events[-1], "source_resources:released")
            self.assertEqual(root, self.spec.run_root)
            def pf(**kw):
                self.assertTrue(all(sc.under(kw[k], root) for k in ("db_dir", "model_dir", "manifest_path")))
                return {"db_dir": str(kw["db_dir"]), "model_path": str(kw["model_dir"]), "manifest_path": str(kw["manifest_path"]),
                        "collection": "papers", "collection_count": 2, "source_count": 1, "manifest_file_count": 2,
                        "manifest_total_chunks": 2, "model_fingerprint": "b" * 64}
            original = Path.open
            def guard(p, *a, **kw):
                if any(sc.under(p, tree) for tree in (self.spec.db, self.spec.model)) or p in self.spec.inputs.values():
                    self.fail("source reopened after release")
                return original(p, *a, **kw)
            with patch.object(Path, "open", guard):
                return sc.validate_copy(root, fingerprint=lambda _: "b" * 64, preflight=pf, expected_counts=(2, 1, 2))
        result = sc.stable_copy(self.spec, validate=validate, checkpoint=checkpoint,
                               observe=lambda p, r: events.append(f"{p}:{r}"))
        self.assertEqual(result["source_before"], baselines[0])
        self.assertLess(events.index("publisher:held"), events.index("writer:held"))
        self.assertEqual(events.count("source_before"), 1)

    def test_existing_reader_fail_closed_cleanup(self):
        with (self.spec.db / "chroma.sqlite3").open("rb"):
            with self.assertRaises(OSError):
                sc.stable_copy(self.spec, validate=lambda _: self.fail("validation reached"))
        self.assertFalse(self.spec.run_root.exists())
        self.released()

    def test_copy_exception_partial_not_opened_or_resumed(self):
        original, calls = sc.Protection.copy, []
        def fail(p, source, target):
            calls.append(target)
            if len(calls) == 2:
                raise OSError("synthetic copy failure")
            original(p, source, target)
        with patch.object(sc.Protection, "copy", fail), self.assertRaises(OSError):
            sc.stable_copy(self.spec, validate=lambda _: self.fail("partial opened"))
        self.assertTrue(self.spec.run_root.exists())
        self.released()
        with self.assertRaises(sc.CopyRefused):
            sc.stable_copy(self.spec, validate=lambda _: self.fail("partial reused"))

    def test_final_recheck_original_baseline_detects_addition(self):
        def checkpoint(stage, _):
            if stage == "before_final":
                (self.spec.db / "new-sidecar").write_bytes(b"change")
        with self.assertRaisesRegex(sc.CopyRefused, "final source recheck"):
            sc.stable_copy(self.spec, checkpoint=checkpoint, validate=lambda _: self.fail("changed copy opened"))
        self.released()

    def test_same_size_mtime_and_key_sha_drift_rejected(self):
        original = sc.snapshot
        for change in ("mtime", "sha"):
            count = [0]
            spec = replace(self.spec, run_root=self.spec.temp_root / change)
            def drift(spec, protection):
                value = original(spec, protection)
                count[0] += 1
                if count[0] == 2:
                    if change == "mtime":
                        size, mtime = value["model"]["files"]["config.json"]
                        value["model"]["files"]["config.json"] = (size, mtime + 1)
                    else:
                        value["sha"]["journal"] = "c" * 64
                return value
            with self.subTest(change=change), patch.object(sc, "snapshot", drift), self.assertRaisesRegex(sc.CopyRefused, "source_after"):
                sc.stable_copy(spec, validate=lambda _: self.fail("drift copy opened"))
            self.released()

    def test_target_tamper_rejected_before_validation(self):
        def checkpoint(stage, _):
            if stage == "copied":
                (self.spec.run_root / "inputs" / "ready.json").write_bytes(b"tampered")
        with self.assertRaisesRegex(sc.CopyRefused, "target key SHA"):
            sc.stable_copy(self.spec, checkpoint=checkpoint, validate=lambda _: self.fail("tampered copy opened"))
        self.released()

    def test_capacity_refusal_and_no_code_tree_scan(self):
        original, scanned = sc.os.scandir, []
        def scan(p):
            scanned.append(Path(p))
            return original(p)
        with patch.object(sc.os, "scandir", scan), patch.object(sc.shutil, "disk_usage", return_value=type("Space", (), {"free": 0})()), self.assertRaisesRegex(sc.CopyRefused, "free space"):
            sc.stable_copy(self.spec, validate=lambda _: self.fail("validation reached"))
        self.assertFalse(self.spec.run_root.exists())
        self.assertNotIn(self.spec.code_leaves[0].parent, scanned)
        self.released()


if __name__ == "__main__":
    unittest.main()
