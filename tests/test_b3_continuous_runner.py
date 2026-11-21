"""Fault-injection tests for the continuous B3 runner.

These tests drive the real control flow (locking, stop handling, reconciliation,
per-source commit, completion verification) with a character-level tokenizer, a
fake encoder and a real Chroma database inside a temporary directory.  No model
weights, no network, and no production path is touched.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "index"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b3_fakes  # noqa: E402
import index_common  # noqa: E402
import rebuild_b3  # noqa: E402
from build_index import EMBEDDING_TEXT_VERSION  # noqa: E402
from file_lock import FileLock  # noqa: E402


def source_text(paragraphs, marker="p"):
    """Markdown whose chunk count equals the number of paragraphs."""
    body = "\n\n".join(
        b3_fakes.paragraph(f"{marker}{index}", 600) for index in range(paragraphs)
    )
    return f"# Results\n\n{body}\n"


class B3RunnerTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.paths = b3_fakes.make_paths(self.root)
        self.addCleanup(self._cleanup)

    def _cleanup(self):
        b3_fakes.release_chroma(self.paths.staging_db)
        try:
            self._tmp.cleanup()
        except PermissionError:  # pragma: no cover - Windows handle latency
            pass

    def write(self, files):
        b3_fakes.write_corpus(self.paths.archive, files)

    def plan(self):
        return b3_fakes.plan_sources(self.paths)

    def backends(self, **kwargs):
        backends = b3_fakes.FakeBackends(self.paths, **kwargs)
        self.addCleanup(backends.close)
        return backends

    def manifest(self):
        return index_common.read_json(self.paths.manifest, default={"files": {}})

    def progress(self):
        return index_common.read_json(self.paths.progress, default={})

    # -- stop semantics ---------------------------------------------------

    def test_existing_stop_request_blocks_all_work(self):
        self.write({"alpha.md": source_text(3)})
        self.plan()
        rebuild_b3.request_stop(self.paths)

        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends)

        self.assertEqual(backends.encoder.calls, [])
        self.assertFalse(backends.collection_opened,
                         "no Chroma client may be opened when a stop is pending")
        self.assertFalse(self.paths.staging_db.exists())
        self.assertEqual(self.progress()["status"], "stopped")

    def test_clear_stop_does_not_swallow_a_newer_request(self):
        self.write({"alpha.md": source_text(3)})
        self.plan()
        rebuild_b3.request_stop(self.paths)
        old_token = rebuild_b3.stop_token(self.paths)
        rebuild_b3.request_stop(self.paths)  # a newer request arrives
        self.assertNotEqual(old_token, rebuild_b3.stop_token(self.paths))

        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends, clear_stop=True,
                              pre_start_stop_token=old_token)

        self.assertEqual(backends.encoder.calls, [])
        self.assertTrue(self.paths.stop_flag.exists(), "the newer stop must survive")
        self.assertEqual(self.progress()["status"], "stopped")

    def test_clear_stop_consumes_the_request_pending_at_startup(self):
        self.write({"alpha.md": source_text(2)})
        self.plan()
        rebuild_b3.request_stop(self.paths)
        token = rebuild_b3.stop_token(self.paths)

        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends, clear_stop=True,
                              pre_start_stop_token=token)

        self.assertFalse(self.paths.stop_flag.exists())
        self.assertEqual(backends.encoder.unit_sizes, [2])
        self.assertEqual(self.progress()["status"], "completed")

    def test_stop_during_encoding_stops_at_the_unit_boundary(self):
        self.write({"alpha.md": source_text(6)})
        self.plan()
        backends = self.backends()

        def stop_on_second_unit(call_index, _texts):
            if call_index == 2:
                rebuild_b3.request_stop(self.paths)

        backends.encoder.on_encode = stop_on_second_unit
        rebuild_b3.run_resume(self.paths, backends=backends, unit_chunks=2)

        self.assertEqual(backends.encoder.unit_sizes, [2, 2],
                         "no third computation unit may start after the request")
        self.assertEqual(self.manifest()["files"], {},
                         "a half-written source must not enter the completion ledger")
        self.assertEqual(backends.collection.count(), 0,
                         "unfinished rows are discarded at the stop boundary")
        self.assertEqual(self.progress()["status"], "stopped")
        self.assertIsNone(self.progress()["in_progress"])

    def test_long_source_stops_without_waiting_for_the_whole_paper(self):
        self.write({"alpha.md": source_text(40)})
        self.plan()
        backends = self.backends()

        def stop_quickly(call_index, _texts):
            if call_index == 3:
                rebuild_b3.request_stop(self.paths)

        backends.encoder.on_encode = stop_quickly
        rebuild_b3.run_resume(self.paths, backends=backends, unit_chunks=4)

        self.assertEqual(backends.encoder.unit_sizes, [4, 4, 4])
        self.assertLess(sum(backends.encoder.unit_sizes), 40,
                        "the whole paper must not be encoded before stopping")
        self.assertEqual(self.manifest()["files"], {})

    # -- crash / resume ---------------------------------------------------

    def test_crash_after_upsert_redoes_only_the_uncommitted_source(self):
        self.write({
            "alpha.md": source_text(3, marker="alpha"),
            "beta.md": source_text(3, marker="beta"),
        })
        self.plan()

        backends = self.backends()
        backends.wrap_collection(b3_fakes.CrashingCollection(
            backends.collection, fail_on_upsert=3
        ))
        with self.assertRaises(RuntimeError):
            rebuild_b3.run_resume(self.paths, backends=backends, unit_chunks=2)

        committed = self.manifest()["files"]
        self.assertEqual(list(committed), ["alpha.md"])
        raw = b3_fakes.FakeBackends(self.paths)
        self.assertEqual(raw.collection.count(), 5, "alpha rows plus beta's partial rows")
        raw.close()

        resumed = self.backends()
        rebuild_b3.run_resume(self.paths, backends=resumed, unit_chunks=2)

        self.assertEqual(set(self.manifest()["files"]), {"alpha.md", "beta.md"})
        self.assertEqual(self.manifest()["total_chunks"], 6)
        self.assertEqual(resumed.collection.count(), 6)
        self.assertFalse(
            any("alpha0" in text for text in resumed.encoder.texts_seen),
            "a committed source must not be encoded again",
        )
        ids = resumed.collection.get(where={"source": "beta.md"})["ids"]
        self.assertEqual(sorted(ids),
                         sorted(rebuild_b3.expected_ids(
                             next(item for item in rebuild_b3.load_sources(self.paths)["sources"]
                                  if item["source"] == "beta.md"), 3)))

    def test_rows_without_a_committed_entry_are_removed_on_resume(self):
        self.write({"alpha.md": source_text(2)})
        self.plan()
        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends)
        self.assertEqual(backends.collection.count(), 2)

        backends.collection.upsert(
            ids=["orphan-001"], documents=["orphan"],
            metadatas=[{"source": "alpha.md", "embedding_text_version": EMBEDDING_TEXT_VERSION}],
            embeddings=[[1.0, 1.0]],
        )
        index_common.write_json_atomic(
            self.paths.progress,
            {"status": "running", "in_progress": {"source": "alpha.md"}},
        )

        resumed = self.backends()
        rebuild_b3.run_resume(self.paths, backends=resumed)
        self.assertEqual(resumed.collection.count(), 2)
        self.assertEqual(resumed.encoder.calls, [],
                         "an already committed source is not re-encoded")

    # -- representation and content changes --------------------------------

    def test_corpus_change_is_refused_instead_of_reusing_stale_ids(self):
        self.write({"alpha.md": source_text(2)})
        self.plan()
        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends)
        first_manifest = self.manifest()

        self.write({"alpha.md": source_text(2, marker="changed")})
        with self.assertRaises(RuntimeError) as caught:
            rebuild_b3.run_resume(self.paths, backends=self.backends())
        self.assertIn("corpus changed", str(caught.exception))
        self.assertEqual(self.manifest(), first_manifest)

    def test_replanned_source_is_reencoded_with_new_ids(self):
        self.write({"alpha.md": source_text(2)})
        self.plan()
        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends)
        old_ids = sorted(backends.collection.get()["ids"])

        self.write({"alpha.md": source_text(2, marker="changed")})
        self.plan()
        resumed = self.backends()
        rebuild_b3.run_resume(self.paths, backends=resumed)

        new_ids = sorted(resumed.collection.get()["ids"])
        self.assertNotEqual(old_ids, new_ids)
        self.assertEqual(len(new_ids), 2)
        source = next(item for item in rebuild_b3.load_sources(self.paths)["sources"]
                      if item["source"] == "alpha.md")
        self.assertEqual(self.manifest()["files"]["alpha.md"]["sha256"], source["sha256"])

    def test_title_change_invalidates_an_otherwise_identical_source(self):
        self.write({"alpha.md": source_text(2)})
        self.plan()
        rebuild_b3.run_resume(self.paths, backends=self.backends())
        self.assertEqual(self.manifest()["files"]["alpha.md"]["chunks"], 2)

        with b3_fakes.patch_meta({"alpha": {"title": "Alpha renamed"}}):
            resumed = self.backends()
            rebuild_b3.run_resume(self.paths, backends=resumed)
        self.assertEqual(resumed.encoder.unit_sizes, [2],
                         "the representation input changed, so the source is redone")

    def test_representation_change_refuses_to_reuse_completed_work(self):
        self.write({"alpha.md": source_text(2)})
        payload = self.plan()
        rebuild_b3.run_resume(self.paths, backends=self.backends())

        payload["embedding_text_version"] = "passage-v3"
        index_common.write_json_atomic(self.paths.sources, payload)
        with self.assertRaises(RuntimeError) as caught:
            rebuild_b3.run_resume(self.paths, backends=self.backends())
        self.assertIn("different configuration", str(caught.exception))

    # -- locking ----------------------------------------------------------

    def test_second_writer_is_rejected_and_the_lock_is_reusable(self):
        self.write({"alpha.md": source_text(2)})
        self.plan()
        holder = FileLock(self.paths.lock).acquire()
        try:
            with self.assertRaises(RuntimeError) as caught:
                rebuild_b3.run_resume(self.paths, backends=self.backends())
            self.assertIn("another B3 writer", str(caught.exception))
        finally:
            holder.release()

        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends)
        self.assertEqual(backends.collection.count(), 2)

    # -- duplicates, zero chunks, completion -------------------------------

    def test_duplicate_alias_and_zero_chunk_source(self):
        duplicate = source_text(2)
        self.write({
            "PMID_1.md": duplicate,
            "legacy copy.md": duplicate,
            "tiny.md": "# T\n\nshort\n",
        })
        self.plan()
        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends)

        files = self.manifest()["files"]
        self.assertEqual(files["PMID_1.md"]["chunks"], 2)
        self.assertEqual(files["legacy copy.md"]["chunks"], 0)
        self.assertEqual(files["legacy copy.md"]["dup_of"], "PMID_1.md")
        self.assertEqual(files["tiny.md"]["chunks"], 0)
        self.assertEqual(backends.collection.count(), 2)

        again = self.backends()
        rebuild_b3.run_resume(self.paths, backends=again)
        self.assertEqual(again.encoder.calls, [], "zero-chunk sources must not loop")
        self.assertEqual(again.collection.count(), 2)

    def test_completed_build_is_verified_and_resume_adds_nothing(self):
        self.write({
            "alpha.md": source_text(2),
            "beta.md": "# Methods\n\n" + b3_fakes.paragraph("beta0", 900) + "\n",
        })
        self.plan()
        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends)

        manifest = self.manifest()
        self.assertEqual(manifest["status"], "complete")
        self.assertEqual(manifest["total_chunks"], backends.collection.count())
        self.assertTrue(
            index_common.read_json(
                self.paths.manifest.with_name(self.paths.manifest.name + ".ready.json")
            )["status"] == "ready"
        )
        self.assertEqual(self.progress()["status"], "completed")

        again = self.backends()
        rebuild_b3.run_resume(self.paths, backends=again)
        self.assertEqual(again.encoder.calls, [])
        self.assertEqual(again.collection.count(), manifest["total_chunks"])
        self.assertTrue(self.paths.staging_db.is_dir(),
                        "the runner must not move/publish the staging directory")

    def test_completion_requires_every_source(self):
        self.write({
            "alpha.md": source_text(2, marker="alpha"),
            "beta.md": source_text(2, marker="beta"),
        })
        self.plan()
        backends = self.backends()
        rebuild_b3.run_resume(self.paths, backends=backends, max_sources=1)

        self.assertEqual(self.manifest()["status"], "partial")
        self.assertFalse(
            self.paths.manifest.with_name(self.paths.manifest.name + ".ready.json").exists()
        )
        with self.assertRaises(RuntimeError):
            rebuild_b3.verify_only(self.paths, backends=backends)

        rebuild_b3.run_resume(self.paths, backends=backends)
        self.assertEqual(self.manifest()["status"], "complete")
        self.assertEqual(
            rebuild_b3.verify_only(self.paths, backends=backends)["chunks"], 4
        )

    # -- light commands ---------------------------------------------------

    def test_status_and_stop_use_no_model_and_no_chroma(self):
        self.write({"alpha.md": source_text(2)})
        self.plan()
        with mock.patch.object(rebuild_b3, "RealBackends",
                               side_effect=AssertionError("must not open Chroma")), \
                mock.patch.object(rebuild_b3, "load_tokenizer",
                                  side_effect=AssertionError("must not load a tokenizer")):
            self.assertEqual(rebuild_b3.main(["--status"], paths=self.paths), 0)
            self.assertEqual(rebuild_b3.main(["--stop"], paths=self.paths), 0)
        self.assertTrue(self.paths.stop_flag.exists())

        status = rebuild_b3.status_payload(self.paths)
        self.assertTrue(status["stop_requested"])
        self.assertEqual(status["remaining_canonical_sources"], 1)
        self.assertEqual(status["next_source"], "alpha.md")


if __name__ == "__main__":
    unittest.main()
