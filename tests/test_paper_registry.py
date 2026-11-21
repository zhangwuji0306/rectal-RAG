"""Synthetic tests for the C1 paper registry (immutable paper identity).

Everything here is built in a temporary directory: no production corpus, no
index, no model.  The registry's promise - a stable internal ``paper_id`` that
is assigned once and never derived from a DOI/PMID again - is what these tests
exercise, together with the checks that keep it honest: identifier conflicts
stay pending, identical content does not merge identities by itself, legacy ids
either resolve or report ambiguity, and a failed registration writes nothing.
"""

from __future__ import annotations

import csv
import json
import shutil
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT / "index") not in sys.path:
    sys.path.insert(0, str(ROOT / "index"))

import paper_registry as pr  # noqa: E402

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64


def record(source, sha=SHA_A, identifiers=None, metadata=None, **extra):
    payload = {"source": source, "content_sha256": sha}
    if identifiers is not None:
        payload["identifiers"] = identifiers
    if metadata is not None:
        payload["metadata"] = metadata
    payload.update(extra)
    return payload


class RegistryTestCase(unittest.TestCase):
    def setUp(self):
        self._dir = Path(tempfile.mkdtemp(prefix="registry-test-"))
        self.addCleanup(shutil.rmtree, self._dir, ignore_errors=True)
        self.db = self._dir / "registry.sqlite3"
        self.registry = pr.PaperRegistry(self.db, create=True)
        self.addCleanup(self.registry.close)

    def kinds(self):
        return sorted({item["kind"] for item in self.registry.conflicts()})


class IdentityRuleTests(RegistryTestCase):
    def test_minimal_record_keeps_missing_fields_explicit(self):
        result = self.registry.register(
            record(
                "PMID_100.md",
                identifiers={"pmid": "PMID 100"},
                metadata={"title": "A study", "year": "2020"},
            )
        )

        self.assertEqual(result["status"], "created")
        self.assertRegex(result["paper_id"], r"^PAP\d{6}$")
        paper = self.registry.paper(result["paper_id"])
        self.assertEqual(paper["pmid"], "100")
        self.assertIn(paper["doi"], (None, ""))
        self.assertEqual(paper["license"], "")
        self.assertEqual(paper["journal"], "")
        self.assertEqual(paper["title"], "A study")
        self.assertEqual(paper["content_sha256"], SHA_A)
        self.assertTrue(paper["metadata_updated_at"])
        provenance = json.loads(self.registry._paper_row(result["paper_id"])["metadata_provenance"])
        self.assertEqual(provenance["title"], "PMID_100.md")

    def test_later_doi_only_adds_an_alias(self):
        first = self.registry.register(
            record("PMID_200.md", identifiers={"pmid": "200"}, metadata={"title": "T"})
        )
        second = self.registry.register(
            record(
                "PMID_200.md",
                identifiers={"pmid": "200", "doi": "https://doi.org/10.1/ABC"},
            )
        )

        self.assertEqual(first["paper_id"], second["paper_id"])
        self.assertEqual(second["status"], "existing")
        self.assertEqual(self.registry.stats()["papers"], 1)
        self.assertEqual(self.registry.resolve("doi", "10.1/abc"), first["paper_id"])
        schemes = {item["scheme"] for item in self.registry.identifiers(first["paper_id"])}
        self.assertEqual(schemes, {"pmid", "doi"})

    def test_replayed_import_is_idempotent(self):
        payload = record(
            "PMID_300.md", identifiers={"pmid": "300"}, metadata={"title": "T"}
        )
        statuses = [self.registry.register(payload)["status"] for _ in range(3)]

        self.assertEqual(statuses, ["created", "existing", "existing"])
        self.assertEqual(self.registry.stats()["artifacts"], 1)
        self.assertEqual(self.registry.stats()["conflicts"], 0)

    def test_same_title_different_paper_stays_separate(self):
        one = self.registry.register(
            record("a.md", identifiers={"doi": "10.1/one"}, metadata={"title": "Same  Title"})
        )
        two = self.registry.register(
            record("b.md", sha=SHA_B, identifiers={"doi": "10.1/two"},
                   metadata={"title": "same title"})
        )

        self.assertNotEqual(one["paper_id"], two["paper_id"])
        self.assertEqual(self.registry.stats()["papers"], 2)
        self.assertIn("title_candidate_collision", self.kinds())
        # A normalized-title hit is a candidate only: no relation, no merge.
        self.assertEqual(self.registry.relations(), [])

    def test_conflicting_identifiers_add_nothing_and_stay_pending(self):
        alpha = self.registry.register(
            record("alpha.md", identifiers={"doi": "10.1/alpha"}, metadata={"title": "A"})
        )
        beta = self.registry.register(
            record("beta.md", sha=SHA_B, identifiers={"doi": "10.1/beta", "pmid": "500"},
                   metadata={"title": "B"})
        )
        clashing = self.registry.register(
            record("clash.md", sha=SHA_C,
                   identifiers={"doi": "10.1/alpha", "pmid": "500"},
                   metadata={"title": "C"})
        )

        self.assertEqual(clashing["status"], "conflict")
        self.assertIn("identifier_conflict", self.kinds())
        self.assertEqual(self.registry.resolve("doi", "10.1/alpha"), alpha["paper_id"])
        self.assertEqual(self.registry.resolve("pmid", "500"), beta["paper_id"])
        self.assertIsNone(self.registry.resolve("doi", "10.1/beta2"))
        self.assertEqual(self.registry.paper(clashing["paper_id"])["identity_status"], "conflict")
        self.assertEqual(self.registry.stats()["papers"], 3)


class ArtifactRoleTests(RegistryTestCase):
    def test_new_content_on_a_known_paper_is_a_version(self):
        first = self.registry.register(
            record("PMID_400.md", identifiers={"pmid": "400"}, metadata={"title": "T"})
        )
        second = self.registry.register(
            record("PMID_400b.md", sha=SHA_B, identifiers={"pmid": "400"})
        )

        self.assertEqual(second["paper_id"], first["paper_id"])
        self.assertEqual(second["role"], "version")
        roles = {item["role"] for item in self.registry.artifacts(first["paper_id"])}
        self.assertEqual(roles, {"main", "version"})

    def test_supplement_is_an_artifact_of_the_main_paper(self):
        main = self.registry.register(
            record("PMID_500.md", identifiers={"pmid": "500", "doi": "10.1/five"},
                   metadata={"title": "Main"})
        )
        supplement = self.registry.register(
            record("PMID_500_supp.md", sha=SHA_C, identifiers={"doi": "10.1/five"},
                   supplement_of={"kind": "doi", "value": "10.1/five"})
        )

        self.assertEqual(supplement["paper_id"], main["paper_id"])
        self.assertEqual(supplement["role"], "supplement")
        self.assertEqual(self.registry.stats()["papers"], 1)
        roles = {item["role"] for item in self.registry.artifacts(main["paper_id"])}
        self.assertEqual(roles, {"main", "supplement"})
        supplement_rows = [
            item for item in self.registry.artifacts(main["paper_id"])
            if item["role"] == "supplement"
        ]
        self.assertEqual(supplement_rows[0]["is_supplement"], 1)

    def test_correction_and_retraction_relations(self):
        original = self.registry.register(
            record("orig.md", identifiers={"doi": "10.1/orig"}, metadata={"title": "O"})
        )
        correction = self.registry.register(
            record("erratum.md", sha=SHA_B, identifiers={"doi": "10.1/erratum"},
                   relations=[
                       {"kind": "correction", "target": {"kind": "doi", "value": "10.1/orig"},
                        "evidence": "erratum notice"}
                   ])
        )
        retraction = self.registry.register(
            record("retraction.md", sha=SHA_C, identifiers={"doi": "10.1/retracted"},
                   relations=[
                       {"kind": "retraction", "target": {"kind": "doi", "value": "10.1/orig"}}
                   ])
        )

        self.assertNotEqual(correction["paper_id"], original["paper_id"])
        self.assertNotEqual(retraction["paper_id"], original["paper_id"])
        by_kind = {
            item["kind"]: item for item in self.registry.relations(correction["paper_id"])
        }
        self.assertEqual(by_kind["correction"]["status"], "confirmed")
        by_kind = {
            item["kind"]: item for item in self.registry.relations(retraction["paper_id"])
        }
        self.assertEqual(by_kind["retraction"]["status"], "pending")

    def test_identical_content_does_not_merge_different_identifiers(self):
        canonical = self.registry.register(
            record("PMID_600.md", identifiers={"pmid": "600"}, metadata={"title": "C"})
        )
        duplicate = self.registry.register(
            record("PMID_601.md", identifiers={"pmid": "601"},
                   duplicate_of={"kind": "source", "value": "PMID_600.md"})
        )

        self.assertNotEqual(canonical["paper_id"], duplicate["paper_id"])
        self.assertEqual(duplicate["role"], "duplicate")
        self.assertEqual(self.registry.resolve("pmid", "600"), canonical["paper_id"])
        self.assertEqual(self.registry.resolve("pmid", "601"), duplicate["paper_id"])
        self.assertIn("content_duplicate_identifier_mismatch", self.kinds())
        relation = self.registry.relations()[0]
        self.assertEqual(relation["kind"], "exact_duplicate")
        self.assertEqual(relation["status"], "pending")

    def test_identical_content_without_identifier_disagreement_shares_the_paper(self):
        canonical = self.registry.register(
            record("PMID_700.md", identifiers={"pmid": "700", "doi": "10.1/seven"},
                   metadata={"title": "S"})
        )
        duplicate = self.registry.register(
            record("copy-of-seven.md", identifiers={"doi": "10.1/seven"},
                   duplicate_of={"kind": "source", "value": "PMID_700.md"})
        )

        self.assertEqual(duplicate["paper_id"], canonical["paper_id"])
        self.assertEqual(duplicate["role"], "duplicate")
        self.assertEqual(self.registry.stats()["papers"], 1)
        self.assertEqual(self.registry.relations(), [])


class TransactionTests(RegistryTestCase):
    def test_failed_registration_leaves_no_partial_write(self):
        self.registry.register(
            record("good.md", identifiers={"doi": "10.1/good"}, metadata={"title": "G"})
        )
        before = self.registry.stats()

        with self.assertRaises(pr.RegistryError) as caught:
            self.registry.register(
                record("broken.md", sha=SHA_D, identifiers={"doi": "10.1/broken"},
                       relations=[
                           {"kind": "correction",
                            "target": {"kind": "doi", "value": "10.1/never-registered"}}
                       ])
            )
        self.assertIn("unknown reference identifier", str(caught.exception))

        self.assertEqual(self.registry.stats(), before)
        self.assertIsNone(self.registry.artifact_by_source("broken.md"))
        self.assertIsNone(self.registry.resolve("doi", "10.1/broken"))
        self.assertEqual(self.registry.relations(), [])

    def test_invalid_record_is_rejected_before_writing(self):
        with self.assertRaises(pr.RegistryError):
            self.registry.register(record("bad.md", sha="not-a-sha"))
        with self.assertRaises(pr.RegistryError):
            self.registry.register(
                record("bad2.md", identifiers={"arxiv": "2101.00001"})
            )
        self.assertEqual(self.registry.stats()["papers"], 0)
        self.assertEqual(self.registry.stats()["artifacts"], 0)

    def test_corrected_identifier_keeps_the_old_alias_for_audit(self):
        created = self.registry.register(
            record("PMID_800.md", identifiers={"doi": "10.1/TYPO"}, metadata={"title": "T"})
        )

        result = self.registry.correct_identifier("doi", "10.1/typo", "10.1/typo-fixed",
                                                  evidence={"reason": "publisher fix"})

        self.assertEqual(result["status"], "corrected")
        self.assertIsNone(self.registry.resolve("doi", "10.1/typo"))
        self.assertEqual(self.registry.resolve("doi", "10.1/typo-fixed"), created["paper_id"])
        rows = {item["value"]: item for item in self.registry.identifiers(created["paper_id"])}
        self.assertEqual(rows["10.1/typo"]["status"], "superseded")
        self.assertEqual(rows["10.1/typo"]["superseded_by"], "10.1/typo-fixed")
        self.assertEqual(rows["10.1/typo-fixed"]["status"], "active")


class LegacyMappingTests(RegistryTestCase):
    def test_legacy_alias_kinds_resolve(self):
        paper = self.registry.register(
            record("PMID_900.md", identifiers={"pmid": "900", "doi": "10.1/nine"},
                   legacy={"b3_paper_id": "doi:10.1/nine", "source_stem": "PMID_900"})
        )

        self.assertEqual(self.registry.resolve_alias("doi:10.1/nine"), paper["paper_id"])
        self.assertEqual(self.registry.resolve_alias("source:PMID_900.md"), paper["paper_id"])
        self.assertEqual(self.registry.resolve_legacy("PMID_900", "source_stem")["status"],
                         "resolved")
        self.assertEqual(self.registry.resolve_legacy("doi:10.1/nine", "b3_paper_id")["paper_id"],
                         paper["paper_id"])

    def test_ambiguous_legacy_id_is_reported_not_guessed(self):
        one = self.registry.register(record("one.md", identifiers={"doi": "10.1/one"}))
        two = self.registry.register(record("two.md", sha=SHA_B, identifiers={"doi": "10.1/two"}))
        self.registry.set_legacy("b3_paper_id", "doi:10.1/shared", one["paper_id"], "one.md")
        conflict = self.registry.set_legacy(
            "b3_paper_id", "doi:10.1/shared", two["paper_id"], "two.md"
        )

        self.assertEqual(conflict["status"], "ambiguous")
        resolved = self.registry.resolve_legacy("doi:10.1/shared", "b3_paper_id")
        self.assertEqual(resolved["status"], "ambiguous")
        self.assertIsNone(resolved["paper_id"])
        self.assertEqual(resolved["candidates"], sorted([one["paper_id"], two["paper_id"]]))
        self.assertIsNone(self.registry.resolve_alias("doi:10.1/shared"))
        self.assertIn("legacy_id_ambiguous", self.kinds())

    def test_unknown_legacy_id_is_unresolved(self):
        result = self.registry.resolve_legacy("source:nowhere.md")

        self.assertEqual(result["status"], "unresolved")
        self.assertIsNone(result["paper_id"])

    def test_explicit_merge_moves_identity_and_clears_ambiguity(self):
        canonical = self.registry.register(
            record("PMID_950.md", identifiers={"doi": "10.1/merge"},
                   legacy={"b3_paper_id": "doi:10.1/merge"})
        )
        duplicate = self.registry.register(
            record("other-file.md", sha=SHA_B, identifiers={"pmid": "951"},
                   legacy={"b3_paper_id": "source:other-file"})
        )
        self.registry.set_legacy("source_stem", "dup-source", duplicate["paper_id"], "other-file.md")
        merged = self.registry.merge_papers(duplicate["paper_id"], canonical["paper_id"],
                                            evidence={"review": "same study"})

        self.assertEqual(merged["status"], "merged")
        self.assertEqual(self.registry.resolve("pmid", "951"), canonical["paper_id"])
        self.assertEqual(self.registry.resolve_alias("source:other-file.md"), canonical["paper_id"])
        moved = self.registry.paper(duplicate["paper_id"])
        self.assertEqual(moved["identity_status"], "merged")
        self.assertEqual(moved["duplicate_of"], canonical["paper_id"])
        self.assertEqual(
            self.registry.resolve_legacy("dup-source", "source_stem")["paper_id"],
            canonical["paper_id"],
        )


class CorpusAdapterTests(RegistryTestCase):
    """Exercise the saved-input adapter without reading any real corpus file."""

    def write_inputs(self):
        base = self._dir / "inputs"
        base.mkdir()
        cards = [
            {
                "source": "PMID_1000.md",
                "paper_id": "doi:10.1/one",
                "doi": "10.1/one",
                "title": "One",
                "authors": "A, B",
                "year": "2020",
                "journal": "J",
                "pub_type": "Journal Article",
                "license": "unverified",
                "mac": "yes",
            },
            {
                "source": "descriptive name.md",
                "paper_id": "doi:10.1/one",
                "doi": "10.1/one",
                "title": "One (fuller title)",
                "authors": "A, B; C, D",
                "year": "2020",
                "journal": "Journal Full",
                "pub_type": "",
                "license": "unverified",
                "mac": "yes",
            },
            {
                "source": "PMID_1001.md",
                "paper_id": "source:PMID_1001",
                "doi": "",
                "title": "No identifiers",
                "authors": "",
                "year": "",
                "journal": "",
                "pub_type": "",
                "license": "unverified",
                "mac": "no",
            },
        ]
        (base / "cards.json").write_text(json.dumps(cards), encoding="utf-8")
        inventory = [
            ["PMID_1000.md", SHA_A],
            ["descriptive name.md", SHA_B],
            ["PMID_1001.md", SHA_C],
            ["PMID_1002.md", SHA_D],
            ["supplement.md", SHA_E],
        ]
        (base / "inventory.json").write_text(json.dumps(inventory), encoding="utf-8")
        manifest = {
            "files": {
                "PMID_1000.md": {"sha256": SHA_A, "chunks": 10, "mac": "yes"},
                "descriptive name.md": {"sha256": SHA_B, "chunks": 8, "mac": "yes"},
                "PMID_1001.md": {"sha256": SHA_C, "chunks": 4, "mac": "no"},
                "PMID_1002.md": {"sha256": SHA_D, "chunks": 0,
                                 "dup_of": "PMID_1000.md", "mac": "yes"},
                "supplement.md": {"sha256": SHA_E, "chunks": 2, "mac": "yes"},
            },
            "total_chunks": 24,
        }
        (base / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        header = ["PMID", "Title", "Abstract", "Authors", "FirstAuthor", "Journal", "ISSN",
                  "Year", "Volume", "Issue", "Pages", "DOI", "PMC", "MeSH", "PubType",
                  "Language", "Status", "PDFPath", "Note", "License"]
        rows = [
            ["1000", "One", "", "A, B", "A", "J", "", "2020", "", "", "",
             "10.1/one", "PMC999", "", "Journal Article", "eng", "done", "", "", ""],
            ["1001", "No identifiers", "", "", "", "", "", "", "", "", "", "",
             "PMC998", "", "", "eng", "done", "", "", ""],
        ]
        with (base / "canonical.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        mapping = {
            "files": [
                {"filename": "supplement.pdf", "folder": "FOLDER1", "doi": "10.1/one"},
            ]
        }
        (base / "mapping.json").write_text(json.dumps(mapping), encoding="utf-8")
        supplement_map = {"supplement": {"kind": "folder", "value": "FOLDER1"}}
        return base, supplement_map

    def test_adapter_derives_roles_identifiers_and_legacy_ids(self):
        base, supplement_map = self.write_inputs()
        records = pr.build_corpus_records(
            base / "cards.json",
            base / "manifest.json",
            base / "inventory.json",
            base / "canonical.csv",
            mapping_path=base / "mapping.json",
            supplement_map=supplement_map,
        )

        self.assertEqual(len(records), 5)
        by_source = {item["source"]: item for item in records}
        self.assertEqual(by_source["PMID_1000.md"]["identifiers"],
                         {"pmid": "1000", "doi": "10.1/one", "pmcid": "PMC999"})
        self.assertEqual(by_source["PMID_1000.md"]["legacy"]["b3_paper_id"], "doi:10.1/one")
        self.assertEqual(by_source["descriptive name.md"]["legacy"]["source_stem"],
                         "descriptive name")
        self.assertEqual(by_source["PMID_1002.md"]["duplicate_of"],
                         {"kind": "source", "value": "PMID_1000.md"})
        self.assertEqual(by_source["supplement.md"]["supplement_of"],
                         {"kind": "doi", "value": "10.1/one"})

    def test_adapter_records_register_into_expected_papers(self):
        base, supplement_map = self.write_inputs()
        records = pr.build_corpus_records(
            base / "cards.json",
            base / "manifest.json",
            base / "inventory.json",
            base / "canonical.csv",
            mapping_path=base / "mapping.json",
            supplement_map=supplement_map,
        )
        for item in sorted(records, key=lambda rec: bool(
                rec.get("supplement_of") or rec.get("duplicate_of"))):
            self.registry.register(item)

        stats = self.registry.stats()
        self.assertEqual(stats["artifacts"], 5)
        self.assertEqual(stats["papers"], 3)
        self.assertEqual(stats["artifacts_by_role"],
                         {"duplicate": 1, "main": 2, "supplement": 1, "version": 1})


class VerifyTests(RegistryTestCase):
    def seed(self):
        self.registry.register(
            record("PMID_1100.md", identifiers={"pmid": "1100", "doi": "10.1/eleven"},
                   metadata={"title": "Eleven"},
                   legacy={"b3_paper_id": "doi:10.1/eleven", "source_stem": "PMID_1100"})
        )
        self.registry.register(
            record("PMID_1101.md", sha=SHA_B, identifiers={"pmid": "1101"},
                   duplicate_of={"kind": "source", "value": "PMID_1100.md"},
                   legacy={"b3_paper_id": "source:PMID_1101", "source_stem": "PMID_1101"})
        )

    def test_verify_passes_when_evidence_agrees(self):
        self.seed()
        result = self.registry.verify(
            inventory=[("PMID_1100.md", SHA_A), ("PMID_1101.md", SHA_B)]
        )
        fails = [item for item in result["checks"] if item["status"] != "pass"]
        self.assertEqual(fails, [])
        # The pending duplicate relation and its conflict stay visible, never closed.
        self.assertEqual(len(self.registry.conflicts("pending")), 1)

    def test_verify_fails_on_missing_source_and_silent_sha_mismatch(self):
        self.seed()
        result = self.registry.verify(
            inventory=[("PMID_1100.md", SHA_B), ("PMID_9999.md", SHA_C)]
        )
        checks = {item["name"]: item for item in result["checks"]}
        self.assertEqual(checks["inventory_coverage"]["status"], "fail")
        self.assertIn("PMID_9999.md", checks["inventory_coverage"]["detail"]["missing"])
        self.assertTrue(checks["inventory_coverage"]["detail"]["sha_mismatch"])

    def test_verify_reports_unresolved_qrels_identity(self):
        self.seed()
        qrels = self._dir / "qrels.jsonl"
        rows = [
            {"query_id": "q1", "query": "q1",
             "judgements": [{"doi": "10.1/unregistered", "source_file": "missing.md",
                             "relevance": 2}]},
            {"query_id": "q2", "query": "q2",
             "judgements": [{"doi": "10.1/eleven", "source_file": "PMID_1100.md",
                             "relevance": 3}]},
        ]
        qrels.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

        result = self.registry.verify(
            inventory=[("PMID_1100.md", SHA_A), ("PMID_1101.md", SHA_B)],
            qrels=qrels,
        )
        checks = {item["name"]: item for item in result["checks"]}
        self.assertEqual(checks["qrels_identity"]["status"], "fail")
        self.assertEqual(checks["qrels_identity"]["detail"]["total_failing"], 1)
        self.assertEqual(checks["qrels_identity"]["detail"]["judgements"], 2)


class IdentityConsistencyTests(RegistryTestCase):
    """Every supplied identifier is compared before an owner is chosen."""

    def test_second_pmid_on_the_same_doi_is_held_as_a_conflict(self):
        first = self.registry.register(
            record("PMID_2000.md", identifiers={"doi": "10.1/shared", "pmid": "2000"})
        )
        second = self.registry.register(
            record(
                "PMID_2001.md",
                sha=SHA_B,
                identifiers={"doi": "10.1/shared", "pmid": "2001"},
            )
        )

        self.assertEqual(first["status"], "created")
        self.assertEqual(second["status"], "conflict")
        self.assertEqual(self.registry.stats()["papers"], 2)
        # The unconfirmed PMID is not adopted as an alias, and the DOI keeps
        # the paper that owned it first.
        self.assertIsNone(self.registry.resolve("pmid", "2001"))
        self.assertEqual(self.registry.resolve("doi", "10.1/shared"), first["paper_id"])
        self.assertIn("identifier_conflict", self.kinds())

    def test_supplement_target_must_agree_with_its_identifiers(self):
        main = self.registry.register(
            record("PMID_2100.md", identifiers={"doi": "10.1/a", "pmid": "2100"})
        )
        other = self.registry.register(
            record("PMID_2101.md", sha=SHA_B, identifiers={"doi": "10.1/b", "pmid": "2101"})
        )
        held = self.registry.register(
            record(
                "supplement.md",
                sha=SHA_C,
                identifiers={"doi": "10.1/b", "pmid": "2102"},
                supplement_of={"kind": "doi", "value": "10.1/a"},
            )
        )

        self.assertEqual(held["status"], "conflict")
        self.assertNotIn(held["paper_id"], (main["paper_id"], other["paper_id"]))
        artifacts = [item["source"] for item in self.registry.artifacts(main["paper_id"])]
        self.assertNotIn("supplement.md", artifacts)
        self.assertIsNone(self.registry.resolve("pmid", "2102"))

    def test_conflict_keeps_the_owner_metadata_and_aliases(self):
        self.registry.register(
            record(
                "PMID_2200.md",
                identifiers={"doi": "10.1/meta", "pmid": "2200"},
                metadata={"title": "Original", "authors": "A"},
            )
        )
        self.registry.register(
            record(
                "PMID_2201.md",
                sha=SHA_B,
                identifiers={"doi": "10.1/meta", "pmid": "2201"},
                metadata={"title": "Other title", "authors": "B"},
            )
        )

        owner = self.registry.resolve("doi", "10.1/meta")
        paper = self.registry.paper(owner)
        self.assertEqual(paper["title"], "Original")
        self.assertEqual(paper["authors"], "A")
        self.assertEqual(paper["pmid"], "2200")
        self.assertIsNone(self.registry.resolve("pmid", "2201"))


class LegacyConflictTests(RegistryTestCase):
    """A shared old id stays ambiguous until someone settles it explicitly."""

    def test_shared_legacy_id_on_first_import_stays_ambiguous(self):
        self.registry.register(
            record(
                "PMID_3000.md",
                identifiers={"pmid": "3000"},
                legacy={"source_stem": "shared"},
            )
        )
        self.registry.register(
            record(
                "PMID_3001.md",
                sha=SHA_B,
                identifiers={"pmid": "3001"},
                legacy={"source_stem": "shared"},
            )
        )

        report = self.registry.resolve_legacy("shared", "source_stem")
        self.assertEqual(report["status"], "ambiguous")
        self.assertEqual(len(report["candidates"]), 2)
        pending = [
            item
            for item in self.registry.conflicts("pending")
            if item["kind"] == "legacy_id_ambiguous"
        ]
        self.assertEqual(len(pending), 1)
        checks = {item["name"]: item for item in self.registry.verify()["checks"]}
        self.assertEqual(checks["legacy_resolution"]["status"], "fail")
        self.assertEqual(checks["legacy_conflict_consistency"]["status"], "pass")

    def test_repeated_registration_keeps_ambiguity_and_merges_candidates(self):
        papers = []
        for index, source in enumerate(("PMID_3100.md", "PMID_3101.md", "PMID_3102.md")):
            result = self.registry.register(
                record(
                    source,
                    sha="%s" % ("abc"[index] * 64),
                    identifiers={"pmid": str(3100 + index)},
                    legacy={"source_stem": "shared-2"},
                )
            )
            papers.append(result["paper_id"])

        report = self.registry.resolve_legacy("shared-2", "source_stem")
        self.assertEqual(report["status"], "ambiguous")
        self.assertEqual(sorted(report["candidates"]), sorted(papers))
        pending = [
            item
            for item in self.registry.conflicts("pending")
            if item["kind"] == "legacy_id_ambiguous"
        ]
        self.assertEqual(len(pending), 1)

    def test_adjudication_is_the_only_explicit_way_to_clear_ambiguity(self):
        first = self.registry.register(
            record("PMID_3200.md", identifiers={"pmid": "3200"},
                   legacy={"source_stem": "shared-3"})
        )
        second = self.registry.register(
            record("PMID_3201.md", sha=SHA_B, identifiers={"pmid": "3201"},
                   legacy={"source_stem": "shared-3"})
        )

        with self.assertRaises(pr.RegistryError):
            self.registry.adjudicate_legacy(
                "source_stem", "shared-3", "PAP999999", "not a candidate"
            )
        outcome = self.registry.adjudicate_legacy(
            "source_stem", "shared-3", second["paper_id"], "second file carries the old name"
        )

        self.assertEqual(outcome["status"], "resolved")
        self.assertEqual(
            self.registry.resolve_legacy("shared-3", "source_stem")["paper_id"],
            second["paper_id"],
        )
        self.assertNotEqual(first["paper_id"], second["paper_id"])
        pending = [
            item
            for item in self.registry.conflicts("pending")
            if item["kind"] == "legacy_id_ambiguous"
        ]
        self.assertEqual(pending, [])

    def test_verify_flags_a_pending_legacy_conflict_with_a_resolved_row(self):
        first = self.registry.register(
            record("PMID_3300.md", identifiers={"pmid": "3300"},
                   legacy={"source_stem": "shared-4"})
        )
        self.registry.register(
            record("PMID_3301.md", sha=SHA_B, identifiers={"pmid": "3301"},
                   legacy={"source_stem": "shared-4"})
        )
        # Simulate a later tool that overwrites the row without settling it.
        connection = sqlite3.connect(self.db)
        try:
            connection.execute(
                "UPDATE legacy_ids SET paper_id = ?, status = 'resolved', "
                "evidence = '{}' WHERE legacy_kind = 'source_stem' AND legacy_id = 'shared-4'",
                (first["paper_id"],),
            )
            connection.commit()
        finally:
            connection.close()

        checks = {item["name"]: item for item in self.registry.verify()["checks"]}
        self.assertEqual(checks["legacy_conflict_consistency"]["status"], "fail")
        self.assertEqual(
            checks["legacy_conflict_consistency"]["detail"]["total_inconsistent"], 1
        )

    def test_cli_import_keeps_a_shared_legacy_id_ambiguous_on_first_run(self):
        records_path = self._dir / "records.jsonl"
        payloads = [
            {
                "source": "PMID_3400.md",
                "content_sha256": SHA_A,
                "identifiers": {"pmid": "3400"},
                "legacy": {"source_stem": "cli-shared"},
            },
            {
                "source": "PMID_3401.md",
                "content_sha256": SHA_B,
                "identifiers": {"pmid": "3401"},
                "legacy": {"source_stem": "cli-shared"},
            },
        ]
        records_path.write_text(
            "\n".join(json.dumps(item) for item in payloads) + "\n", encoding="utf-8"
        )

        exit_code = pr.main(
            ["import-jsonl", "--db", str(self.db), "--records", str(records_path)]
        )

        self.assertEqual(exit_code, 0)
        report = self.registry.resolve_legacy("cli-shared", "source_stem")
        self.assertEqual(report["status"], "ambiguous")
        self.assertEqual(len(report["candidates"]), 2)


class B2CompatibilityTests(RegistryTestCase):
    """Old B2 identities normalise exactly like the accepted evaluator."""

    def test_source_alias_ignores_case_and_keeps_the_md_suffix(self):
        created = self.registry.register(
            record(
                "PMID_16632979.md",
                identifiers={"pmid": "16632979"},
                legacy={"source_stem": "PMID_16632979"},
            )
        )

        self.assertEqual(pr.b2_identity({"source": "PMID_16632979.md"}),
                         "source:pmid_16632979.md")
        self.assertEqual(
            self.registry.resolve_alias("source:PMID_16632979.md"), created["paper_id"]
        )
        self.assertEqual(
            self.registry.resolve_alias("source:pmid_16632979.md"), created["paper_id"]
        )

    def test_internal_whitespace_is_not_collapsed(self):
        created = self.registry.register(
            record("Li  Two Spaces.md", legacy={"source_stem": "Li  Two Spaces"})
        )

        self.assertEqual(
            self.registry.resolve_alias("source:li  two spaces.md"), created["paper_id"]
        )
        self.assertIsNone(self.registry.resolve_alias("source:li two spaces.md"))

    def test_b2_identities_from_cards_and_saved_rankings(self):
        self.registry.register(
            record(
                "PMID_3500.md",
                identifiers={"pmid": "3500", "doi": "10.1/Thirty"},
                legacy={"source_stem": "PMID_3500", "b3_paper_id": "doi:10.1/thirty"},
            )
        )
        cards = self._dir / "cards.json"
        cards.write_text(
            json.dumps([{"source": "PMID_3500.md", "doi": "10.1/Thirty"}]),
            encoding="utf-8",
        )
        rankings = self._dir / "rankings.json"
        rankings.write_text(
            json.dumps(
                {
                    "records": [
                        {
                            "query_id": "B2-Q0001",
                            "evaluation": {
                                "retrieved_rankings": [
                                    {
                                        "rank": 1,
                                        "pre_f06_identity": "source:pmid_3500.md",
                                        "doi": "",
                                        "source": "PMID_3500.md",
                                    }
                                ]
                            },
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

        result = self.registry.verify(b2_cards=cards, b2_rankings=rankings)
        checks = {item["name"]: item for item in result["checks"]}
        self.assertEqual(checks["b2_identity"]["status"], "pass")
        self.assertEqual(checks["b2_identity"]["detail"]["identities"], 2)

    def test_b2_identity_reports_an_unregistered_source(self):
        self.registry.register(
            record(
                "PMID_3600.md",
                identifiers={"pmid": "3600"},
                legacy={"source_stem": "PMID_3600"},
            )
        )
        cards = self._dir / "cards-b2.json"
        cards.write_text(
            json.dumps(
                [{"source": "PMID_3600.md", "doi": ""}, {"source": "PMID_3601.md", "doi": ""}]
            ),
            encoding="utf-8",
        )

        result = self.registry.verify(b2_cards=cards)
        checks = {item["name"]: item for item in result["checks"]}
        self.assertEqual(checks["b2_identity"]["status"], "fail")
        self.assertEqual(checks["b2_identity"]["detail"]["total_failing"], 1)


class IdentifierCorrectionTests(RegistryTestCase):
    """A correction is not undone by replaying an older import."""

    def test_replay_does_not_reactivate_a_corrected_doi(self):
        created = self.registry.register(
            record("PMID_4000.md", identifiers={"doi": "10.1/old", "pmid": "4000"})
        )
        self.registry.correct_identifier(
            "doi", "10.1/old", "10.1/new", created["paper_id"], evidence="typo"
        )

        self.registry.register(
            record("PMID_4000.md", identifiers={"doi": "10.1/old", "pmid": "4000"})
        )

        self.assertIsNone(self.registry.resolve("doi", "10.1/old"))
        self.assertEqual(self.registry.resolve("doi", "10.1/new"), created["paper_id"])
        rows = {item["value"]: item for item in self.registry.identifiers(created["paper_id"])}
        self.assertEqual(rows["10.1/old"]["status"], "superseded")
        self.assertEqual(self.registry.stats()["papers"], 1)

    def test_correcting_onto_another_papers_superseded_alias_is_rejected(self):
        first = self.registry.register(
            record("PMID_4100.md", identifiers={"doi": "10.1/first", "pmid": "4100"})
        )
        second = self.registry.register(
            record("PMID_4101.md", sha=SHA_B, identifiers={"doi": "10.1/second", "pmid": "4101"})
        )
        self.registry.correct_identifier(
            "doi", "10.1/second", "10.1/moved", second["paper_id"], evidence="moved"
        )

        with self.assertRaises(pr.RegistryError):
            self.registry.correct_identifier(
                "doi", "10.1/first", "10.1/second", first["paper_id"], evidence="wrong"
            )

        self.assertEqual(self.registry.resolve("doi", "10.1/first"), first["paper_id"])
        self.assertIsNone(self.registry.resolve("doi", "10.1/second"))


class ArtifactRoleEvidenceTests(CorpusAdapterTests):
    """Identity lookup and artifact role are decided by separate evidence."""

    def test_version_role_merges_identity_without_becoming_a_supplement(self):
        base, _supplement_map = self.write_inputs()
        records = pr.build_corpus_records(
            base / "cards.json",
            base / "manifest.json",
            base / "inventory.json",
            base / "canonical.csv",
            mapping_path=base / "mapping.json",
            supplement_map={
                "supplement": {"kind": "folder", "value": "FOLDER1", "role": "version"}
            },
        )
        by_source = {item["source"]: item for item in records}

        self.assertNotIn("supplement_of", by_source["supplement.md"])
        self.assertEqual(by_source["supplement.md"]["identifiers"]["doi"], "10.1/one")
        for item in sorted(
            records,
            key=lambda item: bool(item.get("supplement_of") or item.get("duplicate_of")),
        ):
            self.registry.register(item)
        stats = self.registry.stats()
        self.assertEqual(stats["papers"], 3)
        self.assertEqual(
            stats["artifacts_by_role"],
            {"duplicate": 1, "main": 2, "version": 2},
        )


class DuplicateIdentityTests(RegistryTestCase):
    """A duplicate claim is about bytes; it never waives identity checks."""

    def _register_shared_doi_conflict(self):
        original = self.registry.register(
            record(
                "a.md",
                identifiers={"doi": "10.1/shared", "pmid": "1"},
                metadata={"title": "Alpha", "authors": "A"},
            )
        )
        held = self.registry.register(
            record(
                "copy.md",
                identifiers={"doi": "10.1/shared", "pmid": "2"},
                metadata={"title": "Beta", "authors": "B"},
                duplicate_of={"kind": "source", "value": "a.md"},
            )
        )
        return original, held

    def test_duplicate_claim_does_not_bypass_identity_consistency(self):
        original, held = self._register_shared_doi_conflict()

        self.assertEqual(held["status"], "conflict")
        self.assertNotEqual(held["paper_id"], original["paper_id"])
        self.assertEqual(
            [issue["kind"] for issue in held["issues"]], ["identifier_mismatch"]
        )
        self.assertEqual(self.registry.resolve("doi", "10.1/shared"), original["paper_id"])
        self.assertEqual(self.registry.resolve("pmid", "1"), original["paper_id"])
        self.assertIsNone(self.registry.resolve("pmid", "2"))
        self.assertEqual(self.registry.stats()["papers"], 2)

        owner_ids = {
            (item["scheme"], item["value"], item["status"])
            for item in self.registry.identifiers(original["paper_id"])
        }
        self.assertEqual(
            owner_ids, {("doi", "10.1/shared", "active"), ("pmid", "1", "active")}
        )
        owner = self.registry.paper(original["paper_id"])
        self.assertEqual((owner["title"], owner["authors"]), ("Alpha", "A"))
        self.assertEqual(self.registry.identifiers(held["paper_id"]), [])

        conflicts = {
            item["kind"]: item
            for item in self.registry.conflicts()
            if item["kind"] in ("identifier_conflict", "content_duplicate_identifier_mismatch")
        }
        self.assertEqual(set(conflicts), {"identifier_conflict",
                                          "content_duplicate_identifier_mismatch"})
        detail = json.loads(conflicts["identifier_conflict"]["detail"])
        self.assertEqual(detail["owners"], [original["paper_id"]])
        self.assertEqual(detail["identifiers"], {"doi": "10.1/shared", "pmid": "2"})
        self.assertEqual(conflicts["identifier_conflict"]["status"], "pending")

    def test_held_duplicate_still_flags_its_shared_content(self):
        original, held = self._register_shared_doi_conflict()

        relations = self.registry.relations(held["paper_id"])
        self.assertEqual(
            [(item["kind"], item["status"]) for item in relations],
            [("exact_duplicate", "pending")],
        )
        self.assertEqual(
            {relations[0]["from_paper_id"], relations[0]["to_paper_id"]},
            {original["paper_id"], held["paper_id"]},
        )

        report = self.registry.verify()
        checks = {item["name"]: item for item in report["checks"]}
        self.assertEqual(checks["shared_content_is_flagged"]["status"], "pass")
        self.assertEqual(report["status"], "PASS")

    def test_replaying_the_held_duplicate_writes_nothing_new(self):
        _original, held = self._register_shared_doi_conflict()
        before = self.registry.stats()

        again = self.registry.register(
            record(
                "copy.md",
                identifiers={"doi": "10.1/shared", "pmid": "2"},
                metadata={"title": "Beta", "authors": "B"},
                duplicate_of={"kind": "source", "value": "a.md"},
            )
        )

        self.assertEqual(again["status"], "conflict")
        self.assertEqual(again["paper_id"], held["paper_id"])
        self.assertEqual(self.registry.stats(), before)


class LegacyFixtureTestCase(RegistryTestCase):
    """Builds a v1-shaped library from real look-alike source names."""

    TRUE_LI = (
        "Li 等 - 2026 - Comparison of Diagnostic Performance of Node-RADS "
        "and ESGAR Criteria                     in Rectal C.md"
    )

    def setUp(self):
        super().setUp()
        self.legacy_db = self._dir / "legacy.sqlite3"
        self.true_stem = pr.stem_of(self.TRUE_LI)
        self.folded_stem = " ".join(self.true_stem.split())
        self.assertNotEqual(self.true_stem, self.folded_stem)

    def open_legacy(self):
        registry = pr.PaperRegistry(self.legacy_db)
        self.addCleanup(registry.close)
        return registry

    def downgrade_to_v1(self, replacements):
        """Replay the v1 shape: folded ids, no v2 columns, v1 schema marker.

        v2 stores the lookup key in ``legacy_id``, so ``old`` is that key and
        ``new`` is the spelling v1 would have kept (folded, original case).
        """
        conn = sqlite3.connect(str(self.legacy_db))
        try:
            for legacy_kind, old, new in replacements:
                conn.execute(
                    "UPDATE legacy_ids SET legacy_id = ? "
                    "WHERE legacy_kind = ? AND legacy_id = ?",
                    (new, legacy_kind, old),
                )
            conn.execute("ALTER TABLE legacy_ids DROP COLUMN raw_ids")
            conn.execute("ALTER TABLE legacy_ids DROP COLUMN resolution")
            conn.execute(
                "UPDATE registry_meta SET value = 'p09-c1-v1' WHERE key = 'schema_version'"
            )
            conn.execute(
                "DELETE FROM registry_meta WHERE key IN ('migrated_from', 'migrated_at')"
            )
            conn.commit()
        finally:
            conn.close()

    def apply_old_v1_to_v2_migration(self):
        """Replay the superseded migration: it only ever looked at legacy_id."""
        conn = sqlite3.connect(str(self.legacy_db))
        try:
            conn.execute(
                "ALTER TABLE legacy_ids ADD COLUMN raw_ids TEXT NOT NULL DEFAULT '[]'"
            )
            conn.execute(
                "ALTER TABLE legacy_ids ADD COLUMN resolution TEXT NOT NULL DEFAULT ''"
            )
            rows = conn.execute(
                "SELECT legacy_kind, legacy_id, paper_id, artifact_source, status, "
                "evidence, created_at FROM legacy_ids"
            ).fetchall()
            conn.execute("DELETE FROM legacy_ids")
            for kind, stored, paper_id, artifact_source, status, evidence, created_at in rows:
                key = pr.normalize_legacy_id(kind, stored)
                raw_ids = [stored] if stored != key else []
                conn.execute(
                    "INSERT INTO legacy_ids (legacy_kind, legacy_id, raw_ids, paper_id, "
                    "artifact_source, status, resolution, evidence, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, '', ?, ?)",
                    (
                        kind,
                        key,
                        json.dumps(raw_ids),
                        paper_id,
                        artifact_source,
                        status,
                        evidence,
                        created_at,
                    ),
                )
            conn.execute(
                "UPDATE registry_meta SET value = 'p09-c1-v2' WHERE key = 'schema_version'"
            )
            conn.execute(
                "INSERT INTO registry_meta (key, value) VALUES "
                "('migrated_from', 'p09-c1-v1'), ('migrated_at', '2026-10-31T00:00:00Z')"
            )
            conn.commit()
        finally:
            conn.close()

    def build_v1_library(self):
        registry = pr.PaperRegistry(self.legacy_db, create=True)
        try:
            paper = registry.register(
                record(
                    self.TRUE_LI,
                    metadata={"title": self.true_stem},
                    legacy={
                        "source_stem": self.true_stem,
                        "b3_paper_id": "source:" + self.true_stem,
                    },
                )
            )
        finally:
            registry.close()
        self.downgrade_to_v1(
            [
                (
                    "source_stem",
                    pr.normalize_legacy_id("source_stem", self.true_stem),
                    self.folded_stem,
                ),
                (
                    "b3_paper_id",
                    pr.normalize_legacy_id("b3_paper_id", "source:" + self.true_stem),
                    "source:" + self.folded_stem,
                ),
            ]
        )
        return paper


class V1MigrationTests(LegacyFixtureTestCase):
    """The upgrade restores spellings the v1 text normalizer had folded."""

    def test_migration_recovers_a_source_spelling_with_consecutive_spaces(self):
        paper = self.build_v1_library()
        registry = self.open_legacy()

        self.assertEqual(
            registry.resolve_alias("source:" + self.TRUE_LI), paper["paper_id"]
        )
        self.assertEqual(
            registry.resolve_legacy(self.folded_stem, "source_stem")["paper_id"],
            paper["paper_id"],
        )
        rows = [
            row
            for row in registry.legacy_rows("source_stem")
            if row["legacy_id"]
            == pr.normalize_legacy_id("source_stem", self.true_stem)
        ]
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["status"], "resolved")
        self.assertEqual(row["paper_id"], paper["paper_id"])
        raws = set(json.loads(row["raw_ids"]))
        self.assertIn(self.true_stem, raws)
        self.assertIn(self.folded_stem, raws)
        self.assertEqual(
            registry.legacy_rows("b3_paper_id")[0]["legacy_id"],
            "source:" + self.true_stem.casefold(),
        )

        self.assertEqual(registry.paper(paper["paper_id"])["source"], self.TRUE_LI)
        self.assertEqual(registry.stats()["papers"], 1)
        self.assertEqual(registry.stats()["artifacts"], 1)
        self.assertEqual(
            registry.artifact_by_source(self.TRUE_LI)["artifact_id"],
            paper["artifact_id"],
        )

        cards = self._dir / "li-cards.json"
        cards.write_text(
            json.dumps([{"source": self.TRUE_LI, "doi": ""}]), encoding="utf-8"
        )
        checks = {
            item["name"]: item
            for item in registry.verify(b2_cards=cards)["checks"]
        }
        self.assertEqual(checks["b2_identity"]["status"], "pass")
        self.assertEqual(checks["b2_identity"]["detail"]["total_failing"], 0)

    def test_reopening_an_upgraded_library_keeps_the_recovered_spelling(self):
        paper = self.build_v1_library()
        self.open_legacy()

        reopened = self.open_legacy()
        self.assertEqual(
            reopened.resolve_alias("source:" + self.TRUE_LI), paper["paper_id"]
        )
        self.assertFalse(reopened.repair_legacy_spellings()["changed"])

    def test_repair_fixes_a_library_the_previous_migration_already_upgraded(self):
        paper = self.build_v1_library()
        self.apply_old_v1_to_v2_migration()

        registry = self.open_legacy()
        self.assertEqual(registry.stats()["legacy_ids"], 2)
        # The superseded migration kept only the folded spelling, so the real
        # B2 identity does not resolve until the repair runs.
        self.assertIsNone(registry.resolve_alias("source:" + self.TRUE_LI))

        dry = registry.repair_legacy_spellings(dry_run=True)
        self.assertEqual(dry["rows_before"], 2)
        self.assertTrue(dry["changed"])
        self.assertIsNone(registry.resolve_alias("source:" + self.TRUE_LI))

        report = registry.repair_legacy_spellings()
        self.assertTrue(report["changed"])
        self.assertEqual(
            registry.resolve_alias("source:" + self.TRUE_LI), paper["paper_id"]
        )
        self.assertEqual(
            registry.resolve_legacy(self.folded_stem, "source_stem")["paper_id"],
            paper["paper_id"],
        )
        self.assertEqual(registry.paper(paper["paper_id"])["source"], self.TRUE_LI)
        self.assertFalse(registry.repair_legacy_spellings()["changed"])


class LegacyCollisionTests(LegacyFixtureTestCase):
    def test_two_stored_ids_that_regroup_onto_one_key_become_ambiguous(self):
        registry = pr.PaperRegistry(self.legacy_db, create=True)
        try:
            first = registry.register(record("A  B.md", identifiers={"doi": "10.1/one"}))
            second = registry.register(
                record("Other.md", sha=SHA_B, identifiers={"doi": "10.1/two"})
            )
            registry.set_legacy("source_stem", "A  B", first["paper_id"], "A  B.md")
            registry.set_legacy("source_stem", "Other", second["paper_id"], "Other.md")
        finally:
            registry.close()
        self.downgrade_to_v1(
            [("source_stem", "a  b", "A B"), ("source_stem", "other", "A  B")]
        )

        migrated = self.open_legacy()
        self.assertEqual(migrated.stats()["papers"], 2)
        self.assertEqual(migrated.stats()["artifacts"], 2)
        self.assertEqual(migrated.resolve("doi", "10.1/one"), first["paper_id"])
        self.assertEqual(migrated.resolve("doi", "10.1/two"), second["paper_id"])
        self.assertIsNone(migrated.resolve_alias("source:A  B.md"))

        rows = migrated.legacy_rows("source_stem")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "ambiguous")
        self.assertIsNone(rows[0]["paper_id"])
        self.assertEqual(set(json.loads(rows[0]["raw_ids"])), {"A B", "A  B"})

        conflicts = [
            item
            for item in migrated.conflicts()
            if item["kind"] == "legacy_id_ambiguous"
        ]
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(
            json.loads(conflicts[0]["detail"])["candidates"],
            sorted([first["paper_id"], second["paper_id"]]),
        )
        checks = {
            item["name"]: item for item in migrated.verify()["checks"]
        }
        self.assertEqual(checks["legacy_resolution"]["status"], "fail")
        self.assertEqual(checks["legacy_conflict_consistency"]["status"], "pass")
        self.assertFalse(migrated.repair_legacy_spellings()["changed"])


if __name__ == "__main__":
    unittest.main()
