# -*- coding: utf-8 -*-
"""A6: local raw-content validation and bibliographic matching.

A5 publishes raw full text into ``corpus_raw/PMID_<pmid>/`` and records the
artifact in SQLite. A6 answers two independent questions about that artifact:

* ``file_valid`` - does a real parser open the file, extract a body above the
  centrally defined minimum length, and expose usable bibliographic metadata?
* ``bibliographic_match`` - does the file belong to the canonical task article,
  decided in the fixed order PMID/PMCID exact > DOI exact > title + author +
  year?

Only ``file_valid = 1`` and ``bibliographic_match = 1`` promote
``tasks.content_status`` to ``fulltext_ready``.

Deliberate boundaries:

* Local validation is not a source request. A6 never inserts into
  ``fetch_attempts`` and never fabricates ``attempt_count``.
* ``tasks.status`` stays A3-owned: A3 only reaches ``fulltext_ready`` from a
  persisted dual-validated attempt, which A6 must not fabricate. A6 therefore
  enforces the readiness gate on ``content_status`` and leaves ``status``
  untouched.
* A6 never deletes or rewrites raw files, and never writes canonical metadata
  (doi/pmcid/title/year/authors/license).
* XML and PDF only. TXT/HTML artifacts are recorded as unverified and can never
  become ``fulltext_ready``.
* A verdict is bound to the exact content version that was read (A-R1): the
  validated path/sha256 are persisted, a commit is refused when the task
  pointer moved on, and any A5 replacement clears the verdict.
* No normalization (A7), no lease/heartbeat/batch (A8), no network access.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import io
import json
import logging
import os
import re
import sys
import unicodedata
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from xml.etree import ElementTree as ET

from pypdf import PdfReader
from pypdf.errors import PyPdfError

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a1_schema import DEFAULT_DB_PATH, _connect, migrate_to_v2, utc_now
from ingest_pmids import normalize_pmids
from pubmed_metadata import normalize_author_family


RAW_ROOT = Path(__file__).resolve().parents[2] / "corpus_raw"
FORMAT_FILENAMES = {
    "xml": "article.xml",
    "txt": "article.txt",
    "html": "article.html",
    "pdf": "article.pdf",
}
# A6 validates XML and PDF. TXT/HTML stay unverified until A7 defines their
# normalization and validation rules, so they can never become fulltext_ready.
SUPPORTED_FORMATS = frozenset({"xml", "pdf"})

# A1 owns the tasks table. These additive columns are the A6 validation
# contract and deliberately do not change A1's schema version.
A6_VALIDATION_COLUMNS = {
    "file_valid": "INTEGER",
    "bibliographic_match": "INTEGER",
    "validated_at": "TEXT",
    "validation_method": "TEXT",
    "validation_error": "TEXT",
    # A-R1: the identity of the artifact this verdict was computed from. A
    # verdict is only usable while the task still points at that content.
    "validated_content_path": "TEXT",
    "validated_content_sha256": "TEXT",
}

# Every column a content replacement must clear: a stale pass must never
# survive alongside new raw bytes.
INVALIDATED_ON_CONTENT_REPLACEMENT = tuple(A6_VALIDATION_COLUMNS)

# A6 error classes are a subset of the A3 taxonomy (see state_machine.ERROR_CLASSES).
ERROR_INVALID_PDF = "invalid_pdf"
ERROR_INVALID_XML = "invalid_xml"
ERROR_CONTENT_TOO_SMALL = "content_too_small"
ERROR_IDENTIFIER_MISMATCH = "identifier_mismatch"
ERROR_METADATA_MISMATCH = "metadata_mismatch"
ERROR_PARSER = "parser_error"
# A-R1: the task pointer moved between reading the bytes and committing the
# verdict, so the verdict describes content that is no longer the incumbent.
# It reuses the module's existing content-integrity class (the same class used
# when raw bytes disagree with source.json provenance) instead of widening the
# frozen A3 taxonomy. The refusal is reported as OUTCOME_CONTENT_CHANGED.
ERROR_CONTENT_CHANGED = ERROR_PARSER
A6_ERROR_CLASSES = (
    ERROR_INVALID_PDF,
    ERROR_INVALID_XML,
    ERROR_CONTENT_TOO_SMALL,
    ERROR_IDENTIFIER_MISMATCH,
    ERROR_METADATA_MISMATCH,
    ERROR_PARSER,
)

MATCH_METHOD_IDENTIFIER = "pmid_pmcid_exact"
MATCH_METHOD_DOI = "doi_exact"
MATCH_METHOD_DESCRIPTIVE = "title_author_year"
A6_MATCH_METHODS = (
    MATCH_METHOD_IDENTIFIER,
    MATCH_METHOD_DOI,
    MATCH_METHOD_DESCRIPTIVE,
)

OUTCOME_VALIDATED = "validated"
OUTCOME_UNSUPPORTED_FORMAT = "unsupported_format"
OUTCOME_NO_RAW_CONTENT = "no_raw_content"
# A-R1: the verdict was refused because the task no longer points at the
# validated content version.
OUTCOME_CONTENT_CHANGED = "content_changed"

CONTENT_STATUS_READY = "fulltext_ready"
CONTENT_STATUS_FETCHED = "fetched"


class ValidationSchemaError(RuntimeError):
    """Raised when the A6 validation columns cannot be migrated or rolled back."""


@dataclass(frozen=True)
class ValidationLimits:
    """Every A6 threshold lives here; no validation magic number lives outside."""

    min_xml_body_chars: int = 1000
    min_pdf_text_chars: int = 1000
    min_title_similarity: float = 0.90
    year_tolerance_years: int = 1
    max_pdf_pages_scanned: int = 50
    pdf_identity_scan_chars: int = 4000
    max_text_title_candidates: int = 3


LIMITS = ValidationLimits()


@dataclass(frozen=True)
class FileEvidence:
    """What a format parser could observe about one raw artifact."""

    format: str
    file_valid: bool
    error_class: str | None = None
    detail: str | None = None
    identifiers: Mapping[str, str] = field(default_factory=dict)
    title_candidates: tuple[str, ...] = ()
    author_families: tuple[str, ...] = ()
    year: int | None = None
    text_chars: int = 0
    page_count: int | None = None


@dataclass(frozen=True)
class BibliographicMatch:
    """Result of the A6 identity decision for one artifact."""

    matched: bool | None
    method: str | None = None
    error_class: str | None = None
    detail: str | None = None


@dataclass(frozen=True)
class ContentBinding:
    """Whether an A6 verdict still describes the content the task points at.

    A-R1: ``file_valid``/``bibliographic_match`` alone never prove that the
    bytes on disk are the bytes that were validated. The binding answers that
    question from the recorded identities, and from the actual bytes whenever
    the caller already read them.
    """

    bound: bool
    reason: str
    validated_path: str | None = None
    validated_sha256: str | None = None
    expected_sha256: str | None = None
    actual_sha256: str | None = None


def _normalized_path(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    return os.path.normcase(os.path.normpath(text))


def content_binding(
    task: Mapping[str, Any], *, actual_sha256: str | None = None
) -> ContentBinding:
    """Return whether the task's A6 verdict is bound to its current content.

    ``actual_sha256`` is the digest of the bytes the caller has already read:
    supplying it turns the recorded identity check into a byte-level check.
    """

    validated_path = _text(task.get("validated_content_path"))
    validated_sha256 = (
        _text(task.get("validated_content_sha256")) or ""
    ).casefold() or None
    content_path = _text(task.get("content_path"))
    content_sha256 = (_text(task.get("content_sha256")) or "").casefold() or None
    actual = (actual_sha256 or "").casefold() or None
    expected = validated_sha256 or content_sha256

    def result(bound: bool, reason: str) -> ContentBinding:
        return ContentBinding(
            bound=bound,
            reason=reason,
            validated_path=validated_path,
            validated_sha256=validated_sha256,
            expected_sha256=expected,
            actual_sha256=actual,
        )

    if _as_int(task.get("file_valid")) != 1 or _as_int(
        task.get("bibliographic_match")
    ) != 1:
        return result(False, "file_valid/bibliographic_match 未同时通过")
    if validated_sha256 and content_sha256 and validated_sha256 != content_sha256:
        return result(
            False,
            "验证记录的 sha256 与当前 content_sha256 不一致: "
            f"{validated_sha256} != {content_sha256}",
        )
    # The recorded identity of the validated content: the dedicated column when
    # present (new rows), otherwise the A5-written content sha256 (rows verified
    # before A-R1 landed, whose verdict was computed on that same artifact).
    if not expected:
        return result(False, "任务未记录内容 sha256，无法确认验证版本")
    if actual is not None and actual != expected:
        return result(
            False,
            f"实际内容 sha256 与验证版本不一致: actual={actual} validated={expected}",
        )
    if (
        validated_path
        and content_path
        and _normalized_path(validated_path) != _normalized_path(content_path)
    ):
        return result(
            False,
            "验证记录的路径与当前 content_path 不一致: "
            f"{validated_path} != {content_path}",
        )
    return result(True, "验证结果绑定当前内容版本")


def has_validated_content(task: Mapping[str, Any]) -> bool:
    """Return whether a usable validated body exists for this task."""

    return content_binding(task).bound


# ---------------------------------------------------------------------------
# additive schema migration (A4-style: additive columns, no version bump)
# ---------------------------------------------------------------------------


def ensure_validation_schema(db_path: str | Path = DEFAULT_DB_PATH) -> dict[str, Any]:
    """Add the A6 validation columns without changing existing task rows."""

    migrate_to_v2(db_path)
    conn = _connect(db_path)
    added: list[str] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        existing = {row[1] for row in conn.execute('PRAGMA table_info("tasks")')}
        for name, definition in A6_VALIDATION_COLUMNS.items():
            if name not in existing:
                conn.execute(f'ALTER TABLE tasks ADD COLUMN "{name}" {definition}')
                added.append(name)
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return {"added_columns": added, "columns": sorted(existing | set(added))}


def rollback_validation_schema(
    db_path: str | Path = DEFAULT_DB_PATH, *, force: bool = False
) -> dict[str, Any]:
    """Drop the A6 columns; refuse while A6 validation evidence exists."""

    conn = _connect(db_path)
    present: list[str] = []
    dropped: list[str] = []
    try:
        conn.execute("BEGIN IMMEDIATE")
        columns = {row[1] for row in conn.execute('PRAGMA table_info("tasks")')}
        present = [name for name in A6_VALIDATION_COLUMNS if name in columns]
        if present and not force:
            evidence = int(
                conn.execute(
                    "SELECT COUNT(*) FROM tasks WHERE file_valid IS NOT NULL "
                    "OR bibliographic_match IS NOT NULL OR validated_at IS NOT NULL"
                ).fetchone()[0]
            )
            if evidence:
                raise ValidationSchemaError(
                    "refusing A6 rollback with validation evidence on "
                    f"{evidence} task(s); restore a verified backup or pass force=True"
                )
        for name in present:
            conn.execute(f'ALTER TABLE tasks DROP COLUMN "{name}"')
            dropped.append(name)
        conn.execute("COMMIT")
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()
    return {"dropped_columns": dropped, "rolled_back": bool(dropped)}


# ---------------------------------------------------------------------------
# normalization helpers (shared by every comparison below)
# ---------------------------------------------------------------------------


def _text(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    return value or None


def _as_int(value: Any) -> int | None:
    """Read a tri-state verification flag that may arrive as int or text."""

    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    text = _text(value)
    if text is None:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def normalized_text(value: Any) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    return " ".join(text.split())


def normalize_title(value: Any) -> str | None:
    """Return the project's punctuation-free title representation."""

    text = normalized_text(value)
    if not text:
        return None
    cleaned = "".join(char if char.isalnum() else " " for char in text)
    return " ".join(cleaned.split()) or None


def title_similarity(left: Any, right: Any) -> float:
    normalized_left = normalize_title(left)
    normalized_right = normalize_title(right)
    if not normalized_left or not normalized_right:
        return 0.0
    return difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()


_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\"'<>,;]+", re.IGNORECASE)
_DOI_TRAILING = ".,;:!?)]}>\"'"


def normalize_doi(value: Any) -> str | None:
    """Strip doi.org URLs, the ``doi:`` prefix, case and trailing punctuation.

    Only DOI-shaped values are returned: prose or a malformed canonical value
    yields None instead of a value that could be compared by accident.
    """

    text = _text(value)
    if not text:
        return None
    text = _DOI_PREFIX_RE.sub("", text).strip()
    match = _DOI_RE.search(text)
    if match is None:
        return None
    normalized = match.group(0).strip().rstrip(_DOI_TRAILING).casefold()
    return normalized or None


def normalize_pmid(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    digits = "".join(char for char in text if char.isdigit())
    if not digits:
        return None
    return str(int(digits))


def normalize_pmcid(value: Any) -> str | None:
    text = normalized_text(value).upper().replace(" ", "")
    if not text:
        return None
    if text.startswith("PMC"):
        text = text[3:]
    digits = "".join(char for char in text if char.isdigit())
    return f"PMC{digits}" if digits else None


def _family_tokens(value: Any) -> tuple[str, ...]:
    text = normalized_text(value).replace("-", " ").replace("'", " ")
    return tuple(part for part in text.split() if part)


def families_match(left: Any, right: Any) -> bool:
    """Compare two family names, tolerating particles and punctuation splits."""

    left_tokens = set(_family_tokens(left))
    right_tokens = set(_family_tokens(right))
    if not left_tokens or not right_tokens:
        return False
    if left_tokens <= right_tokens or right_tokens <= left_tokens:
        return True
    return "".join(_family_tokens(left)) == "".join(_family_tokens(right))



def _identifiers_from_text(value: Any) -> dict[str, str]:
    """Extract the identifiers a document declares about itself."""

    text = _text(value)
    if not text:
        return {}
    identifiers: dict[str, str] = {}
    doi = _DOI_RE.search(text)
    if doi:
        normalized = normalize_doi(doi.group(0))
        if normalized:
            identifiers["doi"] = normalized
    pmid = re.search(r"\bpmid[:\s#]*(\d{1,9})\b", text, re.IGNORECASE)
    if pmid:
        normalized = normalize_pmid(pmid.group(1))
        if normalized:
            identifiers["pmid"] = normalized
    pmcid = re.search(r"\bPMC[\s-]?(\d{1,9})\b", text)
    if pmcid:
        normalized = normalize_pmcid(pmcid.group(1))
        if normalized:
            identifiers["pmcid"] = normalized
    return identifiers


def _looks_like_placeholder_title(value: Any) -> bool:
    text = str(value or "").strip()
    if len(text) < 8:
        return True
    lowered = text.casefold()
    if re.search(r"\.(?:doc|docx|pdf|tex|indd|qxd|pmd)\s*$", lowered):
        return True
    return lowered.startswith(
        ("microsoft word", "untitled", "document1", "download", "http://", "https://")
    )


# ---------------------------------------------------------------------------
# XML / JATS validation
# ---------------------------------------------------------------------------


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].casefold()


def _find_all(root: ET.Element, name: str) -> list[ET.Element]:
    wanted = name.casefold()
    return [element for element in root.iter() if _local_name(element.tag) == wanted]


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    for child in element:
        if _local_name(child.tag) == name.casefold():
            return child
    return None


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    wanted = name.casefold()
    return [child for child in element if _local_name(child.tag) == wanted]


def _element_text(element: ET.Element | None) -> str | None:
    if element is None:
        return None
    text = " ".join("".join(element.itertext()).split())
    return text or None


_JATS_ID_TYPES = {"pmid": "pmid", "pmc": "pmcid", "pmcid": "pmcid", "doi": "doi"}


def _jats_identifiers(article_meta: ET.Element | None) -> dict[str, str]:
    """Read identifiers from article-meta only.

    Reference lists carry the identifiers of cited works, so identifiers are
    never mined from the whole document.
    """

    identifiers: dict[str, str] = {}
    for element in _find_all(article_meta, "article-id") if article_meta is not None else []:
        key = _JATS_ID_TYPES.get((element.get("pub-id-type") or "").strip().casefold())
        if not key or key in identifiers:
            continue
        value = _text(element.text)
        normalized = {
            "pmid": normalize_pmid,
            "pmcid": normalize_pmcid,
            "doi": normalize_doi,
        }[key](value)
        if normalized:
            identifiers[key] = normalized
    return identifiers


def _jats_title_candidates(article_meta: ET.Element | None) -> tuple[str, ...]:
    title_group = _child(article_meta, "title-group")
    candidates: list[str] = []
    for element in _children(title_group, "article-title") + _children(
        title_group, "alt-title"
    ):
        value = _element_text(element)
        if value and value not in candidates:
            candidates.append(value)
    return tuple(candidates)


def _jats_author_families(article_meta: ET.Element | None) -> tuple[str, ...]:
    families: list[str] = []
    for group in _children(article_meta, "contrib-group"):
        for contrib in _children(group, "contrib"):
            contrib_type = (contrib.get("contrib-type") or "author").strip().casefold()
            if contrib_type != "author":
                continue
            name_element = _child(contrib, "name")
            surname = _element_text(_child(name_element, "surname"))
            family = None
            if surname:
                family = normalize_author_family(
                    surname,
                    _element_text(_child(name_element, "given-names")),
                    None,
                )
            if not family:
                family = normalize_author_family(
                    _element_text(_child(contrib, "string-name"))
                    or _element_text(_child(contrib, "collab"))
                )
            if family and family not in families:
                families.append(family)
    return tuple(families)


def _jats_year(article_meta: ET.Element | None) -> int | None:
    pub_dates = _children(article_meta, "pub-date")
    preferred: list[ET.Element] = []
    for pub_type in ("epub", "ppub", "collection"):
        preferred.extend(
            element
            for element in pub_dates
            if (element.get("pub-type") or element.get("date-type") or "").strip().casefold()
            == pub_type
        )
    for element in preferred + pub_dates + _children(article_meta, "article-date"):
        year = _element_text(_child(element, "year"))
        if year and year.isdigit() and len(year) == 4:
            return int(year)
        match = re.search(r"\b((?:19|20)\d{2})\b", _element_text(element) or "")
        if match:
            return int(match.group(1))
    return None


def validate_xml_bytes(data: bytes) -> FileEvidence:
    """Parse a JATS/XML artifact and collect its validation evidence."""

    try:
        root = ET.fromstring(data)
    except ET.ParseError as exc:
        return FileEvidence(
            format="xml",
            file_valid=False,
            error_class=ERROR_INVALID_XML,
            detail=f"XML parser rejected the document: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive, parser independent
        return FileEvidence(
            format="xml",
            file_valid=False,
            error_class=ERROR_PARSER,
            detail=f"unexpected XML parser failure: {exc}",
        )
    if _local_name(root.tag) != "article":
        return FileEvidence(
            format="xml",
            file_valid=False,
            error_class=ERROR_INVALID_XML,
            detail=f"root element is <{_local_name(root.tag) or '?'}>, not a JATS <article>",
        )
    article_meta = _child(_child(root, "front"), "article-meta")
    if article_meta is None:
        article_meta = _child(root, "article-meta")
    if article_meta is None:
        return FileEvidence(
            format="xml",
            file_valid=False,
            error_class=ERROR_INVALID_XML,
            detail="JATS article has no front/article-meta element",
        )
    title_candidates = _jats_title_candidates(article_meta)
    identifiers = _jats_identifiers(article_meta)
    author_families = _jats_author_families(article_meta)
    year = _jats_year(article_meta)
    body = _child(root, "body")
    body_text = normalized_text(_element_text(body)) if body is not None else ""
    evidence = FileEvidence(
        format="xml",
        file_valid=True,
        identifiers=identifiers,
        title_candidates=title_candidates,
        author_families=author_families,
        year=year,
        text_chars=len(body_text),
    )
    if not title_candidates:
        return FileEvidence(
            format="xml",
            file_valid=False,
            error_class=ERROR_INVALID_XML,
            detail="JATS article-title is missing or empty",
            identifiers=identifiers,
            author_families=author_families,
            year=year,
            text_chars=len(body_text),
        )
    if len(body_text) < LIMITS.min_xml_body_chars:
        return FileEvidence(
            format="xml",
            file_valid=False,
            error_class=ERROR_CONTENT_TOO_SMALL,
            detail=(
                f"extracted body {len(body_text)} chars < required "
                f"{LIMITS.min_xml_body_chars}"
            ),
            identifiers=identifiers,
            title_candidates=title_candidates,
            author_families=author_families,
            year=year,
            text_chars=len(body_text),
        )
    return evidence


# ---------------------------------------------------------------------------
# PDF validation
# ---------------------------------------------------------------------------


@contextmanager
def _suppress_pypdf_warnings() -> Iterator[None]:
    """Silence pypdf warnings for expected inputs.

    Truncated or header-less artifacts are legitimate A6 inputs that must be
    classified, not reported as environmental warnings.
    """

    logger = logging.getLogger("pypdf")
    previous = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(previous)


def _pdf_metadata(reader: PdfReader) -> dict[str, str]:
    metadata = getattr(reader, "metadata", None) or {}
    return {
        str(key): str(value)
        for key, value in dict(metadata).items()
        if value is not None
    }


def _pdf_title_candidates(
    metadata: Mapping[str, str], first_page_text: str
) -> tuple[str, ...]:
    candidates: list[str] = []
    metadata_title = _text(metadata.get("/Title"))
    if metadata_title and not _looks_like_placeholder_title(metadata_title):
        candidates.append(metadata_title)
    text_titles = 0
    for line in (first_page_text or "").splitlines():
        line = " ".join(line.split())
        if _looks_like_placeholder_title(line):
            continue
        if line in candidates:
            continue
        candidates.append(line)
        text_titles += 1
        if text_titles >= LIMITS.max_text_title_candidates:
            break
    return tuple(candidates)


def _pdf_author_families(metadata: Mapping[str, str]) -> tuple[str, ...]:
    raw = _text(metadata.get("/Author"))
    if not raw:
        return ()
    families: list[str] = []
    for part in re.split(r";|\band\b|,", raw):
        family = normalize_author_family(part)
        if family and family not in families:
            families.append(family)
    return tuple(families)


def _pdf_year(metadata: Mapping[str, str], text: str) -> int | None:
    for key in ("/CreationDate", "/ModDate"):
        match = re.search(r"((?:19|20)\d{2})", metadata.get(key) or "")
        if match:
            return int(match.group(1))
    match = re.search(r"\b((?:19|20)\d{2})\b", text or "")
    return int(match.group(1)) if match else None


def validate_pdf_bytes(data: bytes) -> FileEvidence:
    """Open a PDF with the real parser and collect its validation evidence."""

    if not data:
        return FileEvidence(
            format="pdf",
            file_valid=False,
            error_class=ERROR_INVALID_PDF,
            detail="raw PDF artifact is empty",
        )
    try:
        with _suppress_pypdf_warnings():
            reader = PdfReader(io.BytesIO(data), strict=False)
            if reader.is_encrypted:
                try:
                    reader.decrypt("")
                except Exception:
                    return FileEvidence(
                        format="pdf",
                        file_valid=False,
                        error_class=ERROR_INVALID_PDF,
                        detail="PDF is encrypted and cannot be opened without a password",
                    )
            page_count = len(reader.pages)
            if page_count <= 0:
                return FileEvidence(
                    format="pdf",
                    file_valid=False,
                    error_class=ERROR_INVALID_PDF,
                    detail="PDF reports no pages",
                )
            page_texts: list[str] = []
            accumulated = 0
            for index, page in enumerate(reader.pages):
                if index >= LIMITS.max_pdf_pages_scanned:
                    break
                page_text = page.extract_text() or ""
                page_texts.append(page_text)
                accumulated += len(page_text)
                if accumulated >= LIMITS.min_pdf_text_chars:
                    break
            metadata = _pdf_metadata(reader)
    except PyPdfError as exc:
        return FileEvidence(
            format="pdf",
            file_valid=False,
            error_class=ERROR_INVALID_PDF,
            detail=f"PDF parser could not open the artifact: {exc}",
        )
    except Exception as exc:  # pragma: no cover - defensive, parser independent
        return FileEvidence(
            format="pdf",
            file_valid=False,
            error_class=ERROR_PARSER,
            detail=f"unexpected PDF parser failure: {exc}",
        )

    first_page_text = page_texts[0] if page_texts else ""
    # Identifiers come from the first page and the PDF metadata only: a full
    # scan would mine the identifiers of cited works from the reference list.
    identity_text = first_page_text[: LIMITS.pdf_identity_scan_chars]
    identifiers = _identifiers_from_text(
        "\n".join([identity_text, *metadata.values()])
    )
    full_text = "\n".join(page_texts)
    normalized_body = normalized_text(full_text)
    evidence = FileEvidence(
        format="pdf",
        file_valid=True,
        identifiers=identifiers,
        title_candidates=_pdf_title_candidates(metadata, first_page_text),
        author_families=_pdf_author_families(metadata),
        year=_pdf_year(metadata, identity_text),
        text_chars=len(normalized_body),
        page_count=page_count,
    )
    if evidence.text_chars < LIMITS.min_pdf_text_chars:
        return FileEvidence(
            format="pdf",
            file_valid=False,
            error_class=ERROR_CONTENT_TOO_SMALL,
            detail=(
                f"extracted text {evidence.text_chars} chars < required "
                f"{LIMITS.min_pdf_text_chars} over {page_count} page(s)"
            ),
            identifiers=evidence.identifiers,
            title_candidates=evidence.title_candidates,
            author_families=evidence.author_families,
            year=evidence.year,
            text_chars=evidence.text_chars,
            page_count=page_count,
        )
    return evidence


# ---------------------------------------------------------------------------
# bibliographic match
# ---------------------------------------------------------------------------


def task_author_families(task: Mapping[str, Any]) -> tuple[str, ...]:
    families: list[str] = []
    primary = normalize_author_family(
        _text(task.get("first_author_family")) or _text(task.get("first_author"))
    )
    if primary:
        families.append(primary)
    for part in re.split(r"[;|]", _text(task.get("authors")) or ""):
        family = normalize_author_family(part)
        if family and family not in families:
            families.append(family)
    return tuple(families)


def _strong_identifier_comparisons(task: Mapping[str, Any]) -> tuple[tuple, ...]:
    """Return the strong identifier comparisons in the A6 priority order."""

    return (
        ("PMID", "pmid", normalize_pmid, normalize_pmid(task.get("pmid"))),
        (
            "PMCID",
            "pmcid",
            normalize_pmcid,
            normalize_pmcid(task.get("pmcid") or task.get("pmc")),
        ),
        ("DOI", "doi", normalize_doi, normalize_doi(task.get("doi"))),
    )


def _strong_identifier_conflicts(
    evidence: FileEvidence, task: Mapping[str, Any]
) -> list[str]:
    """Return conflicts for every comparable strong identifier.

    A declared PMID/PMCID/DOI that contradicts canonical metadata is fatal: it
    can never be overridden by a similar title.
    """

    conflicts: list[str] = []
    for label, key, normalizer, task_value in _strong_identifier_comparisons(task):
        file_value = normalizer((evidence.identifiers or {}).get(key))
        if file_value and task_value and file_value != task_value:
            conflicts.append(f"{label} conflict: file={file_value} task={task_value}")
    return conflicts


def _strong_identifier_hit(
    evidence: FileEvidence, task: Mapping[str, Any]
) -> tuple[str, str, str] | None:
    for label, key, normalizer, task_value in _strong_identifier_comparisons(task):
        file_value = normalizer((evidence.identifiers or {}).get(key))
        if file_value and task_value and file_value == task_value:
            method = MATCH_METHOD_DOI if key == "doi" else MATCH_METHOD_IDENTIFIER
            return method, label, file_value
    return None


def _title_state(evidence: FileEvidence, task: Mapping[str, Any]) -> tuple[bool | None, str]:
    task_title = _text(task.get("title"))
    candidates = [value for value in evidence.title_candidates if _text(value)]
    if not task_title or not candidates:
        return None, "title not comparable (file or canonical title missing)"
    similarities = [title_similarity(candidate, task_title) for candidate in candidates]
    best = max(similarities)
    if best >= LIMITS.min_title_similarity:
        return True, f"title similarity {best:.3f} >= {LIMITS.min_title_similarity:.2f}"
    return (
        False,
        f"title similarity {best:.3f} < {LIMITS.min_title_similarity:.2f} "
        "for every candidate",
    )


def _author_state(evidence: FileEvidence, task: Mapping[str, Any]) -> tuple[bool | None, str]:
    task_families = task_author_families(task)
    if not task_families:
        return None, "author not comparable (canonical author metadata missing)"
    if evidence.author_families:
        matched = any(
            families_match(file_family, task_family)
            for file_family in evidence.author_families
            for task_family in task_families
        )
        if matched:
            return True, f"author family overlap with {list(evidence.author_families)}"
        return (
            False,
            f"author families {list(evidence.author_families)} do not overlap "
            f"canonical {list(task_families)}",
        )
    # Body text is not an author-list source: surnames can occur in
    # acknowledgements, affiliations, narrative text or references. Without
    # file-level author metadata, the required title+author+year comparison is
    # therefore not available.
    return None, "author not comparable (file author metadata missing)"


def _year_state(evidence: FileEvidence, task: Mapping[str, Any]) -> tuple[bool | None, str]:
    task_year = _text(task.get("year"))
    if not task_year or not task_year.isdigit() or not evidence.year:
        return None, "year not comparable (file or canonical year missing)"
    difference = abs(evidence.year - int(task_year))
    if difference <= LIMITS.year_tolerance_years:
        return (
            True,
            f"year {evidence.year} within {LIMITS.year_tolerance_years} of {task_year}",
        )
    return (
        False,
        f"year {evidence.year} differs from canonical {task_year} by more than "
        f"{LIMITS.year_tolerance_years}",
    )


def match_bibliographic(
    evidence: FileEvidence, task: Mapping[str, Any]
) -> BibliographicMatch:
    """Decide whether the artifact belongs to the canonical task article.

    Order: PMID/PMCID exact > DOI exact > title + author + year. Strong
    identifier conflicts always fail. Title similarity alone never passes.
    """

    conflicts = _strong_identifier_conflicts(evidence, task)
    if conflicts:
        return BibliographicMatch(
            matched=False,
            error_class=ERROR_IDENTIFIER_MISMATCH,
            detail="; ".join(conflicts),
        )
    hit = _strong_identifier_hit(evidence, task)
    if hit is not None:
        method, label, value = hit
        return BibliographicMatch(
            matched=True,
            method=method,
            detail=f"{label} exact match: {value}",
        )

    states = {
        "title": _title_state(evidence, task),
        "author": _author_state(evidence, task),
        "year": _year_state(evidence, task),
    }
    conflicts = [
        f"{name}: {detail}" for name, (state, detail) in states.items() if state is False
    ]
    if conflicts:
        return BibliographicMatch(
            matched=False,
            error_class=ERROR_METADATA_MISMATCH,
            detail="descriptive metadata conflict - " + "; ".join(conflicts),
        )
    if all(state is True for state, _detail in states.values()):
        return BibliographicMatch(
            matched=True,
            method=MATCH_METHOD_DESCRIPTIVE,
            detail="; ".join(detail for _state, detail in states.values()),
        )
    missing = [
        f"{name}: {detail}"
        for name, (state, detail) in states.items()
        if state is None
    ]
    return BibliographicMatch(
        matched=None,
        detail="insufficient comparable evidence - " + "; ".join(missing),
    )


# ---------------------------------------------------------------------------
# artifact location, persistence and orchestration
# ---------------------------------------------------------------------------


def _format_from_path(path: Path) -> str | None:
    suffix = path.suffix.casefold().lstrip(".")
    if suffix == "htm":
        return "html"
    return suffix if suffix in FORMAT_FILENAMES else None


def locate_raw_artifact(
    pmid: int, task: Mapping[str, Any], raw_root: str | Path
) -> tuple[Path, str] | None:
    """Return the recorded raw artifact for one task, or None."""

    content_path = _text(task.get("content_path"))
    if content_path:
        candidate = Path(content_path)
        if candidate.is_file():
            return candidate, _format_from_path(candidate) or ""
    directory = Path(raw_root) / f"PMID_{int(pmid)}"
    if not directory.is_dir():
        return None
    source_json = directory / "source.json"
    if source_json.is_file():
        try:
            recorded = json.loads(source_json.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = {}
        fmt = _text((recorded or {}).get("format"))
        if fmt in FORMAT_FILENAMES:
            candidate = directory / FORMAT_FILENAMES[fmt]
            if candidate.is_file():
                return candidate, fmt
    for fmt, filename in FORMAT_FILENAMES.items():
        candidate = directory / filename
        if candidate.is_file():
            return candidate, fmt
    return None


def _recorded_sha256(path: Path) -> str | None:
    source_json = path.parent / "source.json"
    if not source_json.is_file():
        return None
    try:
        recorded = json.loads(source_json.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    value = _text((recorded or {}).get("sha256"))
    return value.casefold() if value else None


def _gated_content_status(
    current: Any, file_valid: int | None, bibliographic_match: int | None
) -> Any:
    """Return the content status allowed by one A6 verdict."""

    if file_valid == 1 and bibliographic_match == 1:
        return CONTENT_STATUS_READY
    if current == CONTENT_STATUS_READY:
        # Never leave a stale ready state behind an unverified artifact.
        return CONTENT_STATUS_FETCHED
    return current


def _validate_error_class(error_class: str | None) -> str | None:
    if error_class is None:
        return None
    if error_class not in A6_ERROR_CLASSES:
        raise ValueError(f"non-A6 error class: {error_class!r}")
    return error_class


def _persist_verdict(
    pmid: int,
    *,
    file_valid: int | None,
    bibliographic_match: int | None,
    method: str | None,
    error_class: str | None,
    detail: str | None,
    validated_at: str,
    db_path: str | Path,
    validated_content_path: str | None = None,
    validated_content_sha256: str | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    """Write one validation verdict for one task in a single transaction.

    A-R1: the verdict is only allowed to pass while the task still points at
    the content version that was read. A pointer that moved on between the read
    and this commit invalidates the verdict instead of promoting it.
    """

    if method is not None and method not in A6_MATCH_METHODS:
        raise ValueError(f"non-A6 match method: {method!r}")
    content_changed = False
    conn = _connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            "SELECT status, content_status, content_path, content_sha256, "
            "worker_id, lease_until FROM tasks WHERE pmid=?",
            (pmid,),
        ).fetchone()
        if row is None:
            raise ValueError(f"task PMID {pmid} does not exist")
        # A-R2: a verdict write is a task-state commit and obeys the incumbent
        # lease; a worker that lost ownership must not promote the task.
        from lease_batch import assert_write_owner_in_connection

        assert_write_owner_in_connection(
            conn,
            pmid,
            worker_id,
            now=validated_at,
            operation="A6 validation verdict",
        )
        if file_valid == 1:
            conflict = _content_version_conflict(
                row, validated_content_path, validated_content_sha256
            )
            if conflict is not None:
                content_changed = True
                file_valid = 0
                bibliographic_match = None
                method = None
                error_class = ERROR_CONTENT_CHANGED
                detail = conflict
        if file_valid == 1 and bibliographic_match == 1:
            binding_path = _text(validated_content_path)
            binding_sha256 = (_text(validated_content_sha256) or "").casefold() or None
        else:
            # No usable verdict means no validated content version either.
            binding_path = None
            binding_sha256 = None
        content_status = _gated_content_status(
            row["content_status"], file_valid, bibliographic_match
        )
        conn.execute(
            """UPDATE tasks SET file_valid=?, bibliographic_match=?, validated_at=?,
                validation_method=?, validation_error=?, content_status=?,
                validated_content_path=?, validated_content_sha256=?, updated_at=?
                WHERE pmid=?""",
            (
                file_valid,
                bibliographic_match,
                validated_at,
                method,
                _validate_error_class(error_class),
                content_status,
                binding_path,
                binding_sha256,
                validated_at,
                pmid,
            ),
        )
        if conn.total_changes != 1:
            raise ValueError(f"task PMID {pmid} disappeared during validation update")
        conn.execute("COMMIT")
        return {
            "status": row["status"],
            "content_status": content_status,
            "detail": detail,
            "error_class": error_class,
            "content_changed": content_changed,
        }
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _content_version_conflict(
    row: Mapping[str, Any],
    validated_content_path: str | None,
    validated_content_sha256: str | None,
) -> str | None:
    """Return why the current task pointer is no longer the validated input.

    Only two recorded values that disagree count as a conflict: a NULL pointer
    means "nothing was recorded yet" (a migration or fixture row), not a change.
    """

    current_path = _text(row["content_path"]) if "content_path" in row.keys() else None
    current_sha256 = _text(row["content_sha256"]) if "content_sha256" in row.keys() else None
    validated_sha256 = (_text(validated_content_sha256) or "").casefold() or None
    if (
        validated_sha256
        and current_sha256
        and validated_sha256 != current_sha256.casefold()
    ):
        return (
            "content version changed during validation: task content_sha256="
            f"{current_sha256.casefold()} validated={validated_sha256}"
        )
    if (
        validated_content_path
        and current_path
        and _normalized_path(validated_content_path) != _normalized_path(current_path)
    ):
        return (
            "content pointer changed during validation: task content_path="
            f"{current_path} validated={validated_content_path}"
        )
    return None


def validate_task_content(
    pmid: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    raw_root: str | Path = RAW_ROOT,
    validated_at: str | None = None,
    worker_id: str | None = None,
) -> dict[str, Any]:
    """Validate one task's recorded raw artifact and persist the verdict.

    ``worker_id`` is the A8 identity of the caller when it holds the task lease;
    passing it lets the verdict commit prove current ownership (A-R2).
    """

    pmid = normalize_pmids([pmid])[0]
    ensure_validation_schema(db_path)
    conn = _connect(db_path)
    try:
        task_row = conn.execute("SELECT * FROM tasks WHERE pmid=?", (pmid,)).fetchone()
    finally:
        conn.close()
    if task_row is None:
        raise ValueError(f"task PMID {pmid} does not exist")
    task = dict(task_row)
    timestamp = validated_at or utc_now()
    located = locate_raw_artifact(pmid, task, raw_root)
    if located is None:
        detail = "no raw artifact is recorded for this task"
        persisted = _persist_verdict(
            pmid,
            file_valid=0,
            bibliographic_match=None,
            method=None,
            error_class=None,
            detail=detail,
            validated_at=timestamp,
            db_path=db_path,
            worker_id=worker_id,
        )
        return {
            "pmid": pmid,
            "outcome": OUTCOME_NO_RAW_CONTENT,
            "written": True,
            "format": None,
            "content_path": None,
            "file_valid": 0,
            "bibliographic_match": None,
            "validation_method": None,
            "validation_error": None,
            "validation_detail": detail,
            "validated_at": timestamp,
            "content_status": persisted["content_status"],
            "status": persisted["status"],
        }

    path, fmt = located
    data = path.read_bytes()
    recorded_sha256 = _recorded_sha256(path)
    actual_sha256 = hashlib.sha256(data).hexdigest()
    outcome = OUTCOME_VALIDATED
    evidence: FileEvidence | None = None
    match = BibliographicMatch(matched=None, detail="identity was not evaluated")
    if fmt not in SUPPORTED_FORMATS:
        outcome = OUTCOME_UNSUPPORTED_FORMAT
        detail = (
            f"A6 validates XML and PDF only; format {fmt or 'unknown'} stays unverified"
        )
    elif recorded_sha256 and recorded_sha256 != actual_sha256:
        evidence = FileEvidence(
            format=fmt,
            file_valid=False,
            error_class=ERROR_PARSER,
            detail=(
                "raw content sha256 does not match source.json provenance: "
                f"recorded={recorded_sha256} actual={actual_sha256}"
            ),
        )
    elif fmt == "xml":
        evidence = validate_xml_bytes(data)
    else:
        evidence = validate_pdf_bytes(data)

    if evidence is not None:
        if evidence.file_valid:
            match = match_bibliographic(evidence, task)
        else:
            match = BibliographicMatch(
                matched=None,
                detail="content validation failed; identity was not evaluated",
            )
        file_valid: int | None = 1 if evidence.file_valid else 0
        bibliographic_match = (
            None if match.matched is None else int(bool(match.matched))
        )
        error_class = (
            evidence.error_class if not evidence.file_valid else match.error_class
        )
        detail = evidence.detail if not evidence.file_valid else match.detail
        method = match.method if bibliographic_match == 1 else None
    else:
        file_valid = None
        bibliographic_match = None
        error_class = None
        method = None

    persisted = _persist_verdict(
        pmid,
        file_valid=file_valid,
        bibliographic_match=bibliographic_match,
        method=method,
        error_class=_validate_error_class(error_class),
        detail=detail,
        validated_at=timestamp,
        db_path=db_path,
        validated_content_path=str(path),
        validated_content_sha256=actual_sha256,
        worker_id=worker_id,
    )
    error_class = persisted["error_class"]
    detail = persisted["detail"]
    file_valid = 0 if persisted["content_changed"] else file_valid
    bibliographic_match = None if persisted["content_changed"] else bibliographic_match
    method = None if persisted["content_changed"] else method
    if persisted["content_changed"]:
        outcome = OUTCOME_CONTENT_CHANGED
    result: dict[str, Any] = {
        "pmid": pmid,
        "outcome": outcome,
        "written": True,
        "format": fmt,
        "content_path": str(path),
        "file_valid": file_valid,
        "bibliographic_match": bibliographic_match,
        "validation_method": method,
        "validation_error": error_class,
        "validation_detail": detail,
        "validated_at": timestamp,
        "content_status": persisted["content_status"],
        "status": persisted["status"],
        "content_sha256": actual_sha256,
        "content_changed": persisted["content_changed"],
    }
    if evidence is not None:
        result.update(
            {
                "text_chars": evidence.text_chars,
                "page_count": evidence.page_count,
                "identifiers": dict(evidence.identifiers),
                "title_candidates": list(evidence.title_candidates),
                "author_families": list(evidence.author_families),
                "file_year": evidence.year,
            }
        )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="A6 raw content validation and bibliographic match"
    )
    subparsers = parser.add_subparsers(dest="command")
    validate = subparsers.add_parser("validate", help="validate one PMID's raw artifact")
    validate.add_argument("pmid", type=int)
    validate.add_argument("--db", default=str(DEFAULT_DB_PATH))
    validate.add_argument("--raw-root", default=str(RAW_ROOT))
    migrate = subparsers.add_parser("migrate", help="add the A6 validation columns")
    migrate.add_argument("--db", default=str(DEFAULT_DB_PATH))
    rollback = subparsers.add_parser("rollback", help="drop the A6 validation columns")
    rollback.add_argument("--db", default=str(DEFAULT_DB_PATH))
    rollback.add_argument("--force", action="store_true")
    parser.set_defaults(command="validate")
    args = parser.parse_args(argv)

    if args.command == "migrate":
        print(json.dumps(ensure_validation_schema(args.db), ensure_ascii=False, sort_keys=True))
        return 0
    if args.command == "rollback":
        result = rollback_validation_schema(args.db, force=args.force)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    if getattr(args, "pmid", None) is None:
        parser.error("validate requires a PMID")
    result = validate_task_content(
        args.pmid, db_path=args.db, raw_root=args.raw_root
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return (
        0
        if result["file_valid"] == 1 and result["bibliographic_match"] == 1
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(run_cli())
