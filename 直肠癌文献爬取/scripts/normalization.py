# -*- coding: utf-8 -*-
"""A7: normalize JATS/XML, TXT and PDF full text into stable Markdown.

The module has two deliberately small layers:

* format parsers turn one source artifact into a title and Markdown body;
* normalize_artifact adds the common front matter and writes normalized/PMID_<pmid>.md.

The PMID adapter is intentionally separate from the parsers. It reads the
canonical task metadata and the A5 source.json sidecar, and by default
requires the A6 fulltext_ready gate plus the A-R1 content-version binding
before producing a normalized file. No production corpus or index is touched
by importing this module.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from a1_schema import DEFAULT_DB_PATH, _connect, migrate_to_v2  # noqa: E402
from content_validation import content_binding, locate_raw_artifact  # noqa: E402


PROJECT = Path(__file__).resolve().parents[2]
RAW_ROOT = PROJECT / "corpus_raw"
NORMALIZED_ROOT = PROJECT / "normalized"
FRONT_MATTER_FIELDS = (
    "pmid",
    "doi",
    "pmcid",
    "title",
    "year",
    "journal",
    "source",
    "license",
    "content_sha256",
)
SUPPORTED_FORMATS = frozenset({"xml", "txt", "pdf", "md", "markdown"})
MINERU_FLASH_LIMIT_BYTES = 10 * 1024 * 1024
MINERU_EXTRACT_TIMEOUT = "7200"


class NormalizationError(ValueError):
    """Raised when an artifact cannot safely become a normalized document."""


@dataclass(frozen=True)
class ParsedDocument:
    """Format-independent parse result used by the common writer."""

    title: str | None
    markdown: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _local_name(tag: Any) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.rsplit("}", 1)[-1].casefold()


def _child(element: ET.Element | None, name: str) -> ET.Element | None:
    if element is None:
        return None
    wanted = name.casefold()
    for child in element:
        if _local_name(child.tag) == wanted:
            return child
    return None


def _children(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    wanted = name.casefold()
    return [child for child in element if _local_name(child.tag) == wanted]


def _find_all(element: ET.Element | None, name: str) -> list[ET.Element]:
    if element is None:
        return []
    wanted = name.casefold()
    return [item for item in element.iter() if _local_name(item.tag) == wanted]


def _text(element: ET.Element | None) -> str:
    if element is None:
        return ""
    return " ".join("".join(element.itertext()).split())


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").replace("\ufeff", "").split())


def _nonempty(value: Any) -> str | None:
    cleaned = _clean_text(value)
    return cleaned or None


def _normalize_doi(value: Any) -> str | None:
    cleaned = _nonempty(value)
    if not cleaned:
        return None
    cleaned = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", cleaned, flags=re.I)
    cleaned = re.sub(r"^doi:\s*", "", cleaned, flags=re.I)
    return cleaned.rstrip(".,;)").casefold() or None


def _normalize_pmcid(value: Any) -> str | None:
    cleaned = _nonempty(value)
    if not cleaned:
        return None
    if not cleaned.upper().startswith("PMC"):
        cleaned = "PMC" + cleaned
    return cleaned.upper()


def _normalize_pmid(value: Any) -> str | None:
    cleaned = _nonempty(value)
    if not cleaned:
        return None
    match = re.search(r"\d+", cleaned)
    return match.group(0) if match else None


def _normalize_year(value: Any) -> int | None:
    if value is None:
        return None
    match = re.search(r"\b((?:19|20)\d{2})\b", str(value))
    return int(match.group(1)) if match else None


def _mapping_value(mapping: Mapping[str, Any], *names: str) -> Any:
    lowered = {str(key).casefold(): value for key, value in mapping.items()}
    for name in names:
        value = lowered.get(name.casefold())
        if value not in (None, ""):
            return value
    return None


def _canonical_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Map A1/A5/source.json spellings to the A7 front-matter contract."""

    result: dict[str, Any] = {}
    result.update(
        {
            "pmid": _normalize_pmid(_mapping_value(metadata, "pmid")),
            "doi": _normalize_doi(_mapping_value(metadata, "doi")),
            "pmcid": _normalize_pmcid(_mapping_value(metadata, "pmcid", "pmc")),
            "title": _nonempty(_mapping_value(metadata, "title")),
            "year": _normalize_year(_mapping_value(metadata, "year")),
            "journal": _nonempty(_mapping_value(metadata, "journal", "journal_title")),
            "source": _nonempty(_mapping_value(metadata, "source", "content_source")),
            "license": _nonempty(_mapping_value(metadata, "license")),
        }
    )
    return result


def _merge_metadata(*mappings: Mapping[str, Any]) -> dict[str, Any]:
    """Merge non-empty metadata left-to-right, preserving canonical values."""

    result: dict[str, Any] = {}
    for mapping in mappings:
        for key, value in _canonical_metadata(mapping).items():
            if value not in (None, ""):
                result[key] = value
    return result


def _heading_key(value: str) -> str:
    value = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", value)
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


_HEADING_ALIASES = {
    "abstract": "Abstract",
    "summary": "Abstract",
    "introduction": "Introduction",
    "background": "Introduction",
    "materials methods": "Methods",
    "materials and methods": "Methods",
    "methods": "Methods",
    "methodology": "Methods",
    "results": "Results",
    "results findings": "Results",
    "discussion": "Discussion",
    "conclusion": "Conclusion",
    "conclusions": "Conclusion",
    "supplement": "Supplementary information",
    "supplements": "Supplementary information",
    "supplementary": "Supplementary information",
    "supplementary information": "Supplementary information",
    "supporting information": "Supplementary information",
    "references": "References",
    "reference": "References",
}


def _canonical_heading(value: str) -> str | None:
    return _HEADING_ALIASES.get(_heading_key(value))


def _append_paragraph(lines: list[str], value: str) -> None:
    value = _clean_text(value)
    if not value:
        return
    if lines and lines[-1] != "":
        lines.append("")
    lines.append(value)
    lines.append("")


def _append_heading(lines: list[str], value: str, level: int) -> None:
    value = _clean_text(value)
    if not value:
        return
    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"{'#' * max(1, level)} {value}")
    lines.append("")


def _markdown_cell(value: str) -> str:
    return _clean_text(value).replace("|", "\\|") or " "


def _table_markdown(table_wrap: ET.Element) -> tuple[str, str]:
    label = _text(_child(table_wrap, "label"))
    caption = _text(_child(table_wrap, "caption"))
    heading = " ".join(part for part in (label, caption) if part).strip() or "Table"
    table = _child(table_wrap, "table")
    rows: list[list[str]] = []
    for row in _find_all(table, "tr"):
        cells = [
            _markdown_cell(_text(cell))
            for cell in row
            if _local_name(cell.tag) in {"th", "td", "entry"}
        ]
        if cells:
            rows.append(cells)
    if not rows:
        body = caption or _text(table_wrap)
        return heading, body
    width = max(len(row) for row in rows)
    rows = [row + [" "] * (width - len(row)) for row in rows]
    header = rows[0]
    markdown = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    markdown.extend("| " + " | ".join(row) + " |" for row in rows[1:])
    if caption:
        return heading, f"{caption}\n\n" + "\n".join(markdown)
    return heading, "\n".join(markdown)


def _figure_markdown(figure: ET.Element) -> tuple[str, str]:
    label = _text(_child(figure, "label"))
    caption = _text(_child(figure, "caption"))
    heading = " ".join(part for part in (label, caption) if part).strip() or "Figure"
    assets: list[str] = []
    for graphic in _find_all(figure, "graphic") + _find_all(figure, "inline-graphic"):
        for key, value in graphic.attrib.items():
            if _local_name(key) == "href" and value:
                assets.append(value)
                break
    body_parts = [caption] if caption else []
    body_parts.extend(f"Figure asset: {asset}" for asset in assets)
    return heading, "\n\n".join(part for part in body_parts if part) or "Figure"


def _render_reference_list(ref_list: ET.Element, lines: list[str]) -> None:
    _append_heading(lines, "References", 2)
    for ref in _children(ref_list, "ref"):
        value = _text(ref)
        if value:
            label = _text(_child(ref, "label"))
            if label and not value.startswith(label):
                value = f"{label} {value}"
            lines.append(f"- {value}")
    if lines and lines[-1] != "":
        lines.append("")


def _render_supplementary(element: ET.Element, lines: list[str], level: int = 2) -> None:
    _append_heading(lines, "Supplementary information", level)
    title = (
        _text(_child(element, "title"))
        or _text(_child(element, "caption"))
        or _text(_child(element, "label"))
    )
    if title:
        _append_paragraph(lines, title)
    for child in element:
        name = _local_name(child.tag)
        if name in {"title", "caption", "label"}:
            continue
        if name in {"p", "abstract", "mixed-citation", "supplementary-material"}:
            _append_paragraph(lines, _text(child))
        elif name == "media":
            href = next(
                (value for key, value in child.attrib.items() if _local_name(key) == "href"),
                None,
            )
            if href:
                _append_paragraph(lines, f"Supplementary asset: {href}")
            else:
                _append_paragraph(lines, _text(child))
        elif name == "table-wrap":
            heading, body = _table_markdown(child)
            _append_heading(lines, heading, level + 1)
            _append_paragraph(lines, body)
        elif name == "fig":
            heading, body = _figure_markdown(child)
            _append_heading(lines, heading, level + 1)
            _append_paragraph(lines, body)


def _render_sec(section: ET.Element, lines: list[str], level: int = 2) -> None:
    title_element = _child(section, "title")
    title = _text(title_element) or "Section"
    heading = _canonical_heading(title) or title
    _append_heading(lines, heading, level)
    for child in section:
        name = _local_name(child.tag)
        if name == "title":
            continue
        if name == "sec":
            _render_sec(child, lines, min(level + 1, 6))
        elif name in {"p", "disp-quote", "statement", "boxed-text", "list"}:
            if name == "list":
                for item in _children(child, "list-item"):
                    text = _text(item)
                    if text:
                        lines.append(f"- {text}")
                if lines and lines[-1] != "":
                    lines.append("")
            else:
                _append_paragraph(lines, _text(child))
        elif name == "table-wrap":
            heading_text, body = _table_markdown(child)
            _append_heading(lines, heading_text, min(level + 1, 6))
            _append_paragraph(lines, body)
        elif name == "fig":
            heading_text, body = _figure_markdown(child)
            _append_heading(lines, heading_text, min(level + 1, 6))
            _append_paragraph(lines, body)
        elif name == "supplementary-material":
            _render_supplementary(child, lines, min(level + 1, 6))


def _render_abstract(abstract: ET.Element, lines: list[str]) -> None:
    _append_heading(lines, "Abstract", 2)
    found = False
    for child in abstract:
        if _local_name(child.tag) in {"title", "label"}:
            continue
        if _local_name(child.tag) in {"p", "sec", "abstract"}:
            _append_paragraph(lines, _text(child))
            found = True
    if not found:
        _append_paragraph(lines, _text(abstract))


def _parse_jats_metadata(root: ET.Element) -> dict[str, Any]:
    front = _child(root, "front")
    article_meta = _child(front, "article-meta")
    if article_meta is None:
        article_meta = _child(root, "article-meta")
    journal_meta = _child(front, "journal-meta")
    metadata: dict[str, Any] = {}
    for article_id in _find_all(article_meta, "article-id"):
        id_type = (article_id.get("pub-id-type") or "").casefold().strip()
        value = _text(article_id)
        if id_type == "pmid" and "pmid" not in metadata:
            metadata["pmid"] = _normalize_pmid(value)
        elif id_type in {"pmc", "pmcid"} and "pmcid" not in metadata:
            metadata["pmcid"] = _normalize_pmcid(value)
        elif id_type == "doi" and "doi" not in metadata:
            metadata["doi"] = _normalize_doi(value)
    metadata["title"] = _text(_child(_child(article_meta, "title-group"), "article-title")) or None
    pub_date = _children(article_meta, "pub-date")
    for candidate in pub_date + _find_all(article_meta, "article-date"):
        year = _normalize_year(_text(_child(candidate, "year")) or _text(candidate))
        if year:
            metadata["year"] = year
            break
    metadata["journal"] = _text(_child(_child(journal_meta, "journal-title-group"), "journal-title")) or None
    return {key: value for key, value in metadata.items() if value not in (None, "")}


def parse_jats_xml(data: bytes | str) -> ParsedDocument:
    """Parse a JATS article while retaining sections, tables, figures and refs."""

    raw = data.encode("utf-8") if isinstance(data, str) else bytes(data)
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise NormalizationError(f"invalid XML/JATS document: {exc}") from exc
    if _local_name(root.tag) != "article":
        raise NormalizationError("JATS root element must be <article>")

    metadata = _parse_jats_metadata(root)
    title = metadata.get("title") or "Untitled article"
    lines: list[str] = []
    _append_heading(lines, title, 1)

    front = _child(root, "front")
    article_meta = _child(front, "article-meta")
    if article_meta is None:
        article_meta = _child(root, "article-meta")
    for abstract in _children(article_meta, "abstract"):
        _render_abstract(abstract, lines)

    body = _child(root, "body")
    unsectioned_started = False
    if body is not None:
        for child in body:
            name = _local_name(child.tag)
            if name == "sec":
                _render_sec(child, lines)
            elif name in {"p", "disp-quote", "statement", "boxed-text"}:
                if not unsectioned_started:
                    _append_heading(lines, "Full text", 2)
                    unsectioned_started = True
                _append_paragraph(lines, _text(child))
            elif name == "table-wrap":
                _append_heading(lines, "Tables", 2)
                heading, body_text = _table_markdown(child)
                _append_heading(lines, heading, 3)
                _append_paragraph(lines, body_text)
            elif name == "fig":
                _append_heading(lines, "Figures", 2)
                heading, body_text = _figure_markdown(child)
                _append_heading(lines, heading, 3)
                _append_paragraph(lines, body_text)
            elif name == "supplementary-material":
                _render_supplementary(child, lines)

    back = _child(root, "back")
    if back is not None:
        for child in back:
            name = _local_name(child.tag)
            if name == "ref-list":
                _render_reference_list(child, lines)
            elif name == "supplementary-material":
                _render_supplementary(child, lines)
            elif name == "sec":
                _render_sec(child, lines)

    markdown = "\n".join(lines).strip() + "\n"
    return ParsedDocument(title=title, markdown=markdown, metadata=metadata)


def _strip_existing_front_matter(lines: list[str]) -> list[str]:
    if lines and lines[0].strip() == "---":
        try:
            end = next(index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---")
        except StopIteration:
            return lines
        return lines[end + 1 :]
    return lines


def _text_heading(line: str) -> str | None:
    value = line.strip()
    if not value:
        return None
    value = re.sub(r"^#{1,6}\s*", "", value)
    value = re.sub(r"^\s*\d+(?:\.\d+)*[.)]?\s+", "", value)
    return value.strip()


def _is_table_heading(value: str) -> bool:
    return bool(re.match(r"^table(?:\s+\d+)?(?:\s*[:.)-]|\s+|$)", value, flags=re.I))


def _is_figure_heading(value: str) -> bool:
    return bool(re.match(r"^fig(?:ure)?(?:\s+\d+)?(?:\s*[:.)-]|\s+|$)", value, flags=re.I))


def parse_text(data: bytes | str) -> ParsedDocument:
    """Normalize plain TXT or MinerU Markdown text into stable headings."""

    raw = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else str(data)
    source_lines = _strip_existing_front_matter(raw.replace("\r\n", "\n").replace("\r", "\n").splitlines())
    lines = [line.rstrip() for line in source_lines]
    nonempty = next((line.strip() for line in lines if line.strip()), "Untitled article")
    title = _text_heading(nonempty) or nonempty
    if re.match(r"^title\s*:", title, flags=re.I):
        title = title.split(":", 1)[1].strip() or "Untitled article"

    output: list[str] = []
    _append_heading(output, title, 1)
    first_nonempty = True
    active_heading: str | None = None
    tables_started = False
    figures_started = False
    recognized_seen = False

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            if output and output[-1] != "":
                output.append("")
            continue
        if first_nonempty and stripped == nonempty:
            first_nonempty = False
            continue
        first_nonempty = False

        candidate = _text_heading(stripped)
        canonical = _canonical_heading(candidate or "") if candidate else None
        if candidate and canonical:
            _append_heading(output, canonical, 2)
            active_heading = canonical
            recognized_seen = True
            continue
        if candidate and _is_table_heading(candidate):
            if not tables_started:
                _append_heading(output, "Tables", 2)
                tables_started = True
            _append_heading(output, candidate, 3)
            active_heading = "Tables"
            recognized_seen = True
            continue
        if candidate and _is_figure_heading(candidate):
            if not figures_started:
                _append_heading(output, "Figures", 2)
                figures_started = True
            _append_heading(output, candidate, 3)
            active_heading = "Figures"
            recognized_seen = True
            continue
        if active_heading is None and not recognized_seen:
            _append_heading(output, "Full text", 2)
            active_heading = "Full text"
        output.append(stripped)

    markdown = "\n".join(output).strip() + "\n"
    return ParsedDocument(title=title, markdown=markdown, metadata={"title": title})


def parse_txt(data: bytes | str) -> ParsedDocument:
    """Explicit TXT spelling for callers of the shared text parser."""

    return parse_text(data)


MinerUConverter = Callable[[Path], str | bytes | Path]


def _mineru_markdown_candidates(output_dir: Path, source_path: Path) -> list[Path]:
    """Find Markdown layouts emitted by supported MinerU CLI versions."""

    stem = source_path.stem
    preferred = [
        output_dir / f"{stem}.md",
        output_dir / stem / f"{stem}.md",
        output_dir / stem / "output.md",
        output_dir / stem / "0.md",
    ]
    candidates = [path for path in preferred if path.is_file()]
    if candidates:
        return candidates
    return sorted(path for path in output_dir.rglob("*.md") if path.is_file())


def _mineru_result_text(
    result: subprocess.CompletedProcess[str],
    output_dir: Path,
    source_path: Path,
) -> str | None:
    candidates = _mineru_markdown_candidates(output_dir, source_path)
    if candidates:
        return candidates[0].read_text(encoding="utf-8")
    stdout = result.stdout.strip()
    return stdout or None


def _run_mineru_command(
    command: Sequence[str],
    *,
    cwd: Path,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(cwd),
            check=False,
        )
    except OSError as exc:
        raise NormalizationError(
            "MinerU command is unavailable; install mineru-open-api and ensure it is on PATH"
        ) from exc


def convert_pdf_with_mineru(source_path: str | Path) -> str:
    """Convert one PDF with the project's existing MinerU CLI workflow.

    Small PDFs use ``flash-extract`` first and retry with ``extract`` when a
    token is available. Large PDFs use ``extract`` directly. The converter
    returns only MinerU Markdown; PDF bytes are never used as normalized body
    text. Temporary output is removed when this function returns.
    """

    path = Path(source_path)
    if not path.is_file():
        raise NormalizationError(f"PDF source does not exist: {path}")
    token = os.environ.get("MINERU_TOKEN", "")
    use_extract = path.stat().st_size > MINERU_FLASH_LIMIT_BYTES
    if use_extract and not token:
        raise NormalizationError(
            f"PDF exceeds {MINERU_FLASH_LIMIT_BYTES // (1024 * 1024)} MB; "
            "MinerU extract mode requires MINERU_TOKEN"
        )

    with tempfile.TemporaryDirectory(prefix=f".a7-mineru-{path.stem}-") as temporary:
        output_dir = Path(temporary)
        attempts: list[tuple[str, list[str], Path]] = []
        if not use_extract:
            attempts.append(
                (
                    "flash-extract",
                    [
                        "mineru-open-api",
                        "flash-extract",
                        str(path),
                        "-o",
                        str(output_dir),
                        "--language",
                        "en",
                    ],
                    PROJECT,
                )
            )
        if use_extract or token:
            attempts.append(
                (
                    "extract" if use_extract else "extract-retry",
                    [
                        "mineru-open-api",
                        "extract",
                        str(path),
                        "--format",
                        "md",
                        "--model",
                        "pipeline",
                        "--language",
                        "en",
                        "--timeout",
                        MINERU_EXTRACT_TIMEOUT,
                    ],
                    output_dir,
                )
            )

        failures: list[str] = []
        for mode, command, cwd in attempts:
            result = _run_mineru_command(command, cwd=cwd)
            if result.returncode == 0:
                markdown = _mineru_result_text(result, output_dir, path)
                if markdown:
                    return markdown
            detail = (result.stderr or result.stdout or "no Markdown output").strip()
            failures.append(f"{mode} exit={result.returncode}: {detail[:500]}")

    raise NormalizationError("MinerU conversion failed: " + " | ".join(failures))


def _parse_pdf_with_mineru(
    source_path: Path,
    converter: MinerUConverter | None = None,
) -> ParsedDocument:
    active_converter = converter or convert_pdf_with_mineru
    try:
        converted = active_converter(source_path)
    except NormalizationError:
        raise
    except Exception as exc:  # noqa: BLE001 - normalize converter failures
        raise NormalizationError(f"MinerU PDF conversion failed: {exc}") from exc
    if isinstance(converted, Path):
        try:
            converted = converted.read_text(encoding="utf-8")
        except OSError as exc:
            raise NormalizationError(f"MinerU Markdown output cannot be read: {converted}") from exc
    if isinstance(converted, bytes):
        converted = converted.decode("utf-8-sig", errors="replace")
    if not isinstance(converted, str) or not converted.strip():
        raise NormalizationError("MinerU converter returned empty Markdown")
    return parse_text(converted)


def parse_pdf_bytes(
    data: bytes,
    *,
    source_path: str | Path | None = None,
    mineru_converter: MinerUConverter | None = None,
) -> ParsedDocument:
    """Compatibility spelling that still routes PDF bytes through MinerU.

    ``data`` is intentionally not parsed directly. When no source path is
    supplied, a temporary PDF is created solely to give the MinerU adapter a
    path-based input.
    """

    if source_path is not None:
        return _parse_pdf_with_mineru(Path(source_path), mineru_converter)
    with tempfile.TemporaryDirectory(prefix=".a7-mineru-bytes-") as temporary:
        temporary_path = Path(temporary) / "input.pdf"
        temporary_path.write_bytes(bytes(data))
        return _parse_pdf_with_mineru(temporary_path, mineru_converter)


def parse_pdf(
    data: bytes,
    *,
    mineru_converter: MinerUConverter | None = None,
) -> ParsedDocument:
    """Compatibility alias for byte-oriented PDF normalization via MinerU."""

    return parse_pdf_bytes(data, mineru_converter=mineru_converter)


def normalize_mineru_markdown(
    source_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    output_dir: str | Path = NORMALIZED_ROOT,
) -> dict[str, Any]:
    """Normalize Markdown emitted by the existing MinerU conversion path."""

    return normalize_artifact(
        source_path, metadata, output_dir=output_dir, source_format="md"
    )

def _format_from_path(path: Path) -> str:
    suffix = path.suffix.casefold().lstrip(".")
    if suffix == "htm":
        suffix = "html"
    return suffix


def _load_sidecar(path: Path) -> dict[str, Any]:
    sidecar = path.parent / "source.json"
    if not sidecar.is_file():
        return {}
    try:
        value = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise NormalizationError(f"invalid source.json next to {path}: {exc}") from exc
    return value if isinstance(value, dict) else {}


def _infer_pmid(path: Path, metadata: Mapping[str, Any]) -> str | None:
    pmid = _normalize_pmid(_mapping_value(metadata, "pmid"))
    if pmid:
        return pmid
    for value in (path.stem, path.parent.name):
        match = re.search(r"PMID[_-]?(\d+)", value, flags=re.I)
        if match:
            return match.group(1)
    return None


def _front_scalar(key: str, value: Any) -> str:
    if value in (None, ""):
        return '""'
    if key in {"pmid", "year"}:
        return str(value)
    return json.dumps(str(value), ensure_ascii=False)


def _front_matter(metadata: Mapping[str, Any], content_sha256: str) -> str:
    values = dict(metadata)
    values["content_sha256"] = content_sha256
    lines = ["---"]
    for key in FRONT_MATTER_FIELDS:
        lines.append(f"{key}: {_front_scalar(key, values.get(key))}")
    lines.extend(["---", ""])
    return "\n".join(lines)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def normalize_artifact(
    source_path: str | Path,
    metadata: Mapping[str, Any] | None = None,
    *,
    output_dir: str | Path = NORMALIZED_ROOT,
    source_format: str | None = None,
    mineru_converter: MinerUConverter | None = None,
    raw_bytes: bytes | None = None,
    expected_content_sha256: str | None = None,
) -> dict[str, Any]:
    """Normalize one raw artifact and write its PMID-named Markdown output.

    ``raw_bytes`` lets a caller that already read and verified the artifact pass
    exactly those bytes, so nothing between verification and parsing can swap
    the file (A-R1). ``expected_content_sha256`` re-checks the verified content
    version here and, for PDFs, converts the verified bytes rather than the live
    path.
    """

    path = Path(source_path)
    if raw_bytes is None and not path.is_file():
        raise NormalizationError(f"source artifact does not exist: {path}")
    raw = path.read_bytes() if raw_bytes is None else bytes(raw_bytes)
    if expected_content_sha256:
        digest = hashlib.sha256(raw).hexdigest()
        expected = str(expected_content_sha256).casefold()
        if digest != expected:
            raise NormalizationError(
                "raw content does not match the validated content version: "
                f"actual={digest} validated={expected}"
            )
    fmt = (source_format or _format_from_path(path)).casefold().lstrip(".")
    if fmt not in SUPPORTED_FORMATS:
        raise NormalizationError(f"unsupported normalization format: {fmt!r}")

    if fmt == "xml":
        parsed = parse_jats_xml(raw)
    elif fmt == "pdf":
        if raw_bytes is None:
            parsed = _parse_pdf_with_mineru(path, mineru_converter)
        else:
            # A-R1: the converter must consume the bytes that were verified,
            # not whatever the path holds now.
            parsed = parse_pdf_bytes(raw, mineru_converter=mineru_converter)
    else:
        parsed = parse_text(raw)

    sidecar = _load_sidecar(path)
    merged = _merge_metadata(sidecar, parsed.metadata, metadata or {})
    pmid = _infer_pmid(path, merged)
    if not pmid:
        raise NormalizationError("PMID is required for normalized output naming")
    merged["pmid"] = pmid
    if not merged.get("title"):
        merged["title"] = parsed.title or "Untitled article"
    if not merged.get("source"):
        merged["source"] = "unknown"
    if not merged.get("license"):
        merged["license"] = "unverified"
    if not merged.get("journal"):
        merged["journal"] = ""

    digest = hashlib.sha256(raw).hexdigest()
    output_path = Path(output_dir) / f"PMID_{pmid}.md"
    document = _front_matter(merged, digest) + parsed.markdown.strip() + "\n"
    _atomic_write_text(output_path, document)
    headings = [
        line[3:].strip()
        for line in parsed.markdown.splitlines()
        if line.startswith("## ")
    ]
    return {
        "pmid": int(pmid),
        "format": fmt,
        "source_path": str(path),
        "output_path": str(output_path),
        "content_sha256": digest,
        "metadata": {key: merged.get(key) for key in FRONT_MATTER_FIELDS},
        "headings": headings,
    }


def normalize_document(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Compatibility spelling for callers that use the generic document name."""

    return normalize_artifact(*args, **kwargs)


def normalize_one(
    pmid: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    raw_root: str | Path = RAW_ROOT,
    normalized_root: str | Path = NORMALIZED_ROOT,
    require_validated: bool = True,
) -> dict[str, Any]:
    """Normalize the A5 artifact for one PMID using canonical task metadata.

    A-R1: the A6 verdict must still describe the artifact that is actually
    parsed. The raw bytes are read once, checked against the validated content
    version, and parsed as those same bytes; a failure raises before any output
    is written, so an existing normalized document is never overwritten.
    """

    migrate_to_v2(db_path)
    conn = _connect(db_path)
    try:
        row = conn.execute("SELECT * FROM tasks WHERE pmid=?", (int(pmid),)).fetchone()
    finally:
        conn.close()
    if row is None:
        raise NormalizationError(f"task PMID {pmid} does not exist")
    task = dict(row)
    if require_validated:
        required = (
            task.get("content_status"),
            task.get("file_valid"),
            task.get("bibliographic_match"),
        )
        if required != ("fulltext_ready", 1, 1):
            raise NormalizationError(
                f"PMID {pmid} is not A6-ready: content_status={required[0]!r}, "
                f"file_valid={required[1]!r}, bibliographic_match={required[2]!r}"
            )
        recorded = content_binding(task)
        if not recorded.bound:
            raise NormalizationError(
                f"PMID {pmid} has no validated content version: {recorded.reason}"
            )
    located = locate_raw_artifact(int(pmid), task, raw_root)
    if located is None:
        raise NormalizationError(f"no raw artifact is available for PMID {pmid}")
    path, fmt = located
    raw = path.read_bytes()
    expected_sha256 = None
    if require_validated:
        digest = hashlib.sha256(raw).hexdigest()
        binding = content_binding(task, actual_sha256=digest)
        if not binding.bound:
            raise NormalizationError(
                f"PMID {pmid} artifact no longer matches the validated content "
                f"version: {binding.reason}"
            )
        expected_sha256 = digest
    return normalize_artifact(
        path,
        task,
        output_dir=normalized_root,
        source_format=fmt,
        raw_bytes=raw,
        expected_content_sha256=expected_sha256,
    )


def run_cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="A7 normalize one full-text artifact")
    parser.add_argument("pmid", type=int)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--raw-root", default=str(RAW_ROOT))
    parser.add_argument("--normalized-root", default=str(NORMALIZED_ROOT))
    parser.add_argument(
        "--allow-unvalidated",
        action="store_true",
        help="allow a fixture or migration task without the A6 fulltext_ready gate",
    )
    args = parser.parse_args(argv)
    result = normalize_one(
        args.pmid,
        db_path=args.db,
        raw_root=args.raw_root,
        normalized_root=args.normalized_root,
        require_validated=not args.allow_unvalidated,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(run_cli())
