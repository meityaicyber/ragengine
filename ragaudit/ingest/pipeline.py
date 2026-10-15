"""Ingest pipeline: parse, validate, normalise, chunk, embed, index.

Separated from serving (D03). The previous harness re-parsed, re-chunked and
re-embedded the whole corpus on every run and wrote a fresh index per run,
which is unworkable at 3,000 reports and left duplicate indexes on disk.

Properties this pipeline guarantees:

* *Incremental.* A report whose source hash is unchanged is skipped.
* *Idempotent.* Re-ingesting replaces a report's rows and vectors rather than
  appending, so chunk ids and counts stay stable.
* *Resumable.* Progress is committed per report, so an interrupted run
  continues where it stopped.
* *Gated.* A report failing a blocking validation check is quarantined, not
  indexed. Answering from corrupt evidence is worse than not answering.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ragaudit.domain.models import FIGURE_TOKEN_RE, FigureRef, Report
from ragaudit.ingest.chunking import FieldAwareChunker
from ragaudit.ingest.normalize import normalize_report
from ragaudit.ingest.validate import ValidationResult, validate_report
from ragaudit.store.catalog import Catalog
from ragaudit.store.vectors import VectorIndex

logger = logging.getLogger("ragaudit.ingest")


@dataclass
class IngestOutcome:
    report_id: str
    status: str                       # indexed | quarantined | skipped | failed
    chunks: int = 0
    findings: int = 0
    figures: int = 0
    figures_missing: int = 0
    blocking: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class IngestReport:
    outcomes: list[IngestOutcome] = field(default_factory=list)

    def count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    def summary(self) -> dict[str, int]:
        return {
            "total": len(self.outcomes),
            "indexed": self.count("indexed"),
            "quarantined": self.count("quarantined"),
            "skipped": self.count("skipped"),
            "failed": self.count("failed"),
        }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def derive_report_id(data: dict, path: Path) -> str:
    """Stable id for a report.

    Prefers an id the extractor supplies; falls back to the filename. The
    corpus uses ``"NA"`` as a placeholder report id, which would collide across
    every report, so placeholders are rejected.
    """
    meta = data.get("report_meta") if isinstance(data.get("report_meta"), dict) else {}
    control = data.get("document_control") if isinstance(data.get("document_control"), dict) else {}
    for candidate in (meta.get("report_id"), control.get("document_id")):
        text = str(candidate or "").strip()
        if text and text.upper() not in {"NA", "N/A", "NONE", "-", "TBD"}:
            return text
    return path.stem.upper()


def collect_figures(report: Report, figure_dir: Path | None) -> list[FigureRef]:
    """Resolve figure placeholder tokens to staged image files.

    A token with no file is kept with ``asset_path=None`` rather than dropped.
    The previous parser deleted these tokens outright, which made the evidence
    look complete when it was not (D07).
    """
    tokens: set[str] = set()
    for finding in report.findings:
        tokens.update(finding.figure_tokens)
    blob = json.dumps(report.raw, default=str)
    tokens.update(f"[IMAGE_PAGE_{p}_FIG_{i}]" for p, i in FIGURE_TOKEN_RE.findall(blob))

    figures: list[FigureRef] = []
    for token in sorted(tokens):
        asset_path = None
        if figure_dir:
            stem = token.strip("[]")
            for suffix in (".png", ".jpg", ".jpeg", ".webp"):
                candidate = figure_dir / report.report_id / f"{stem}{suffix}"
                if candidate.exists():
                    asset_path = str(candidate)
                    break
        ref = FigureRef.from_token(token, report.report_id, asset_path)
        if ref:
            figures.append(ref)
    return figures


class IngestPipeline:
    def __init__(
        self,
        catalog: Catalog,
        index: VectorIndex,
        embedder=None,
        *,
        chunker: FieldAwareChunker | None = None,
        figure_dir: Path | None = None,
        fail_on_validation_error: bool = True,
        contextualizer=None,
    ) -> None:
        self.catalog = catalog
        self.index = index
        self.embedder = embedder
        self.chunker = chunker or FieldAwareChunker()
        self.figure_dir = figure_dir
        self.fail_on_validation_error = fail_on_validation_error
        self.contextualizer = contextualizer

    # -- public ----------------------------------------------------------

    def ingest_paths(self, paths: Iterable[Path], *, force: bool = False) -> IngestReport:
        report = IngestReport()
        for path in paths:
            try:
                report.outcomes.append(self.ingest_file(Path(path), force=force))
            except Exception as exc:  # one bad report must not kill the run
                logger.exception("ingest failed for %s", path)
                report.outcomes.append(
                    IngestOutcome(report_id=Path(path).stem, status="failed", error=str(exc))
                )
        return report

    def ingest_file(self, path: Path, *, force: bool = False) -> IngestOutcome:
        data = json.loads(path.read_text(encoding="utf-8"))
        report_id = derive_report_id(data, path)
        digest = sha256_file(path)

        if not force and self.catalog.source_hash(report_id) == digest:
            logger.info("skipping %s; source unchanged", report_id)
            return IngestOutcome(report_id=report_id, status="skipped")

        validation = validate_report(data, report_id)
        if validation.blocking and self.fail_on_validation_error:
            return self._quarantine(data, report_id, path, validation)

        report = normalize_report(data, report_id, path)
        report.figures = collect_figures(report, self.figure_dir)
        report.chunks = self.chunker.chunk_report(report)

        if self.contextualizer is not None and report.chunks:
            self.contextualizer.annotate(report)

        self._write(report, validation, status="indexed")

        missing = sum(1 for f in report.figures if not f.available)
        return IngestOutcome(
            report_id=report_id,
            status="indexed",
            chunks=len(report.chunks),
            findings=len(report.findings),
            figures=len(report.figures),
            figures_missing=missing,
            blocking=[str(i) for i in validation.blocking],
            warnings=[str(i) for i in validation.warnings],
        )

    # -- internals -------------------------------------------------------

    def _quarantine(
        self, data: dict, report_id: str, path: Path, validation: ValidationResult
    ) -> IngestOutcome:
        """Record the report and its defects without indexing its content.

        Quarantined reports stay visible in the catalogue so an operator can
        see what failed and why, but their chunks never reach the index, so the
        chatbot cannot cite them.
        """
        logger.warning(
            "quarantined %s: %d blocking issue(s)", report_id, len(validation.blocking)
        )
        report = normalize_report(data, report_id, path)
        report.findings = []
        report.chunks = []
        report.figures = []
        self._write(report, validation, status="quarantined")
        return IngestOutcome(
            report_id=report_id,
            status="quarantined",
            blocking=[str(i) for i in validation.blocking],
            warnings=[str(i) for i in validation.warnings],
        )

    def _write(self, report: Report, validation: ValidationResult, *, status: str) -> None:
        payload = {
            "ok": validation.ok,
            "blocking": [str(i) for i in validation.blocking],
            "warnings": [str(i) for i in validation.warnings],
        }
        # Remove the old vectors first: a re-ingest that only appended would
        # leave the previous extraction retrievable and citable.
        self.index.remove_report(report.report_id)
        self.catalog.upsert_report(report, status=status, validation=payload)

        if report.chunks and self.embedder is not None:
            vectors = self.embedder.embed_passages([c.embedding_text for c in report.chunks])
            self.index.add(
                [c.chunk_id for c in report.chunks],
                vectors,
                [report.report_id] * len(report.chunks),
            )

    def save_index(self, directory: Path) -> None:
        self.index.save(directory)


def discover_reports(root: Path, pattern: str = "*.json") -> list[Path]:
    """Find candidate report files, skipping obvious non-reports."""
    skip = {"benchmark", "example_bank", "report_structure", "eval_config", "metrics", "results"}
    out: list[Path] = []
    for path in sorted(Path(root).rglob(pattern)):
        if any(token in path.stem.lower() for token in skip):
            continue
        out.append(path)
    return out


def ingest_directory(
    directory: Path,
    catalog: Catalog,
    index: VectorIndex,
    embedder=None,
    *,
    figure_dir: Path | None = None,
    force: bool = False,
    chunker: FieldAwareChunker | None = None,
    fail_on_validation_error: bool = True,
) -> IngestReport:
    pipeline = IngestPipeline(
        catalog,
        index,
        embedder,
        chunker=chunker,
        figure_dir=figure_dir,
        fail_on_validation_error=fail_on_validation_error,
    )
    return pipeline.ingest_paths(discover_reports(directory), force=force)
