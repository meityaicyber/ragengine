"""Ingest pipeline: parse, validate, normalise, chunk, embed, index.

Separated from serving (D03). The previous harness re-parsed, re-chunked and
re-embedded the whole corpus on every run and wrote a fresh index per run,
which is unworkable at 3,000 reports and left duplicate indexes on disk.

Properties this pipeline guarantees:

* *Incremental.* A report whose source hash is unchanged, and whose vectors
  are all present in the index, is skipped.
* *Idempotent.* Re-ingesting replaces a report's rows and vectors rather than
  appending, so chunk ids and counts stay stable.
* *Resumable.* The vector index is checkpointed to disk every few reports and
  again when the run ends (including on Ctrl-C or a crash). A report whose
  catalogue rows exist but whose vectors were lost is detected and re-embedded
  on the next run instead of being silently skipped. (Previously the index was
  saved only once, at the very end, so an interrupted run left reports in SQL
  that semantic search could never find.)
* *Gated.* A report failing a blocking validation check is quarantined, not
  indexed. Answering from corrupt evidence is worse than not answering.
* *Collision-safe ids.* Two different files that derive the same report id
  (same file name in different folders, or the same ``report_id`` reused by an
  extractor) no longer overwrite each other; the second gets a suffixed id.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path

from ragaudit.domain.models import FIGURE_TOKEN_RE, FigureRef, Report
from ragaudit.ingest.chunking import FieldAwareChunker
from ragaudit.ingest.normalize import expand_reports, normalize_report
from ragaudit.ingest.validate import ValidationResult, validate_report
from ragaudit.store.catalog import Catalog
from ragaudit.store.vectors import VectorIndex

logger = logging.getLogger("ragaudit.ingest")

#: Bump when validation, normalisation or chunking changes in a way that can
#: change a report's outcome. Reports ingested under another version are
#: re-processed on the next ingest even if the file is unchanged; otherwise a
#: report quarantined by older, stricter logic would stay quarantined forever.
INGEST_VERSION = 7


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
    path: str = ""
    note: str = ""


@dataclass
class IngestReport:
    outcomes: list[IngestOutcome] = field(default_factory=list)
    interrupted: bool = False

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


def load_json(path: Path):
    """Read a JSON file, tolerating a UTF-8 byte-order mark.

    Files written by Windows tools often start with a BOM, which plain
    ``utf-8`` decoding turns into an "Expecting value" JSON error.
    """
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


#: Placeholder report ids that would collide across every report.
_PLACEHOLDER_IDS = {"NA", "N/A", "NONE", "NULL", "-", "TBD", "0", ""}


def derive_report_id(data: dict, path: Path) -> str:
    """Base id for a report, before collision handling.

    Prefers an id the extractor supplies; falls back to the filename. The
    corpus uses ``"NA"`` as a placeholder report id, which would collide across
    every report, so placeholders are rejected.
    """
    meta = data.get("report_meta") if isinstance(data.get("report_meta"), dict) else {}
    control = data.get("document_control") if isinstance(data.get("document_control"), dict) else {}
    for candidate in (meta.get("report_id"), control.get("document_id")):
        text = str(candidate or "").strip()
        if text.upper() not in _PLACEHOLDER_IDS:
            return text
    return path.stem.upper()


def collect_figures(report: Report, figure_dir: Path | None) -> list[FigureRef]:
    """Resolve figure placeholder tokens to staged image files.

    A token with no file is kept with ``asset_path=None`` rather than dropped.
    The previous parser deleted these tokens outright, which made the evidence
    look complete when it was not (D07).

    Images are looked up at ``<figure_dir>/<report_id>/IMAGE_PAGE_<p>_FIG_<i>.png``
    (also .jpg/.jpeg/.webp).
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
        checkpoint_dir: Path | None = None,
        checkpoint_every: int = 10,
    ) -> None:
        self.catalog = catalog
        self.index = index
        self.embedder = embedder
        self.chunker = chunker or FieldAwareChunker()
        self.figure_dir = figure_dir
        self.fail_on_validation_error = fail_on_validation_error
        self.contextualizer = contextualizer
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every = max(1, checkpoint_every)
        self._claimed: dict[str, str] = {}

    # -- public ----------------------------------------------------------

    def ingest_paths(
        self,
        paths: Iterable[Path],
        *,
        force: bool = False,
        progress: Callable[[int, int, IngestOutcome, float], None] | None = None,
    ) -> IngestReport:
        """Ingest files one by one. Never lets one bad file stop the run.

        The index is checkpointed every ``checkpoint_every`` indexed reports
        and always on exit, including KeyboardInterrupt, so at most the last
        few reports need re-embedding after a hard crash (and the next run
        detects and redoes them automatically).
        """
        paths = [Path(p) for p in paths]
        report = IngestReport()
        self._claimed = {}
        since_checkpoint = 0
        started = time.monotonic()
        try:
            for n, path in enumerate(paths, start=1):
                try:
                    outcomes = self.ingest_file_all(path, force=force)
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # one bad report must not kill the run
                    logger.exception("ingest failed for %s", path)
                    outcomes = [IngestOutcome(
                        report_id=path.stem, status="failed", error=f"{type(exc).__name__}: {exc}",
                        path=str(path),
                    )]
                for outcome in outcomes:
                    report.outcomes.append(outcome)
                    if outcome.status == "indexed":
                        since_checkpoint += 1
                        if since_checkpoint >= self.checkpoint_every:
                            self._checkpoint()
                            since_checkpoint = 0
                    if progress:
                        progress(n, len(paths), outcome, time.monotonic() - started)
        except KeyboardInterrupt:
            report.interrupted = True
            logger.warning("ingest interrupted; saving progress")
        finally:
            self._checkpoint()
        return report

    def ingest_file(self, path: Path, *, force: bool = False) -> IngestOutcome:
        """Ingest one file and return its (first) outcome."""
        return self.ingest_file_all(path, force=force)[0]

    def ingest_file_all(self, path: Path, *, force: bool = False) -> list[IngestOutcome]:
        """Ingest every report a file holds (usually one; see expand_reports)."""
        data = load_json(path)
        reports = expand_reports(data)
        if not reports:
            kind = type(data).__name__
            return [IngestOutcome(
                report_id=path.stem.upper(), status="failed", path=str(path),
                error=(f"top-level JSON is a {kind} that holds neither a report object, a list of "
                       "reports, nor a list of findings"),
            )]
        digest = sha256_file(path)
        return [self._ingest_one(data_i, path, digest, suffix, force=force) for suffix, data_i in reports]

    def _ingest_one(self, data: dict, path: Path, digest: str, suffix: str, *, force: bool) -> IngestOutcome:
        report_id, note = self._resolve_report_id(data, path, digest, suffix)
        # Stored as the report's ingest_path; "#n" distinguishes reports that
        # share one file.
        self._ingest_label = str(path.resolve()) + suffix
        # The same file used to be stored under a different id (its
        # report_id was edited, or id derivation changed). Remove the old
        # copy, or every finding in it would be counted twice.
        previous = self.catalog.report_id_for_path(self._ingest_label)
        if previous and previous != report_id:
            logger.warning("%s was previously ingested as %s; replacing it", path, previous)
            self.index.remove_report(previous)
            self.catalog.delete_report(previous)
            note = (note + "; " if note else "") + f"replaces earlier copy stored as '{previous}'"

        if not force and self._is_current(report_id, digest):
            logger.info("skipping %s; source unchanged and fully indexed", report_id)
            return IngestOutcome(report_id=report_id, status="skipped", path=str(path), note=note)

        validation = validate_report(data, report_id)
        if validation.blocking and self.fail_on_validation_error:
            outcome = self._quarantine(data, report_id, path, validation)
            outcome.note = note
            return outcome

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
            path=str(path) + suffix,
            note=note,
        )

    def save_index(self, directory: Path) -> None:
        self.index.save(directory)

    # -- internals -------------------------------------------------------

    def _checkpoint(self) -> None:
        if self.checkpoint_dir is not None:
            self.index.save(self.checkpoint_dir)

    def _is_current(self, report_id: str, digest: str) -> bool:
        """Unchanged source AND nothing missing from the index.

        The vector check is what makes a crashed run self-heal: a report
        whose SQL rows were committed but whose vectors never reached disk
        has fewer vectors than chunks, so it is re-ingested.
        """
        if self.catalog.source_hash(report_id) != digest:
            return False
        if self.catalog.validation_record(report_id).get("ingest_version") != INGEST_VERSION:
            return False
        if self.embedder is None:
            return True
        if self.catalog.report_status(report_id) != "indexed":
            # Quarantined last time. Still current while the gate is on; with
            # the gate off (--no-quarantine) it must be re-ingested to index it.
            return self.fail_on_validation_error
        expected = len(self.catalog.chunk_ids(report_id))
        return self.index.vector_count(report_id) == expected

    def _resolve_report_id(
        self, data: dict, path: Path, digest: str, suffix: str = ""
    ) -> tuple[str, str]:
        """Pick an id that does not belong to a different file.

        Tries the base id, then ``<id>__<PARENT_FOLDER>``, then a path hash.
        An id is free if nobody owns it, if this exact path owns it, or if the
        owner has identical content (the folder was moved).
        """
        base = derive_report_id(data, path)
        if suffix and base == path.stem.upper():
            # Several reports in one file and no id of their own: number them.
            base = f"{base}-{suffix.lstrip('#')}"
        resolved = str(path.resolve()) + suffix
        path_hash = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:8].upper()
        candidates = [base, f"{base}__{path.parent.name.upper()}", f"{base}__{path_hash}"]

        for candidate in candidates:
            owner = self._claimed.get(candidate)
            if owner is None:
                owner = self.catalog.ingest_path(candidate)
            free = (
                owner is None
                or owner in ("", resolved)
                or (candidate not in self._claimed and self.catalog.source_hash(candidate) == digest)
            )
            if free:
                self._claimed[candidate] = resolved
                if candidate != base:
                    note = (f"report id '{base}' already belongs to another file; "
                            f"stored as '{candidate}'")
                    logger.warning("%s (%s)", note, path)
                    return candidate, note
                return candidate, ""
        # Practically unreachable (path hash is unique per path).
        self._claimed[candidates[-1]] = resolved
        return candidates[-1], "id collision"

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
            path=str(path),
        )

    def _write(self, report: Report, validation: ValidationResult, *, status: str) -> None:
        payload = {
            "ingest_version": INGEST_VERSION,
            "ok": validation.ok,
            "blocking": [str(i) for i in validation.blocking],
            "warnings": [str(i) for i in validation.warnings],
        }
        if getattr(self, "_ingest_label", ""):
            report.meta.ingest_path = self._ingest_label
        # Embed BEFORE touching storage. Embedding is the step most likely to
        # fail (out of memory, model error); doing it first means a failure
        # leaves the previous version of the report fully intact.
        vectors = None
        if report.chunks and self.embedder is not None:
            vectors = self.embedder.embed_passages([c.embedding_text for c in report.chunks])

        # Remove the old vectors first: a re-ingest that only appended would
        # leave the previous extraction retrievable and citable.
        self.index.remove_report(report.report_id)
        self.catalog.upsert_report(report, status=status, validation=payload)

        if vectors is not None:
            self.index.add(
                [c.chunk_id for c in report.chunks],
                vectors,
                [report.report_id] * len(report.chunks),
            )


#: Files that are clearly not reports (eval sets, schemas, our own outputs).
#: Matched against the WHOLE file stem, case-insensitive. Previously any stem
#: merely *containing* "results" or "metrics" was silently dropped, which
#: would have skipped a real report named e.g. "vapt_results_2024.json".
SKIP_STEMS = {
    "benchmark", "benchmark_questions", "example_bank", "report_structure",
    "eval_config", "metrics", "results", "questions", "transcript", "ids",
}
SKIP_DIRS = {".git", ".venv", "venv", "__pycache__", "node_modules", "var", "models"}


def discover_reports(
    root: Path,
    pattern: str = "*.json",
    *,
    include_all: bool = False,
    skipped: list[Path] | None = None,
) -> list[Path]:
    """Find candidate report files under ``root`` (recursive, sorted).

    Skipped files are appended to ``skipped`` when a list is passed, so the
    caller can show them instead of dropping them silently.
    """
    root = Path(root)
    if root.is_file():
        return [root]
    out: list[Path] = []
    for path in sorted(root.rglob(pattern)):
        rel_parts = {p.lower() for p in path.relative_to(root).parts[:-1]}
        if not include_all and (
            path.stem.lower() in SKIP_STEMS or rel_parts & SKIP_DIRS or path.name.startswith(".")
        ):
            if skipped is not None:
                skipped.append(path)
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
