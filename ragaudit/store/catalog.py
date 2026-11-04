"""Structured catalogue: SQLite over normalised findings.

Why this exists (D02). Most audit questions are predicates over fields the
extractor already recovered: "are all critical findings closed?", "which
reports cite CWE-284?", "how many high findings are open?". Answered by SQL
these are exact, instant, and explainable to the party being audited. Answered
by top-k retrieval they are unsound by construction -- a count over a sample of
k chunks cannot be trusted, because retrieval never promises it saw every row.

Retrieval is reserved for the narrative fields, which are genuinely prose.

Postgres is the production target; the DDL here is deliberately portable and
the access layer is narrow enough to swap. SQLite is used so the system runs
air-gapped with no server, which is also what the tests use.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from ragaudit.domain.models import (
    Finding,
    FindingStatus,
    Report,
    ReportMeta,
    Severity,
)

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS reports (
    report_id          TEXT PRIMARY KEY,
    title              TEXT NOT NULL DEFAULT '',
    report_type        TEXT NOT NULL DEFAULT '',
    audit_type         TEXT NOT NULL DEFAULT '',
    organisation       TEXT NOT NULL DEFAULT '',
    auditor            TEXT NOT NULL DEFAULT '',
    classification     TEXT NOT NULL DEFAULT '',
    release_date       TEXT NOT NULL DEFAULT '',
    audit_period_from  TEXT NOT NULL DEFAULT '',
    audit_period_to    TEXT NOT NULL DEFAULT '',
    standards_json     TEXT NOT NULL DEFAULT '[]',
    source_file        TEXT NOT NULL DEFAULT '',
    source_sha256      TEXT NOT NULL DEFAULT '',
    extractor_version  TEXT NOT NULL DEFAULT '',
    ingest_path        TEXT NOT NULL DEFAULT '',
    ingested_at        TEXT NOT NULL DEFAULT '',
    ingest_status      TEXT NOT NULL DEFAULT 'pending',
    validation_json    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS findings (
    finding_key      TEXT PRIMARY KEY,
    report_id        TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    finding_id       TEXT NOT NULL,
    title            TEXT NOT NULL DEFAULT '',
    severity         TEXT NOT NULL DEFAULT 'Unknown',
    severity_rank    INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL DEFAULT 'Unknown',
    cvss_score       REAL,
    cvss_vector      TEXT NOT NULL DEFAULT '',
    is_repeat        INTEGER,
    observation      TEXT NOT NULL DEFAULT '',
    impact           TEXT NOT NULL DEFAULT '',
    recommendation   TEXT NOT NULL DEFAULT '',
    preconditions    TEXT NOT NULL DEFAULT '',
    raw_json         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_findings_report   ON findings(report_id);
CREATE INDEX IF NOT EXISTS idx_findings_severity ON findings(severity_rank);
CREATE INDEX IF NOT EXISTS idx_findings_status   ON findings(status);

-- Multi-valued attributes get their own tables so they stay queryable.
-- Storing them as JSON strings would push "which reports cite CWE-284" back
-- into a LIKE scan, which is the thing this layer exists to avoid.
CREATE TABLE IF NOT EXISTS finding_identifiers (
    finding_key TEXT NOT NULL REFERENCES findings(finding_key) ON DELETE CASCADE,
    kind        TEXT NOT NULL,          -- 'cwe' | 'cve'
    value       TEXT NOT NULL,
    PRIMARY KEY (finding_key, kind, value)
);
CREATE INDEX IF NOT EXISTS idx_ident_value ON finding_identifiers(kind, value);

CREATE TABLE IF NOT EXISTS finding_assets (
    finding_key TEXT NOT NULL REFERENCES findings(finding_key) ON DELETE CASCADE,
    asset       TEXT NOT NULL,
    PRIMARY KEY (finding_key, asset)
);
CREATE INDEX IF NOT EXISTS idx_assets_asset ON finding_assets(asset);

CREATE TABLE IF NOT EXISTS figures (
    token       TEXT NOT NULL,
    report_id   TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    finding_id  TEXT NOT NULL DEFAULT '',
    page        INTEGER NOT NULL DEFAULT 0,
    idx         INTEGER NOT NULL DEFAULT 0,
    asset_path  TEXT,
    PRIMARY KEY (report_id, token)
);
CREATE INDEX IF NOT EXISTS idx_figures_finding ON figures(report_id, finding_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id     TEXT PRIMARY KEY,
    report_id    TEXT NOT NULL REFERENCES reports(report_id) ON DELETE CASCADE,
    finding_id   TEXT,
    section      TEXT NOT NULL DEFAULT '',
    field_path   TEXT NOT NULL DEFAULT '',
    text         TEXT NOT NULL,
    context_prefix TEXT NOT NULL DEFAULT '',
    start_char   INTEGER NOT NULL DEFAULT 0,
    end_char     INTEGER NOT NULL DEFAULT 0,
    token_count  INTEGER NOT NULL DEFAULT 0,
    figures_json TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_chunks_report ON chunks(report_id);
CREATE INDEX IF NOT EXISTS idx_chunks_finding ON chunks(finding_id);

-- Lexical half of hybrid retrieval. FTS5 is used in place of a hand-rolled
-- BM25 so the lexical index is incremental and does not need rebuilding from
-- the full corpus on every ingest (D05).
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    chunk_id UNINDEXED,
    report_id UNINDEXED,
    tokenize = "unicode61 remove_diacritics 2"
);
"""

SEVERITY_RANK = {
    Severity.CRITICAL: 5,
    Severity.HIGH: 4,
    Severity.MEDIUM: 3,
    Severity.LOW: 2,
    Severity.INFORMATIONAL: 1,
    Severity.UNKNOWN: 0,
}


def finding_key(report_id: str, finding_id: str) -> str:
    return f"{report_id}::{finding_id}"


class Catalog:
    """Narrow data-access layer over the structured store."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Bring a catalogue created by an older version up to this schema.

        ``CREATE TABLE IF NOT EXISTS`` never adds columns to an existing
        table, so new columns are added here explicitly.
        """
        columns = {r["name"] for r in self._conn.execute("PRAGMA table_info(reports)")}
        if "ingest_path" not in columns:
            self._conn.execute("ALTER TABLE reports ADD COLUMN ingest_path TEXT NOT NULL DEFAULT ''")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Catalog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # -- writes ----------------------------------------------------------

    def upsert_report(self, report: Report, *, status: str, validation: dict[str, Any]) -> None:
        """Replace a report and everything derived from it, atomically.

        Delete-then-insert rather than merge: a re-ingest must not leave
        findings from a previous extraction behind, since those would still be
        retrievable and would be cited as current evidence.
        """
        meta = report.meta
        with self.transaction() as conn:
            conn.execute("DELETE FROM reports WHERE report_id = ?", (meta.report_id,))
            conn.execute(
                "DELETE FROM chunks_fts WHERE report_id = ?",
                (meta.report_id,),
            )
            conn.execute(
                """
                INSERT INTO reports (
                    report_id, title, report_type, audit_type, organisation, auditor,
                    classification, release_date, audit_period_from, audit_period_to,
                    standards_json, source_file, source_sha256, extractor_version,
                    ingest_path, ingested_at, ingest_status, validation_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    meta.report_id,
                    meta.title,
                    meta.report_type,
                    meta.audit_type,
                    meta.organisation,
                    meta.auditor,
                    meta.classification,
                    meta.release_date,
                    meta.audit_period_from,
                    meta.audit_period_to,
                    json.dumps(meta.standards),
                    meta.source_file,
                    meta.source_sha256,
                    meta.extractor_version,
                    meta.ingest_path,
                    meta.ingested_at,
                    status,
                    json.dumps(validation),
                ),
            )

            for finding in report.findings:
                key = finding_key(meta.report_id, finding.finding_id)
                conn.execute(
                    """
                    INSERT INTO findings (
                        finding_key, report_id, finding_id, title, severity, severity_rank,
                        status, cvss_score, cvss_vector, is_repeat,
                        observation, impact, recommendation, preconditions, raw_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        key,
                        meta.report_id,
                        finding.finding_id,
                        finding.title,
                        finding.severity.value,
                        SEVERITY_RANK.get(finding.severity, 0),
                        finding.status.value,
                        finding.cvss_score,
                        finding.cvss_vector,
                        None if finding.is_repeat is None else int(finding.is_repeat),
                        finding.observation,
                        finding.impact,
                        finding.recommendation,
                        finding.preconditions,
                        json.dumps(finding.raw, default=str),
                    ),
                )
                rows = [(key, "cwe", v) for v in finding.cwe_ids]
                rows += [(key, "cve", v) for v in finding.cve_ids]
                if rows:
                    conn.executemany(
                        "INSERT OR IGNORE INTO finding_identifiers (finding_key, kind, value) VALUES (?,?,?)",
                        rows,
                    )
                if finding.affected_assets:
                    conn.executemany(
                        "INSERT OR IGNORE INTO finding_assets (finding_key, asset) VALUES (?,?)",
                        [(key, a) for a in finding.affected_assets],
                    )

            for figure in report.figures:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO figures (token, report_id, finding_id, page, idx, asset_path)
                    VALUES (?,?,?,?,?,?)
                    """,
                    (
                        figure.token,
                        meta.report_id,
                        _figure_owner(report, figure.token),
                        figure.page,
                        figure.index,
                        figure.asset_path,
                    ),
                )

            for chunk in report.chunks:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO chunks (
                        chunk_id, report_id, finding_id, section, field_path, text,
                        context_prefix, start_char, end_char, token_count, figures_json
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.report_id,
                        chunk.finding_id,
                        chunk.section,
                        chunk.field_path,
                        chunk.text,
                        chunk.context_prefix,
                        chunk.start_char,
                        chunk.end_char,
                        chunk.token_count,
                        json.dumps(chunk.figure_tokens),
                    ),
                )
                conn.execute(
                    "INSERT INTO chunks_fts (text, chunk_id, report_id) VALUES (?,?,?)",
                    (chunk.embedding_text, chunk.chunk_id, chunk.report_id),
                )

    def delete_report(self, report_id: str) -> None:
        with self.transaction() as conn:
            conn.execute("DELETE FROM chunks_fts WHERE report_id = ?", (report_id,))
            conn.execute("DELETE FROM reports WHERE report_id = ?", (report_id,))

    # -- reads -----------------------------------------------------------

    def report_ids(self, *, status: str | None = None) -> list[str]:
        sql = "SELECT report_id FROM reports"
        params: list[Any] = []
        if status:
            sql += " WHERE ingest_status = ?"
            params.append(status)
        sql += " ORDER BY report_id"
        return [r["report_id"] for r in self._conn.execute(sql, params)]

    def get_report_meta(self, report_id: str) -> ReportMeta | None:
        row = self._conn.execute("SELECT * FROM reports WHERE report_id = ?", (report_id,)).fetchone()
        if row is None:
            return None
        return ReportMeta(
            report_id=row["report_id"],
            title=row["title"],
            report_type=row["report_type"],
            audit_type=row["audit_type"],
            organisation=row["organisation"],
            auditor=row["auditor"],
            classification=row["classification"],
            release_date=row["release_date"],
            audit_period_from=row["audit_period_from"],
            audit_period_to=row["audit_period_to"],
            standards=json.loads(row["standards_json"]),
            source_file=row["source_file"],
            source_sha256=row["source_sha256"],
            extractor_version=row["extractor_version"],
            ingest_path=row["ingest_path"],
            ingested_at=row["ingested_at"],
        )

    def source_hash(self, report_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT source_sha256 FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        return row["source_sha256"] if row else None

    def ingest_path(self, report_id: str) -> str | None:
        """Absolute path the report was ingested from, or None if unknown."""
        row = self._conn.execute(
            "SELECT ingest_path FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        return row["ingest_path"] if row else None

    def report_id_for_path(self, path: str) -> str | None:
        """Report id previously ingested from this exact file, if any."""
        row = self._conn.execute(
            "SELECT report_id FROM reports WHERE ingest_path = ? AND ingest_path != ''", (path,)
        ).fetchone()
        return row["report_id"] if row else None

    def chunk_ids(self, report_id: str) -> list[str]:
        return [
            r["chunk_id"]
            for r in self._conn.execute(
                "SELECT chunk_id FROM chunks WHERE report_id = ? ORDER BY chunk_id", (report_id,)
            )
        ]

    def chunks_for_finding(
        self, finding_id: str, report_ids: Sequence[str] | None = None
    ) -> list[sqlite3.Row]:
        """Chunks of one finding id (optionally scoped), via SQL not a full scan."""
        sql = "SELECT * FROM chunks WHERE finding_id = ?"
        params: list[Any] = [finding_id]
        if report_ids:
            sql += f" AND report_id IN ({','.join('?' * len(report_ids))})"
            params.extend(report_ids)
        return list(self._conn.execute(sql + " ORDER BY report_id, chunk_id", params))

    def findings_by_id(self, finding_id: str, report_ids: Sequence[str] | None = None) -> list[Finding]:
        """Findings with this exact id (case-insensitive), optionally scoped."""
        sql = "SELECT f.* FROM findings f WHERE UPPER(f.finding_id) = UPPER(?)"
        params: list[Any] = [finding_id]
        if report_ids:
            sql += f" AND f.report_id IN ({','.join('?' * len(report_ids))})"
            params.extend(report_ids)
        found = [self._row_to_finding(r) for r in self._conn.execute(sql, params)]
        self._attach_multivalued(found)
        return found

    def reports_overview(self) -> list[dict[str, Any]]:
        """One row per report with status and finding count, in one query."""
        rows = self._conn.execute(
            "SELECT r.report_id, r.title, r.classification, r.report_type, r.ingest_status, "
            "r.release_date, (SELECT COUNT(*) FROM findings f WHERE f.report_id = r.report_id) AS findings "
            "FROM reports r ORDER BY r.report_id"
        )
        return [dict(r) for r in rows]

    def report_titles(self) -> dict[str, str]:
        """report_id -> title for every report, in one query."""
        return {r["report_id"]: r["title"] for r in self._conn.execute("SELECT report_id, title FROM reports")}

    def distinct_finding_ids(self) -> set[str]:
        return {r[0] for r in self._conn.execute("SELECT DISTINCT finding_id FROM findings")}

    def validation_record(self, report_id: str) -> dict[str, Any]:
        row = self._conn.execute(
            "SELECT validation_json FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        try:
            return json.loads(row["validation_json"]) if row and row["validation_json"] else {}
        except ValueError:
            return {}

    def report_status(self, report_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT ingest_status FROM reports WHERE report_id = ?", (report_id,)
        ).fetchone()
        return row["ingest_status"] if row else None

    def findings(
        self,
        *,
        report_ids: Sequence[str] | None = None,
        severities: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
        min_severity_rank: int | None = None,
        cwe: str | None = None,
        cve: str | None = None,
        asset_like: str | None = None,
        limit: int | None = None,
    ) -> list[Finding]:
        """Filtered finding query. This is the exact-answer path."""
        joins, where, params = self._filter_sql(
            report_ids=report_ids, severities=severities, statuses=statuses,
            min_severity_rank=min_severity_rank, cwe=cwe, cve=cve, asset_like=asset_like,
        )
        sql = ["SELECT f.* FROM findings f", *joins]
        if where:
            sql.append("WHERE " + " AND ".join(where))
        sql.append("GROUP BY f.finding_key ORDER BY f.severity_rank DESC, f.finding_id")
        if limit:
            sql.append("LIMIT ?")
            params.append(limit)

        rows = self._conn.execute(" ".join(sql), params).fetchall()
        findings = [self._row_to_finding(r) for r in rows]
        self._attach_multivalued(findings)
        return findings

    @staticmethod
    def _filter_sql(
        *,
        report_ids: Sequence[str] | None = None,
        severities: Sequence[str] | None = None,
        statuses: Sequence[str] | None = None,
        min_severity_rank: int | None = None,
        cwe: str | None = None,
        cve: str | None = None,
        asset_like: str | None = None,
    ) -> tuple[list[str], list[str], list[Any]]:
        """Shared JOIN/WHERE builder so list and count can never disagree."""
        sql: list[str] = []
        where: list[str] = []
        params: list[Any] = []

        identifiers = []
        if cwe:
            identifiers.append(("cwe", _ident_pattern(cwe, "CWE")))
        if cve:
            identifiers.append(("cve", _ident_pattern(cve, "CVE")))
        # One join per identifier. Previously, when both a CWE and a CVE were
        # given, the CVE filter was silently dropped.
        for n, (kind, pattern) in enumerate(identifiers):
            alias = f"fi{n}"
            sql.append(f"JOIN finding_identifiers {alias} ON {alias}.finding_key = f.finding_key")
            # Whole-identifier match. The old LIKE '%CWE-28%' also matched
            # CWE-284 and CWE-287, inflating exact counts.
            where.append(f"{alias}.kind = ? AND (UPPER({alias}.value) || ' ') GLOB ?")
            params.extend([kind, pattern])
        if asset_like:
            sql.append("JOIN finding_assets fa ON fa.finding_key = f.finding_key")
            where.append("fa.asset LIKE ?")
            params.append(f"%{asset_like}%")
        if report_ids:
            where.append(f"f.report_id IN ({','.join('?' * len(report_ids))})")
            params.extend(report_ids)
        if severities:
            where.append(f"f.severity IN ({','.join('?' * len(severities))})")
            params.extend(severities)
        if statuses:
            where.append(f"f.status IN ({','.join('?' * len(statuses))})")
            params.extend(statuses)
        if min_severity_rank is not None:
            where.append("f.severity_rank >= ?")
            params.append(min_severity_rank)
        return sql, where, params

    def _attach_multivalued(self, findings: list[Finding]) -> None:
        """Populate CWE/CVE/asset lists in one round trip.

        These live in their own tables so they stay indexable; re-attaching
        them here keeps the Finding object whole for callers.
        """
        if not findings:
            return
        keys = [finding_key(f.report_id, f.finding_id) for f in findings]
        by_key = {k: f for k, f in zip(keys, findings)}
        placeholders = ",".join("?" * len(keys))

        for row in self._conn.execute(
            f"SELECT finding_key, kind, value FROM finding_identifiers WHERE finding_key IN ({placeholders})",
            keys,
        ):
            target = by_key.get(row["finding_key"])
            if target is None:
                continue
            (target.cwe_ids if row["kind"] == "cwe" else target.cve_ids).append(row["value"])

        for row in self._conn.execute(
            f"SELECT finding_key, asset FROM finding_assets WHERE finding_key IN ({placeholders})",
            keys,
        ):
            target = by_key.get(row["finding_key"])
            if target is not None:
                target.affected_assets.append(row["asset"])

    def count_findings(self, **kwargs: Any) -> int:
        """Exact count in SQL. Previously loaded every matching row to len() it."""
        kwargs.pop("limit", None)
        joins, where, params = self._filter_sql(**kwargs)
        sql = ["SELECT COUNT(DISTINCT f.finding_key) FROM findings f", *joins]
        if where:
            sql.append("WHERE " + " AND ".join(where))
        return int(self._conn.execute(" ".join(sql), params).fetchone()[0])

    def severity_breakdown(self, report_ids: Sequence[str] | None = None) -> dict[str, int]:
        sql = "SELECT severity, COUNT(*) AS n FROM findings"
        params: list[Any] = []
        if report_ids:
            sql += f" WHERE report_id IN ({','.join('?' * len(report_ids))})"
            params.extend(report_ids)
        sql += " GROUP BY severity ORDER BY COUNT(*) DESC"
        return {r["severity"]: r["n"] for r in self._conn.execute(sql, params)}

    def status_breakdown(self, report_ids: Sequence[str] | None = None) -> dict[str, int]:
        sql = "SELECT status, COUNT(*) AS n FROM findings"
        params: list[Any] = []
        if report_ids:
            sql += f" WHERE report_id IN ({','.join('?' * len(report_ids))})"
            params.extend(report_ids)
        sql += " GROUP BY status ORDER BY COUNT(*) DESC"
        return {r["status"]: r["n"] for r in self._conn.execute(sql, params)}

    def figures_for_finding(self, report_id: str, finding_id: str) -> list[sqlite3.Row]:
        return list(
            self._conn.execute(
                "SELECT * FROM figures WHERE report_id = ? AND finding_id = ? ORDER BY page, idx",
                (report_id, finding_id),
            )
        )

    def figures_for_tokens(self, report_id: str, tokens: Sequence[str]) -> list[sqlite3.Row]:
        if not tokens:
            return []
        placeholders = ",".join("?" * len(tokens))
        return list(
            self._conn.execute(
                f"SELECT * FROM figures WHERE report_id = ? AND token IN ({placeholders}) ORDER BY page, idx",
                [report_id, *tokens],
            )
        )

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, sqlite3.Row]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})", list(chunk_ids)
        )
        return {r["chunk_id"]: r for r in rows}

    def all_chunks(self, report_ids: Sequence[str] | None = None) -> list[sqlite3.Row]:
        sql = "SELECT * FROM chunks"
        params: list[Any] = []
        if report_ids:
            sql += f" WHERE report_id IN ({','.join('?' * len(report_ids))})"
            params.extend(report_ids)
        sql += " ORDER BY chunk_id"
        return list(self._conn.execute(sql, params))

    def search_lexical(
        self, query: str, *, report_ids: Sequence[str] | None = None, limit: int = 60
    ) -> list[tuple[str, float]]:
        """BM25 over the FTS5 index. Returns (chunk_id, score), best first.

        The report filter is applied inside the query, not after it. Filtering
        afterwards both shrinks the result below the requested depth and -- in
        a multi-tenant corpus -- risks surfacing another organisation's text.
        """
        match = _fts_query(query)
        if not match:
            return []
        sql = "SELECT chunk_id, bm25(chunks_fts) AS score FROM chunks_fts WHERE chunks_fts MATCH ?"
        params: list[Any] = [match]
        if report_ids:
            sql += f" AND report_id IN ({','.join('?' * len(report_ids))})"
            params.extend(report_ids)
        sql += " ORDER BY score LIMIT ?"
        params.append(limit)
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.OperationalError:
            return []
        # bm25() returns a negative score where more negative is better.
        return [(r["chunk_id"], -float(r["score"])) for r in rows]

    def stats(self) -> dict[str, Any]:
        one = lambda sql: self._conn.execute(sql).fetchone()[0]  # noqa: E731
        return {
            "reports": one("SELECT COUNT(*) FROM reports"),
            "reports_indexed": one("SELECT COUNT(*) FROM reports WHERE ingest_status = 'indexed'"),
            "reports_quarantined": one("SELECT COUNT(*) FROM reports WHERE ingest_status = 'quarantined'"),
            "findings": one("SELECT COUNT(*) FROM findings"),
            "chunks": one("SELECT COUNT(*) FROM chunks"),
            "figures": one("SELECT COUNT(*) FROM figures"),
            "figures_available": one("SELECT COUNT(*) FROM figures WHERE asset_path IS NOT NULL"),
        }

    @staticmethod
    def _row_to_finding(row: sqlite3.Row) -> Finding:
        return Finding(
            finding_id=row["finding_id"],
            report_id=row["report_id"],
            title=row["title"],
            severity=Severity.parse(row["severity"]),
            status=FindingStatus.parse(row["status"]),
            cvss_score=row["cvss_score"],
            cvss_vector=row["cvss_vector"],
            observation=row["observation"],
            impact=row["impact"],
            recommendation=row["recommendation"],
            preconditions=row["preconditions"],
            is_repeat=None if row["is_repeat"] is None else bool(row["is_repeat"]),
            raw=json.loads(row["raw_json"]) if row["raw_json"] else {},
        )


def _ident_pattern(value: str, prefix: str) -> str:
    """GLOB pattern matching one CWE/CVE id as a whole token.

    "284" and "cwe-284" both become CWE-284; the trailing [^0-9] stops
    CWE-28 matching CWE-284. Stored values such as "CWE-284: Improper Access
    Control" still match because a space is appended before matching.
    """
    text = "".join(ch for ch in str(value).upper() if ch.isalnum() or ch == "-").strip("-")
    if text.isdigit():
        text = f"{prefix}-{text}"
    return f"*{text}[^0-9]*"


def _figure_owner(report: Report, token: str) -> str:
    for finding in report.findings:
        if token in finding.figure_tokens:
            return finding.finding_id
    return ""


_FTS_SAFE = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_. ")


def _fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Every term is quoted, so punctuation in identifiers such as ``CVE-2024-1234``
    cannot be read as FTS operator syntax.
    """
    cleaned = "".join(ch if ch in _FTS_SAFE else " " for ch in query)
    terms = [t for t in cleaned.split() if len(t) > 1]
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms[:32])
