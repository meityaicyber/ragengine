"""Structured answering: questions resolved by querying fields.

No language model is involved. The answer text is templated from query
results, so it is deterministic, reproducible, and traceable to rows that can
be re-run by hand. That is what makes it defensible to the party being
audited (D02).

Every answer carries the predicate it evaluated, so a reader can check the
question was understood the way they meant it.
"""

from __future__ import annotations

from collections.abc import Sequence

from ragaudit.domain.models import Answer, Citation, Finding
from ragaudit.retrieve.router import QueryPlan
from ragaudit.store.catalog import Catalog


class StructuredAnswerer:
    def __init__(self, catalog: Catalog) -> None:
        self.catalog = catalog

    def answer(
        self, question: str, plan: QueryPlan, *, report_ids: Sequence[str] | None = None
    ) -> Answer:
        findings = self.catalog.findings(
            report_ids=report_ids,
            severities=plan.severities or None,
            statuses=plan.statuses or None,
            cwe=plan.cwe,
            cve=plan.cve,
        )
        predicate = _describe_predicate(plan, report_ids)

        if plan.wants_count:
            text = self._count_text(findings, predicate, plan, report_ids)
        elif plan.wants_list:
            text = self._list_text(findings, predicate)
        elif plan.universal:
            text = self._universal_text(findings, plan, predicate, report_ids)
        else:
            text = self._presence_text(findings, predicate, plan)

        answer = Answer(
            text=text,
            citations=[_citation(f) for f in findings[:20]],
            route="structured",
            abstained=False,
        )
        answer.review_flags.append(f"predicate: {predicate}")
        if not findings and not plan.negated:
            answer.review_flags.append("no rows matched; verify the question was parsed as intended")
        return answer

    # -- renderers -------------------------------------------------------

    def _count_text(
        self,
        findings: list[Finding],
        predicate: str,
        plan: QueryPlan,
        report_ids: Sequence[str] | None,
    ) -> str:
        total = len(findings)
        lines = [f"{total} finding{'s' if total != 1 else ''} match {predicate}."]
        if total:
            breakdown: dict[str, int] = {}
            for f in findings:
                breakdown[f.severity.value] = breakdown.get(f.severity.value, 0) + 1
            lines.append("By severity: " + ", ".join(f"{k} {v}" for k, v in breakdown.items()) + ".")
        scope = self.catalog.severity_breakdown(report_ids)
        if scope:
            lines.append(
                "Across the scope queried: "
                + ", ".join(f"{k} {v}" for k, v in scope.items())
                + "."
            )
        return " ".join(lines)

    def _list_text(self, findings: list[Finding], predicate: str) -> str:
        if not findings:
            return f"No findings match {predicate}."
        head = [
            f"- [{f.finding_id}] {f.title or '(untitled)'} - {f.severity.value}, {f.status.value}"
            for f in findings[:25]
        ]
        more = f"\n... and {len(findings) - 25} more." if len(findings) > 25 else ""
        return f"{len(findings)} finding(s) match {predicate}:\n" + "\n".join(head) + more

    def _universal_text(
        self,
        findings: list[Finding],
        plan: QueryPlan,
        predicate: str,
        report_ids: Sequence[str] | None,
    ) -> str:
        """Answer a universal claim by looking for counterexamples.

        "Are all findings closed?" is answered by querying for findings that
        are *not* closed. One counterexample settles it, and the complete
        scan is what makes the negative answer trustworthy.
        """
        if plan.statuses:
            target = plan.statuses
            everything = self.catalog.findings(
                report_ids=report_ids,
                severities=plan.severities or None,
                cwe=plan.cwe,
                cve=plan.cve,
            )
            counter = [f for f in everything if f.status.value not in target]
            if not everything:
                return f"There are no findings in scope, so the claim holds vacuously ({predicate})."
            if not counter:
                return (
                    f"Yes. All {len(everything)} finding(s) in scope have status "
                    f"{' or '.join(target)}."
                )
            sample = ", ".join(f"{f.finding_id} ({f.status.value})" for f in counter[:5])
            return (
                f"No. {len(counter)} of {len(everything)} finding(s) do not have status "
                f"{' or '.join(target)}: {sample}"
                + ("..." if len(counter) > 5 else "")
            )
        return self._presence_text(findings, predicate, plan)

    def _presence_text(self, findings: list[Finding], predicate: str, plan: QueryPlan) -> str:
        total = len(findings)
        if plan.negated:
            if total == 0:
                return f"Correct, there are no findings matching {predicate}."
            return (
                f"No. {total} finding(s) match {predicate}, for example "
                f"{findings[0].finding_id} ({findings[0].severity.value})."
            )
        if total == 0:
            return f"No findings match {predicate}."
        example = findings[0]
        return (
            f"Yes. {total} finding(s) match {predicate}. For example "
            f"[{example.finding_id}] {example.title or '(untitled)'} "
            f"({example.severity.value}, {example.status.value})."
        )


def _describe_predicate(plan: QueryPlan, report_ids: Sequence[str] | None) -> str:
    bits = []
    if plan.severities:
        bits.append("severity in {" + ", ".join(plan.severities) + "}")
    if plan.statuses:
        bits.append("status in {" + ", ".join(plan.statuses) + "}")
    if plan.cwe:
        bits.append(f"CWE matching {plan.cwe}")
    if plan.cve:
        bits.append(f"CVE matching {plan.cve}")
    if not bits:
        bits.append("any finding")
    scope = f" within report(s) {', '.join(report_ids)}" if report_ids else " across all reports"
    return " and ".join(bits) + scope


def _citation(finding: Finding) -> Citation:
    return Citation(
        chunk_id="",
        report_id=finding.report_id,
        field_path=f"findings[{finding.finding_id}]",
        quote=f"{finding.title} - {finding.severity.value}, {finding.status.value}",
        start_char=0,
        end_char=0,
        finding_id=finding.finding_id,
    )
