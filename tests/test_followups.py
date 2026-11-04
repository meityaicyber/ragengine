"""Regressions from real runs with qwen3:8b on two machines.

Each test scripts the exact tool calls the model made in the run that went
wrong, so the fix is checked against the failure as it actually happened:

- "Which findings cite CWE-284?" -> count_findings -> a bare count.
- "list them" -> the agent never saw what "them" was, called list_reports,
  and abstained.
- "tell me in detail about medium level findings" -> the findings were found
  but their text never fetched: one machine abstained ("no details
  available"), the other wrote an incomplete, uncited answer that passed.
- An uncited claim after a find_report call was accepted as an answer.
"""

from __future__ import annotations

import json
import re

from conftest import make_app

from ragaudit.generate.llm import Completion, ToolCall

MEDIUM = {("SAMPLE-WEB-001", "OBS-03"), ("SAMPLE-WEB-001", "OBS-04"), ("SAMPLE-API-002", "V-3")}


class ScriptedLLM:
    """Replays fixed tool-call steps per question, then finalizes.

    ``steps`` is the script for one question: a list of steps, each a list of
    (tool, args) calls. ``per_question`` instead gives one such script per
    question asked, in order. ``finalize`` is "cite" (quote the first evidence
    passage, like a model that does its job) or a fixed answer dict. Every
    prompt is recorded.
    """

    model_name = "scripted"
    prompt_budget_tokens = 12000

    def __init__(self, steps=None, finalize="cite", rewrite=None, per_question=None, visuals=(),
                 stop="WRITE"):
        self.visuals = list(visuals)
        self.stop = stop                  # the model's SHOW / WRITE decision when it stops
        self.scripts = [list(s) for s in per_question] if per_question else [list(steps or [])]
        self.finalize = finalize
        self.rewrite = rewrite
        self.user_messages: list[str] = []
        self.histories: list[list[dict]] = []
        self.finalize_prompts: list[str] = []
        self.turns: list = []
        self._i = 0

    def health(self):
        return True

    def close(self):
        pass

    def generate(self, prompt, *, system=None, format_schema=None, max_tokens=None, timeout_s=None):
        m = re.search(r"Follow-up question: (.+)", prompt)
        text = self.rewrite or (m.group(1) if m else prompt[:200])
        return Completion(text=text, latency_s=0.0, model=self.model_name)

    def chat(self, messages, *, format_schema=None, tools=None, max_tokens=None, timeout_s=None):
        if format_schema:
            prompt = messages[-1]["content"]
            self.finalize_prompts.append(prompt)
            if self.finalize == "cite":
                evidence = _evidence(prompt)
                if not evidence:
                    payload = {"evidence_review": "none", "answer": "The report does not say.",
                               "citations": [], "answered": False, "missing_information": "none"}
                else:
                    cid, line = evidence[0]
                    quote = " ".join(line.split()[:8])
                    payload = {"evidence_review": "ok", "answer": f"The report says: {quote}",
                               "citations": [{"chunk_id": cid, "quote": quote}], "answered": True,
                               "describes_report_text": True, "visuals": self.visuals}
            else:
                payload = self.finalize
            return Completion(text=json.dumps(payload), latency_s=0.0, model=self.model_name)

        if messages[-1]["role"] == "user":
            # A new question starts: load its script.
            self.user_messages.append(messages[-1]["content"])
            self.histories.append([m for m in messages[1:-1]])
            q = len(self.user_messages) - 1
            self.turns = self.scripts[q] if q < len(self.scripts) else []
            self._i = 0
        if self._i >= len(self.turns):
            stop = self.stop[len(self.user_messages) - 1] if isinstance(self.stop, list) else self.stop
            return Completion(text=stop, latency_s=0.0, model=self.model_name)
        calls = self.turns[self._i]
        self._i += 1
        return Completion(text="", latency_s=0.0, model=self.model_name,
                          tool_calls=[ToolCall(n, a) for n, a in calls])


def _evidence(prompt: str) -> list[tuple[str, str]]:
    """(chunk_id, first quotable line) for each passage in a finalize prompt."""
    out = []
    for cid, body in re.findall(r"\[chunk_id=(\w+) \|[^\n]*\]\n(.*?)(?=\n\n---\n\n|\Z)", prompt, re.S):
        lines = [ln for ln in body.splitlines() if ln.strip() and not ln.startswith("(context:")]
        if lines:
            out.append((cid, lines[0]))
    return out


def _app(tmp_path, corpus, llm):
    app = make_app(tmp_path, llm=llm)
    app.build_ingest().ingest_paths(sorted(corpus.glob("*.json")))
    return app


# --------------------------------------------------------------------------
# 1. "Which ..." answered with a count
# --------------------------------------------------------------------------


def test_count_names_the_findings_whatever_the_wording(tmp_path, corpus):
    """The model answered "Which findings cite CWE-284?" with count_findings.
    A small count now carries the findings, so the answer names them, with no
    reading of the question's wording."""
    for question in ("Which findings cite CWE-284?", "How many findings cite CWE-284?"):
        llm = ScriptedLLM([[("count_findings", {"cwe": "CWE-284"})]], stop="SHOW")
        app = _app(tmp_path / question[:5], corpus, llm)
        answer, _ = app.build_chat().ask(question)
        assert answer.text.startswith("2 findings match CWE-284:"), answer.text
        assert "OBS-01" in answer.text and "V-4" in answer.text
        assert "CWE CWE" not in answer.text
        assert not llm.finalize_prompts                # still no extra model call
        app.close()


# --------------------------------------------------------------------------
# 2. Follow-ups reach the agent with their context
# --------------------------------------------------------------------------


def test_followup_history_carries_previous_tool_filters(tmp_path, corpus):
    llm = ScriptedLLM(per_question=[[[("count_findings", {"cwe": "CWE-284"})]],
                                    [[("query_findings", {"cwe": "CWE-284"})]]], stop="SHOW")
    app = _app(tmp_path, corpus, llm)
    chat = app.build_chat()
    _, session = chat.ask("How many findings cite CWE-284?")
    answer, _ = chat.ask("list them", session_id=session.session_id)

    history = " ".join(m["content"] for m in llm.histories[-1] if m["role"] == "assistant")
    assert 'count_findings {"cwe": "CWE-284"}' in history, history
    assert "V-4" in answer.text and "OBS-01" in answer.text
    app.close()


def test_agent_receives_the_rewritten_followup(tmp_path, corpus):
    llm = ScriptedLLM(per_question=[[[("count_findings", {"cwe": "CWE-284"})]],
                                    [[("query_findings", {"cwe": "CWE-284"})]]],
                      rewrite="List the findings that cite CWE-284")
    app = _app(tmp_path, corpus, llm)
    chat = app.build_chat()
    _, session = chat.ask("How many findings cite CWE-284?")
    chat.ask("list them", session_id=session.session_id)

    last = llm.user_messages[-1]
    assert last.startswith("List the findings that cite CWE-284"), last
    assert '"list them"' in last
    assert "cwe=['CWE-284']" in last          # pre-resolution ran on the rewrite
    app.close()


# --------------------------------------------------------------------------
# 3. "Tell me in detail" gets the findings' text, cited
# --------------------------------------------------------------------------


def _finalize_passages(app, llm) -> set[tuple[str, str]]:
    """(report_id, finding_id) of every passage the finalize call was shown."""
    ids = re.findall(r"\[chunk_id=(\w+) \|", llm.finalize_prompts[-1])
    rows = app.catalog.get_chunks(ids)
    return {(r["report_id"], r["finding_id"]) for r in rows.values()}


def test_one_query_then_write_gives_a_written_cited_answer(tmp_path, corpus):
    """Live qwen3:8b run: 'tell me in detail about medium level findings' ->
    one query_findings. It used to be answered with the bare list because only
    one tool ran. The model said WRITE, so the answer is written from the
    findings' text, with citations."""
    llm = ScriptedLLM([[("query_findings", {"severity": "Medium", "report_id": "all"})]], stop="WRITE")
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("tell me in detail about medium level findings")
    assert llm.finalize_prompts, "the shortcut answered with the bare list"
    assert MEDIUM <= _finalize_passages(app, llm)
    assert answer.citations and not answer.abstained
    app.close()


def test_one_query_then_show_is_the_list_with_no_extra_call(tmp_path, corpus):
    llm = ScriptedLLM([[("query_findings", {"severity": "Medium"})]], stop="SHOW")
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("list the medium findings")
    assert not llm.finalize_prompts
    assert answer.text.startswith("3 findings matching severity Medium")
    app.close()


def test_detail_request_after_query_cites_finding_text(tmp_path, corpus):
    """The colleague's run: count, aggregate, query -> previously abstained."""
    llm = ScriptedLLM([
        [("count_findings", {"severity": "Medium"})],
        [("aggregate_findings", {"by": "status", "then_by": "severity", "report_id": "all"})],
        [("query_findings", {"severity": "Medium", "limit": 100})],
    ])
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("tell me in detail about medium level findings")

    assert MEDIUM <= _finalize_passages(app, llm), llm.finalize_prompts[-1][:2000]
    assert not answer.abstained, answer.text
    assert answer.citations and all(c.chunk_id for c in answer.citations)
    # The status-by-severity chart answered nothing that was asked.
    assert not any(b["kind"] == "chart" for b in answer.blocks)
    app.close()


def test_answer_with_no_passages_shown_restates_exact_results(tmp_path, corpus):
    """Only aggregates ran, so the model saw no report text: whatever it wrote
    comes from the exact results, and is not refused for lacking a citation."""
    llm = ScriptedLLM(
        [[("aggregate_findings", {"by": "severity", "then_by": "status"})],
         [("aggregate_findings", {"by": "status"})]],
        finalize={"evidence_review": "x", "answer": "There are 3 medium findings: 1 Open, 2 In Progress.",
                  "citations": [], "answered": True, "describes_report_text": True, "visuals": []},
    )
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("tell me in detail about medium level findings")
    assert not answer.abstained, answer.text
    app.close()


def test_uncited_detail_answer_abstains_but_keeps_the_exact_list(tmp_path, corpus):
    llm = ScriptedLLM(
        [[("query_findings", {"severity": "Medium"})], [("aggregate_findings", {"by": "status"})]],
        finalize={"evidence_review": "x", "answer": "There is 1 medium finding, in SAMPLE-WEB-001.",
                  "citations": [], "answered": True, "describes_report_text": True, "visuals": []},
    )
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("tell me in detail about medium level findings")
    assert answer.abstained and answer.abstention_reason == "no_verified_citation"
    assert "There is 1 medium finding" not in answer.text
    for _, fid in MEDIUM:
        assert fid in answer.text                      # the true list survives
    app.close()


def test_queried_findings_are_citable_whatever_the_wording(tmp_path, corpus):
    """No detail keyword at all: the findings the model queried still get
    their text into the final prompt, so the answer can cite them."""
    llm = ScriptedLLM([[("query_findings", {"severity": "Medium"})],
                       [("aggregate_findings", {"by": "status"})]])
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("what's happening with the medium ones")
    assert MEDIUM <= _finalize_passages(app, llm)
    assert answer.citations
    app.close()


def test_no_corpus_pull_without_a_query_or_detail_request(tmp_path, corpus):
    """Findings the model never looked up are never pulled in."""
    llm = ScriptedLLM([[("aggregate_findings", {"by": "severity"})],
                       [("aggregate_findings", {"by": "status"})]])
    app = _app(tmp_path, corpus, llm)
    app.build_chat().ask("How do the medium findings compare with the rest?")
    assert "queried finding(s) included" not in llm.finalize_prompts[-1]
    app.close()


def test_requested_breakdown_keeps_its_chart(tmp_path, corpus):
    llm = ScriptedLLM([[("aggregate_findings", {"by": "severity"})],
                       [("search_reports", {"query": "severity"})]], visuals=[0])
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("Chart the severity breakdown and explain it")
    assert any(b["kind"] == "chart" for b in answer.blocks)
    assert "C. AVAILABLE VISUALS" in llm.finalize_prompts[-1]
    app.close()


# --------------------------------------------------------------------------
# 4. Looking a report up does not excuse an uncited claim
# --------------------------------------------------------------------------


def test_orientation_tool_does_not_excuse_an_uncited_claim(tmp_path, corpus):
    llm = ScriptedLLM(
        [[("find_report", {"query": "web portal"})], [("search_reports", {"query": "encryption"})]],
        finalize={"evidence_review": "x", "answer": "The portal uses military-grade quantum encryption.",
                  "citations": [], "answered": True},
    )
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("What encryption does the web portal use?")
    assert answer.abstained, answer.text
    assert "quantum" not in answer.text
    assert "agent_forced_abstention" in answer.review_flags
    app.close()


def test_uncited_detail_keeps_the_exact_count(tmp_path, corpus):
    """'How many ... and what are they about?' answered with only an uncited
    count: the 'about' part is refused, the exact count is still given."""
    llm = ScriptedLLM(
        [[("count_findings", {"severity": "High"})], [("search_reports", {"query": "high"})]],
        finalize={"evidence_review": "x", "answer": "There are several high findings.",
                  "citations": [], "answered": True},
    )
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("How many high findings are there and what are they about?")
    assert answer.abstained
    assert "findings match severity High" in answer.text, answer.text
    assert "several" not in answer.text
    app.close()


def test_exact_results_still_answer_without_citation(tmp_path, corpus):
    """A count-backed statement needs no chunk citation: the number is exact."""
    llm = ScriptedLLM(
        [[("count_findings", {"severity": "High"})], [("count_findings", {"status": "Open"})]],
        finalize={"evidence_review": "x", "answer": "4 findings are High; 4 are Open.",
                  "citations": [], "answered": True, "describes_report_text": False, "visuals": []},
    )
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("How many findings are high, and how many are open?")
    assert not answer.abstained, answer.text
    app.close()


# --------------------------------------------------------------------------
# 5. Scope: a named report is a default the model may widen; the UI choice is not
# --------------------------------------------------------------------------


def test_model_cannot_widen_the_ui_selected_scope(tmp_path, corpus):
    llm = ScriptedLLM([[("query_findings", {"report_id": "all"})],
                       [("count_findings", {"report_ids": ["SAMPLE-API-002"]})]])
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("list everything", report_ids=["SAMPLE-WEB-001"])
    listed = [s for s in answer.steps if s["tool"] == "query_findings"][0]
    assert listed["summary"].endswith("of 5"), listed          # SAMPLE-WEB-001 only
    counted = [s for s in answer.steps if s["tool"] == "count_findings"][0]
    assert counted["summary"] == "count_findings: count=5"     # API-002 refused, WEB-001 kept
    app.close()


def test_model_can_widen_a_report_named_in_the_question(tmp_path, corpus):
    llm = ScriptedLLM([[("count_findings", {"report_id": "all"})],
                       [("count_findings", {})]])
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("How does the Kestrel report compare with the others?")
    widened, default = [s["summary"] for s in answer.steps]
    assert widened == "count_findings: count=9"                # all indexed reports
    assert default == "count_findings: count=5"                # the named report
    assert "default report scope=['SAMPLE-WEB-001']" in llm.user_messages[0]
    app.close()


def test_written_answer_from_list_reports_is_not_refused(tmp_path, corpus):
    """Live: 'How many audit reports are indexed?' -> list_reports, model said
    WRITE, answer was refused because list_reports counted as orientation."""
    llm = ScriptedLLM(
        [[("list_reports", {})]], stop="WRITE",
        finalize={"evidence_review": "x", "answer": "3 reports are indexed.", "citations": [],
                  "answered": True, "describes_report_text": True, "visuals": []},
    )
    app = _app(tmp_path, corpus, llm)
    answer, _ = app.build_chat().ask("How many audit reports are indexed?")
    assert not answer.abstained, answer.text
    assert answer.text == "3 reports are indexed."
    app.close()
