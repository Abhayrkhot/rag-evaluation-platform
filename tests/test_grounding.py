import pytest

from rageval.eval.grounding import AnswerInput, StatementInput, score_answers

DOCS = [("A", "alpha fact"), ("B", "beta fact"), ("C", "noise")]


def fake_entail(premises, hyps):
    # a hypothesis "x+y" is entailed iff every listed token appears in the premise
    out = []
    for p, h in zip(premises, hyps, strict=True):
        out.append(all(tok in p for tok in h.split("+")))
    return out


def test_single_supported_citation():
    a = AnswerInput("q", DOCS, [StatementInput("alpha", [0])])
    s = score_answers([a], fake_entail)[0]
    assert s.citation_recall == 1.0 and s.citation_precision == 1.0


def test_unsupported_statement_makes_citations_imprecise():
    a = AnswerInput("q", DOCS, [StatementInput("alpha", [2])])
    s = score_answers([a], fake_entail)[0]
    assert s.citation_recall == 0.0 and s.citation_precision == 0.0


def test_unnecessary_extra_citation_is_imprecise():
    # alpha is supported by doc 0 alone; doc 2 is unnecessary
    a = AnswerInput("q", DOCS, [StatementInput("alpha", [0, 2])])
    s = score_answers([a], fake_entail)[0]
    assert s.citation_recall == 1.0
    assert s.citation_precise == [[True, False]]
    assert s.citation_precision == 0.5


def test_multi_hop_needs_both_citations():
    a = AnswerInput("q", DOCS, [StatementInput("alpha+beta", [0, 1])])
    s = score_answers([a], fake_entail)[0]
    assert s.citation_recall == 1.0 and s.citation_precision == 1.0


def test_no_citation_and_invalid_citation():
    a = AnswerInput("q", DOCS, [StatementInput("alpha", []), StatementInput("beta", [1, 7])])
    s = score_answers([a], fake_entail)[0]
    assert s.statement_supported == [False, True]
    # citation 7 is out of range: counted, never precise
    assert s.n_citations == 2
    assert s.citation_precision == pytest.approx(0.5)


def test_judge_calls_are_deduplicated():
    calls = []

    def counting(p, h):
        calls.append(len(p))
        return fake_entail(p, h)

    a = AnswerInput("q", DOCS, [StatementInput("alpha", [0])] * 3)
    score_answers([a, a], counting)
    assert sum(calls) == 1
