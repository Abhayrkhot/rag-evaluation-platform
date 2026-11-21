import numpy as np

from rageval.generation.citations import parse_answer
from rageval.generation.verify import verify_answer

DOCS = [("A", "Alpha"), ("B", "Beta"), ("C", "Noise")]


def prob(premises, hyps):
    # entailed iff every '+'-separated token of the hypothesis (minus the final '.') is in the premise
    return np.array(
        [
            1.0 if all(t in p for t in h.rstrip(".").split("+")) else 0.0
            for p, h in zip(premises, hyps, strict=True)
        ]
    )


def test_drops_unnecessary_citation():
    p = parse_answer("Alpha [1][3].")
    out, st = verify_answer(p, DOCS, prob)
    assert out.statements[0].citations == [1]
    assert st.dropped_citations == 1


def test_recites_wrong_citation():
    p = parse_answer("Beta [3].")
    out, st = verify_answer(p, DOCS, prob)
    assert out.statements[0].citations == [2] and st.recited == 1


def test_pair_support_for_multi_hop():
    p = parse_answer("Alpha+Beta [3].")
    out, _ = verify_answer(p, DOCS, prob)
    assert out.statements[0].citations == [1, 2]


def test_unsupported_repair_keeps_prune_removes():
    p = parse_answer("Alpha [1]. Gamma [2].")
    out, st = verify_answer(p, DOCS, prob, policy="repair")
    assert len(out.claims) == 2 and st.unsupported == 1
    out, st = verify_answer(p, DOCS, prob, policy="prune")
    assert [s.text for s in out.claims] == ["Alpha."] and st.pruned == 1


def test_all_pruned_becomes_abstention():
    out, st = verify_answer(parse_answer("Gamma [1]."), DOCS, prob, policy="prune")
    assert out.abstained and st.became_abstention
