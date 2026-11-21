from rageval.generation.citations import (
    extract_answer,
    is_abstention,
    parse_answer,
    parse_citations,
    render,
    split_sentences,
)
from rageval.generation.prompts import ABSTAIN, FEW_SHOT, build_messages
from rageval.types import Doc


def test_split_keeps_abbreviations_and_initials():
    text = "J. K. Rowling was born in the U.S. Army hospital [1]. Dr. Smith disagreed [2]."
    assert split_sentences(text) == [
        "J. K. Rowling was born in the U.S. Army hospital [1].",
        "Dr. Smith disagreed [2].",
    ]


def test_citations_after_period_attach_to_previous_sentence():
    s = split_sentences("He was born in 1950. [1][2] She was born in 1960. [3]")
    assert s == ["He was born in 1950 [1][2].", "She was born in 1960 [3]."]


def test_parse_citations_variants():
    assert parse_citations("x [1, 3] and [2][1]") == [1, 3, 2]
    assert parse_citations("no cites") == []


def test_parse_answer_statements_and_range_filter():
    p = parse_answer("Paris is in France [1][9]. It is the capital [2].", n_docs=5)
    assert [s.citations for s in p.statements] == [[1], [2]]
    assert p.statements[0].text == "Paris is in France."
    assert not p.abstained
    assert len(p.claims) == 2


def test_abstention_detection():
    assert parse_answer(ABSTAIN).abstained
    assert is_abstention("The documents do not mention his birthplace.")
    partial = parse_answer("The documents do not mention X [1]. Y was born in 1950 [2].")
    assert not partial.abstained
    assert len(partial.claims) == 1


def test_cot_extraction():
    r, a = extract_answer("<reasoning>doc 1 says A</reasoning>\n<answer>A is true [1].</answer>")
    assert r == "doc 1 says A" and a == "A is true [1]."
    r, a = extract_answer("<reasoning>thinking...</reasoning> Answer: B [2].")
    assert a == "B [2]."
    assert extract_answer("Plain answer [1].") == ("", "Plain answer [1].")


def test_render_roundtrip():
    p = parse_answer("A is B [1][2]. C is D.")
    p.statements[0].citations = [2]
    p.statements[1].citations = [1, 3]
    assert render(p.statements) == "A is B [2]. C is D [1][3]."


def test_messages_shapes():
    docs = [Doc("d1", "T", ["Some text."])]
    assert len(build_messages("zero_shot", "q?", docs)) == 2
    assert len(build_messages("few_shot", "q?", docs)) == 2 + 2 * len(FEW_SHOT)
    assert "<answer>" in build_messages("cot", "q?", docs)[0]["content"]
    # few-shot exemplars cite only documents they were given
    for ex in FEW_SHOT:
        assert all(1 <= c <= len(ex.docs) for c in parse_citations(ex.answer))
