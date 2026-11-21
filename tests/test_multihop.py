from rageval.retrieval.multihop import merge_hops, parse_followup, plan_messages


def test_parse_followup():
    assert parse_followup("Shirley Temple") == "Shirley Temple"
    assert parse_followup('Query: "Shirley Temple"\nextra') == "Shirley Temple"
    assert parse_followup("NONE") is None
    assert parse_followup("none.") is None
    assert parse_followup("") is None
    assert parse_followup("word " * 30) is None


def test_merge_hops_splices_after_head_and_dedupes():
    hop1 = ["a", "b", "c", "d", "e"]
    hop2 = ["b", "x", "y", "z"]
    assert merge_hops(hop1, hop2, keep=2, take=2) == ["a", "b", "x", "y", "c", "d", "e"]
    assert merge_hops(hop1, [], keep=2, take=2) == hop1
    assert merge_hops(hop1, hop2, keep=0, take=1)[:2] == ["b", "a"]


def test_plan_messages_numbered():
    m = plan_messages("q?", [("T1", "x"), ("T2", "y")])
    assert "[2] Title: T2" in m[1]["content"] and m[0]["role"] == "system"
