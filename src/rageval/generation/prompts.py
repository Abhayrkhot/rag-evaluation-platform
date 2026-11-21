"""Prompt strategies for grounded, cited answers.

naive      : the common "answer and cite" baseline (pipeline stage S0/S1)
zero_shot  : explicit citation and grounding rules, no examples
few_shot   : zero_shot rules + three worked examples (bridge, comparison, unanswerable) drawn
             from HotpotQA *train*, so no evaluation question leaks into the prompt
cot        : zero_shot rules + reason step by step in <reasoning>, then answer in <answer>
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rageval.types import Doc

ABSTAIN = "I cannot answer this question based on the provided documents."

RULES = f"""You are a careful research assistant. Answer the question using only the numbered documents provided.
Rules:
1. Write one fact per sentence, and end every sentence with citations of the documents that state it, written as [n] or [n][m].
2. If a sentence combines facts from two documents, cite both. Cite only documents that directly state the facts in that sentence.
3. Use full names instead of pronouns, so each sentence can be checked on its own.
4. Do not add facts that are not stated in the documents, even if you know them.
5. If the documents do not contain enough information to answer, reply exactly: "{ABSTAIN}"
6. Keep the answer brief: one to three sentences."""

# v1 rules (dev iteration 1, see docs/RESULTS_LOG.md): no one-fact-per-sentence, multi-document or
# pronoun rules; two-hop sentences often cited only one of the two documents they relied on.

REVISE_INSTRUCTIONS = """Your previous answer is below. The sentences listed after it are not supported by the documents they cite.
Rewrite the answer so that every sentence is stated by the documents it cites, following the rules. Split combined facts into separate sentences, cite every document a sentence relies on, and drop anything the documents do not state. If the documents do not contain the answer, reply exactly: "{abstain}"

Previous answer:
{answer}

Unsupported sentences:
{unsupported}"""

NAIVE_SYSTEM = "You are a helpful assistant. Answer the question based on the documents below and cite the documents you use with [n]."

COT_SUFFIX = """
Before answering, reason step by step inside <reasoning></reasoning>: identify which documents state the relevant facts and how those facts connect to answer the question. Then give the final answer inside <answer></answer>, following the rules above."""


@dataclass(frozen=True)
class Exemplar:
    question: str
    docs: tuple[tuple[str, str], ...]  # (title, text)
    answer: str


FEW_SHOT: tuple[Exemplar, ...] = (
    Exemplar(
        question="What French region was the 1968 Winter Olympics held in?",
        docs=(
            (
                "1968 Alpine Skiing World Cup – Men's Slalom",
                "Note: Round 3/Race 9 was the Olympic event, which counts also for the World Cup. See also 1968 Winter Olympics and Alpine skiing at the 1968 Winter Olympics",
            ),
            (
                "Grenoble",
                "Grenoble is a city in southeastern France, at the foot of the French Alps where the river Drac joins the Isère. Located in the Auvergne-Rhône-Alpes region, Grenoble is the capital of the department of Isère and is an important European scientific centre.",
            ),
            (
                "Iran at the 1968 Winter Olympics",
                "Iran had four competitors at the 1968 Winter Olympics in Grenoble, France. All took part in the men's Alpine Skiing events, with the highest finish being 66th place by Lotfollah Kiashemshaki in the Downhill.",
            ),
        ),
        answer="The 1968 Winter Olympics were held in Grenoble, France [3]. Grenoble is located in the Auvergne-Rhône-Alpes region [2].",
    ),
    Exemplar(
        question="Which astronaut was born first, Christer Fuglesang or Philippe Perrin?",
        docs=(
            (
                "Christer Fuglesang",
                "Arne Christer Fuglesang (born March 18, 1957 in Stockholm) is a Swedish physicist and an ESA astronaut. He was first launched aboard the STS-116 Space Shuttle mission on December 10, 2006, making him the first Swedish citizen in space.",
            ),
            (
                "Wale Adebanwi",
                "Professor Wale Adebanwi, (born 1969), is a Nigerian - born first Black African Rhodes Professor at Oxford University",
            ),
            (
                "Philippe Perrin",
                "Philippe Perrin (Colonel, French Air Force) (born January 6, 1963) is a French test pilot and former CNES and European Space Agency astronaut.",
            ),
        ),
        answer="Christer Fuglesang was born on March 18, 1957 [1]. Philippe Perrin was born on January 6, 1963 [3]. Christer Fuglesang was born first [1][3].",
    ),
    Exemplar(
        question="Mimi Lockhart's actress from September 1999 til March 2007 also portrayed what character in ABC's \"One Life to Live\" from October 2007 until January 2012?",
        docs=(
            (
                "Austin Williams",
                'Austin Williams (born November 13, 1996) is an American soap opera actor. In 2005, Austin was cast in the film "The Good Shepherd", which starred Matt Damon, as the young version of Damon\'s character Edward Wilson.',
            ),
            (
                "Mimi Lockhart",
                'Mimi Lockhart (formerly Brady) is a fictional character from "Days of Our Lives". Mimi was originated by Doren Fein from August 17 to 19, 1999. She is most recognized by actress Farah Fath who played the role from September 16, 1999, to March 1, 2007.',
            ),
            (
                "One Life to Live storylines (1990–99)",
                '"One Life to Live" is an American soap opera that has been broadcast on the ABC network since 1968. The series starts with "One Life to Live" storylines (1968–1979).',
            ),
        ),
        answer=ABSTAIN,
    ),
)

STRATEGIES = ("naive", "zero_shot", "few_shot", "cot")


def format_docs(docs: Sequence[tuple[str, str]]) -> str:
    return "\n\n".join(f"[{i}] Title: {t}\n{x}" for i, (t, x) in enumerate(docs, start=1))


def user_turn(question: str, docs: Sequence[tuple[str, str]]) -> str:
    return f"Documents:\n{format_docs(docs)}\n\nQuestion: {question}"


def build_messages(strategy: str, question: str, docs: Sequence[Doc]) -> list[dict[str, str]]:
    pairs = [(d.title, d.text) for d in docs]
    if strategy == "naive":
        return [
            {"role": "system", "content": NAIVE_SYSTEM},
            {"role": "user", "content": user_turn(question, pairs)},
        ]
    if strategy == "zero_shot":
        return [
            {"role": "system", "content": RULES},
            {"role": "user", "content": user_turn(question, pairs)},
        ]
    if strategy == "few_shot":
        msgs = [{"role": "system", "content": RULES}]
        for ex in FEW_SHOT:
            msgs.append({"role": "user", "content": user_turn(ex.question, ex.docs)})
            msgs.append({"role": "assistant", "content": ex.answer})
        msgs.append({"role": "user", "content": user_turn(question, pairs)})
        return msgs
    if strategy == "cot":
        return [
            {"role": "system", "content": RULES + COT_SUFFIX},
            {"role": "user", "content": user_turn(question, pairs)},
        ]
    raise ValueError(f"unknown prompt strategy {strategy}")


def max_tokens_for(strategy: str) -> int:
    return 1024 if strategy == "cot" else 320


def revise_messages(
    question: str, docs: Sequence[Doc], answer: str, unsupported: Sequence[str]
) -> list[dict[str, str]]:
    """Second-pass prompt for the `revise` verification policy."""
    pairs = [(d.title, d.text) for d in docs]
    bullets = "\n".join(f"- {u}" for u in unsupported)
    return [
        {"role": "system", "content": RULES},
        {"role": "user", "content": user_turn(question, pairs)},
        {"role": "assistant", "content": answer},
        {
            "role": "user",
            "content": REVISE_INSTRUCTIONS.format(
                abstain=ABSTAIN, answer=answer, unsupported=bullets
            ),
        },
    ]
