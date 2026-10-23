"""Tests for query rewriting guardrails.

No API key needed. The rewriter's failure modes matter more than its happy path:
a rewrite that silently mangles the question is worse than no rewrite at all, so
every degenerate case must fall back to the original.
"""

from __future__ import annotations

from app.rewrite import MAX_HISTORY_TURNS, format_history, rewrite_question


class StubLLM:
    """Minimal stand-in. `rewrite_question` only ever calls `(prompt | llm).invoke`,
    so a Runnable is needed - but we can shortcut by patching at the chain level."""

    def __init__(self, reply: str | None = None, raises: bool = False):
        self.reply = reply
        self.raises = raises


def _rewrite_with(monkeypatch, reply=None, raises=False):
    """Patch the prompt-pipe so tests exercise rewrite_question's logic, not LangChain's."""

    class FakeChain:
        def invoke(self, _payload):
            if raises:
                raise RuntimeError("provider exploded")

            class R:
                content = reply

            return R()

    import app.rewrite as rewrite_mod

    class FakePrompt:
        def __or__(self, _other):
            return FakeChain()

    monkeypatch.setattr(rewrite_mod, "REWRITE_PROMPT", FakePrompt())


HISTORY = [
    {"role": "user", "content": "how many vacation days do I get?"},
    {"role": "assistant", "content": "Twenty per year [1]."},
]


def test_no_history_skips_the_llm_entirely():
    """A first question is already standalone - rewriting it costs tokens and can
    only make it worse."""
    sentinel = object()  # would raise if the LLM were touched
    assert rewrite_question("what is the policy?", [], sentinel) == "what is the policy?"
    assert rewrite_question("what is the policy?", None, sentinel) == "what is the policy?"


def test_rewrite_is_used_when_history_exists(monkeypatch):
    _rewrite_with(monkeypatch, reply="Can unused vacation days be carried over?")
    out = rewrite_question("can I carry them over?", HISTORY, None)
    assert out == "Can unused vacation days be carried over?"


def test_empty_rewrite_falls_back(monkeypatch):
    _rewrite_with(monkeypatch, reply="   ")
    assert rewrite_question("can I carry them over?", HISTORY, None) == "can I carry them over?"


def test_llm_failure_falls_back(monkeypatch):
    """A broken rewriter must never break the query."""
    _rewrite_with(monkeypatch, raises=True)
    assert rewrite_question("can I carry them over?", HISTORY, None) == "can I carry them over?"


def test_answer_shaped_rewrite_falls_back(monkeypatch):
    """A rewrite far longer than the input means the model answered instead of
    rewriting. Reject it."""
    _rewrite_with(monkeypatch, reply="Yes. " + "x" * 900)
    assert rewrite_question("can I carry them over?", HISTORY, None) == "can I carry them over?"


def test_surrounding_quotes_are_stripped(monkeypatch):
    _rewrite_with(monkeypatch, reply='"Can unused days be carried over?"')
    assert rewrite_question("carry over?", HISTORY, None) == "Can unused days be carried over?"


def test_format_history_truncates_long_assistant_turns():
    long_turn = [{"role": "assistant", "content": "x" * 5000}]
    formatted = format_history(long_turn)
    assert len(formatted) < 700
    assert formatted.endswith("...")


def test_format_history_caps_turn_count():
    many = [{"role": "user", "content": f"question {i}"} for i in range(50)]
    formatted = format_history(many)
    assert formatted.count("\n") + 1 == MAX_HISTORY_TURNS
    assert "question 49" in formatted
    assert "question 0" not in formatted


def test_format_history_skips_empty_turns():
    formatted = format_history([{"role": "user", "content": "   "},
                                {"role": "user", "content": "real"}])
    assert formatted == "User: real"


def test_rewrite_accepts_gemini3_content_parts():
    """Gemini 3 returns content as a list of parts; a str-only rewriter silently fell back."""
    from langchain_core.messages import AIMessage

    class PartsLLM:
        def invoke(self, *_a, **_k):
            return AIMessage(content=[{"type": "text", "text": "Can leave carry over?"}])

    from langchain_core.runnables import RunnableLambda

    llm = RunnableLambda(lambda _p: PartsLLM().invoke())
    history = [{"role": "user", "content": "how much leave?"}]
    assert rewrite_question("can it carry over?", history, llm) == "Can leave carry over?"
