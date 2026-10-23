"""History-aware query rewriting.

This is the single highest-leverage addition in multi-turn RAG, and the reason is
worth stating precisely.

Retrieval is stateless. When a user asks "what about the second one?", that string
is what gets embedded and searched. It contains no nouns, no domain vocabulary,
nothing to match against - so it retrieves noise, and the answer is wrong no matter
how good the generation step is. The LLM never gets a chance.

The fix is to resolve the follow-up against the conversation *before* retrieval:

    history: "How many vacation days do I get?" -> "Twenty per year..."
    follow-up: "can I carry them over?"
    rewritten: "Can unused vacation days be carried over to the next year?"

That rewritten string has vocabulary a vector search can actually work with.

Two guardrails matter here:

1. **Skip the call entirely when there is no history.** A first question is already
   standalone. Rewriting it costs latency and tokens and can only make it worse.
2. **Fall back to the original on anything suspicious.** A rewrite that comes back
   empty, or many times longer than the input, means the model answered the question
   instead of rewriting it. The original question is always a safe fallback.
"""

from __future__ import annotations

import logging

from langchain_core.prompts import ChatPromptTemplate

from app.prompts import text_of

logger = logging.getLogger(__name__)

# Turns of history to consider. Enough to resolve a reference, few enough that the
# rewriter doesn't drift toward summarising the whole conversation.
MAX_HISTORY_TURNS = 6

REWRITE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You rewrite follow-up questions into standalone ones for a document "
            "search engine.\n\n"
            "Given the conversation so far and the user's latest message, output a "
            "single self-contained question that preserves the user's intent and "
            "resolves every pronoun and implicit reference using the conversation.\n\n"
            "Rules:\n"
            "- Output ONLY the rewritten question. No preamble, no explanation, no "
            "quotes around it.\n"
            "- Do NOT answer the question. You are rewriting, not answering.\n"
            "- Keep the user's own terminology and any specific names, codes or "
            "identifiers exactly as written.\n"
            "- If the latest message is already standalone, output it unchanged.\n"
            "- Never invent detail that is not present in the conversation.",
        ),
        (
            "human",
            "CONVERSATION SO FAR:\n{history}\n\n"
            "LATEST USER MESSAGE: {question}\n\n"
            "Standalone question:",
        ),
    ]
)


def format_history(history: list[dict], max_turns: int = MAX_HISTORY_TURNS) -> str:
    """Render recent turns as plain text for the rewriter.

    Assistant turns are truncated: the rewriter needs enough of the previous answer
    to resolve a reference, not the whole thing.
    """
    recent = history[-max_turns:]
    lines = []
    for turn in recent:
        role = turn.get("role", "user")
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        if role == "assistant":
            content = content[:500] + ("..." if len(content) > 500 else "")
        lines.append(f"{'User' if role == 'user' else 'Assistant'}: {content}")
    return "\n".join(lines)


def rewrite_question(question: str, history: list[dict] | None, llm) -> str:
    """Resolve `question` against `history` into a standalone question."""
    if not history:
        return question

    formatted = format_history(history)
    if not formatted:
        return question

    try:
        response = (REWRITE_PROMPT | llm).invoke(
            {"history": formatted, "question": question}
        )
        rewritten = text_of(response).strip().strip('"')
    except Exception as exc:  # noqa: BLE001 - a failed rewrite must never fail the query
        logger.warning("query rewrite failed (%s); using original question", exc)
        return question

    if not rewritten:
        logger.warning("rewrite came back empty; using original question")
        return question

    # A rewrite far longer than the input means the model answered rather than
    # rewrote. Original question is always safe.
    if len(rewritten) > max(400, len(question) * 6):
        logger.warning("rewrite looks like an answer, not a question; using original")
        return question

    if rewritten != question:
        logger.info("rewrote %r -> %r", question, rewritten)
    return rewritten
