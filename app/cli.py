"""Command line interface.

    python -m app.cli check                    # is my key/model wired up?
    python -m app.cli ingest                   # index everything in data/raw
    python -m app.cli ingest --rebuild         # re-embed from scratch
    python -m app.cli stats                    # what's in the index
    python -m app.cli ask "question"           # one grounded answer
    python -m app.cli chat                     # interactive loop with memory
    python -m app.cli debug "question"         # per-stage retrieval inspector
    python -m app.cli eval                     # golden-set evaluation (Phase 4)
    python -m app.cli serve                    # FastAPI service on :8000
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from app.config import get_settings

app = typer.Typer(add_completion=False, help="Citewise - answers with receipts")
console = Console()


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )


@app.command()
def check() -> None:
    """Verify the API key works and list models your key can actually reach."""
    settings = get_settings()
    if not settings.google_api_key:
        console.print("[red]GOOGLE_API_KEY is not set.[/red]")
        console.print("Copy .env.example to .env and add a key from "
                      "https://aistudio.google.com/apikey")
        raise typer.Exit(1)

    console.print(f"Configured chat model:      [cyan]{settings.chat_model}[/cyan]")
    console.print(f"Configured embedding model: [cyan]{settings.embedding_model}[/cyan] "
                  f"({settings.embedding_dimensions} dims)")

    try:
        from google import genai

        client = genai.Client(api_key=settings.google_api_key)
        available = [m.name.replace("models/", "") for m in client.models.list()]
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Could not list models: {exc}[/red]")
        raise typer.Exit(1)

    gen = sorted(m for m in available if "embedding" not in m)
    emb = sorted(m for m in available if "embedding" in m)

    console.print(f"\n[bold]Embedding models available ({len(emb)}):[/bold]")
    for m in emb:
        mark = " [green]<- configured[/green]" if m == settings.embedding_model else ""
        console.print(f"  {m}{mark}")

    console.print(f"\n[bold]Generation models available ({len(gen)}):[/bold]")
    for m in gen[:25]:
        mark = " [green]<- configured[/green]" if m == settings.chat_model else ""
        console.print(f"  {m}{mark}")
    if len(gen) > 25:
        console.print(f"  ... and {len(gen) - 25} more")

    problems = []
    if settings.chat_model not in available:
        problems.append(f"chat model '{settings.chat_model}' not in your model list")
    if settings.embedding_model not in available:
        problems.append(f"embedding model '{settings.embedding_model}' not in your model list")

    if problems:
        console.print("\n[yellow]Heads up:[/yellow]")
        for p in problems:
            console.print(f"  - {p}")
        console.print("  Update CHAT_MODEL / EMBEDDING_MODEL in .env to one listed above.")
    else:
        console.print("\n[green]Both configured models are available.[/green]")


@app.command()
def ingest(
    path: Path = typer.Option(None, "--path", help="Folder to ingest (default: data/raw)"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Drop the index and re-embed everything"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Load, chunk, embed and index documents. Idempotent: unchanged files are skipped."""
    _setup_logging(verbose)
    from app.ingest import ingest as run_ingest

    settings = get_settings()
    target = path or settings.data_dir
    console.print(f"Ingesting from [cyan]{target}[/cyan]\n")

    with console.status("Working..."):
        report = run_ingest(data_dir=target, rebuild=rebuild,
                            progress=lambda m: console.print(f"  {m}"))

    console.print()
    table = Table(title="Ingestion report", show_header=False)
    table.add_row("Added", str(len(report.added)))
    table.add_row("Updated", str(len(report.updated)))
    table.add_row("Skipped (unchanged)", str(len(report.skipped)))
    table.add_row("Failed", str(len(report.failed)))
    table.add_row("Chunks written", str(report.chunks_written))
    if report.lexical_documents:
        table.add_row("BM25 index (chunks)", str(report.lexical_documents))
    console.print(table)

    for name, reason in report.failed:
        console.print(f"[red]FAILED[/red] {name}: {reason}")

    if not report.added and not report.updated and not report.skipped:
        console.print(
            f"\n[yellow]Nothing to ingest.[/yellow] Put PDFs, DOCX, MD or TXT files "
            f"in {target} and run this again."
        )


@app.command()
def stats() -> None:
    """Show what is currently in the index."""
    from app.vectorstore import collection_stats

    info = collection_stats()
    if not info["chunks"]:
        console.print("[yellow]Index is empty.[/yellow] Run: python -m app.cli ingest")
        return

    table = Table(title=f"{info['chunks']} chunks indexed")
    table.add_column("Source")
    table.add_column("Chunks", justify="right")
    for name, count in info["sources"].items():
        table.add_row(name, str(count))
    console.print(table)

    from app.config import get_settings
    from app.lexical import get_lexical_index

    settings = get_settings()
    index = get_lexical_index()
    lexical_count = len(index) if index else 0
    console.print(
        f"\nRetrieval mode: [cyan]{settings.retrieval_mode}[/cyan] · "
        f"reranker: [cyan]{settings.rerank_backend}[/cyan]"
    )
    if lexical_count:
        console.print(f"BM25 lexical index: [green]{lexical_count} chunks[/green]")
    elif settings.retrieval_mode == "hybrid":
        console.print(
            "[yellow]No BM25 index - hybrid mode is silently running dense-only. "
            "Run `ingest` to build it.[/yellow]"
        )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    k: int = typer.Option(None, "--k", help="Chunks kept after reranking"),
    threshold: float = typer.Option(None, "--threshold", help="Relevance floor (0-1)"),
    mode: str = typer.Option(None, "--mode", help="hybrid | dense | lexical"),
    candidate_k: int = typer.Option(None, "--candidates", help="Candidates per retriever"),
    rerank: str = typer.Option(None, "--rerank", help="llm | cross-encoder | none"),
    source: str = typer.Option(None, "--source", help="Only search this document"),
    show_sources: bool = typer.Option(True, "--sources/--no-sources"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Ask one question and get a grounded, cited answer."""
    _setup_logging(verbose)
    from app.graph import ask as run_ask

    with console.status("Thinking..."):
        result = run_ask(
            question, k=k, threshold=threshold, mode=mode,
            candidate_k=candidate_k, rerank_backend=rerank,
            where={"source": source} if source else None,
        )

    console.print()
    console.print(Panel(Markdown(result["answer"]), title="Answer",
                        border_style="green" if result["grounded"] else "yellow"))

    if show_sources and result["sources"]:
        table = Table(title="Sources", show_lines=False)
        table.add_column("#", justify="right", width=3)
        table.add_column("Document")
        table.add_column("Page", justify="right", width=5)
        table.add_column("Score", justify="right", width=7)
        table.add_column("Found by", width=14)
        for s in result["sources"]:
            table.add_row(
                str(s["n"]), s["source"], str(s["page"]), f"{s['score']:.3f}",
                "+".join(s.get("found_by") or []) or "-",
            )
        console.print(table)

    if verbose:
        console.print(
            f"[dim]mode={result.get('mode')} · retrieved {result['retrieved_count']}, "
            f"kept {result['kept_count']} above threshold[/dim]"
        )


@app.command()
def debug(
    question: str = typer.Argument(..., help="Question to inspect retrieval for"),
    candidate_k: int = typer.Option(10, "--candidates", help="Candidates per retriever"),
    k: int = typer.Option(5, "--k", help="Chunks kept after reranking"),
    rerank: str = typer.Option(None, "--rerank", help="llm | cross-encoder | none"),
    source: str = typer.Option(None, "--source", help="Only search this document"),
) -> None:
    """Show what each retrieval stage does, without generating an answer.

    This is the first thing to run when an answer is wrong. Most bad RAG answers
    are bad retrieval, not bad generation - and this shows you exactly which stage
    lost the passage you expected.
    """
    from app.config import get_settings
    from app.fusion import reciprocal_rank_fusion
    from app.retrieval import dense_candidates, grade, lexical_candidates, retrieve

    settings = get_settings()
    where = {"source": source} if source else None

    console.print(f"[bold]Query:[/bold] {question}")
    console.print(
        f"[dim]mode={settings.retrieval_mode} · candidates={candidate_k}/retriever · "
        f"rerank={rerank or settings.rerank_backend} · k={k}[/dim]\n"
    )

    # Stage 1
    dense = dense_candidates(question, candidate_k, where)
    lexi = lexical_candidates(question, candidate_k, where)

    stage1 = Table(title="Stage 1 - candidates (each retriever independently)")
    stage1.add_column("Rank", justify="right", width=5)
    stage1.add_column("Dense (semantic)")
    stage1.add_column("score", justify="right", width=7)
    stage1.add_column("Lexical (BM25)")
    stage1.add_column("score", justify="right", width=7)
    for i in range(max(len(dense), len(lexi))):
        d = dense[i] if i < len(dense) else None
        l = lexi[i] if i < len(lexi) else None
        stage1.add_row(
            str(i + 1),
            f"{d[0].metadata.get('source','?')} p.{d[0].metadata.get('page','?')}" if d else "-",
            f"{d[1]:.3f}" if d else "",
            f"{l[0].metadata.get('source','?')} p.{l[0].metadata.get('page','?')}" if l else "-",
            f"{l[1]:.2f}" if l else "",
        )
    console.print(stage1)
    if not lexi:
        console.print(
            "[yellow]No lexical results.[/yellow] Either no keyword overlap, or the "
            "BM25 index is missing - run `ingest` to build it.\n"
        )

    # Stage 2
    lists = {}
    if dense:
        lists["dense"] = dense
    if lexi:
        lists["lexical"] = lexi
    fused = reciprocal_rank_fusion(
        lists, k=settings.rrf_k,
        weights={"dense": settings.dense_weight, "lexical": settings.lexical_weight},
    )
    if fused:
        stage2 = Table(title="Stage 2 - reciprocal rank fusion")
        stage2.add_column("Rank", justify="right", width=5)
        stage2.add_column("Chunk")
        stage2.add_column("RRF", justify="right", width=8)
        stage2.add_column("Found by", width=16)
        stage2.add_column("Ranks", width=22)
        for i, r in enumerate(fused[:candidate_k], start=1):
            m = r.document.metadata
            stage2.add_row(
                str(i), f"{m.get('source','?')} p.{m.get('page','?')}",
                f"{r.fused_score:.5f}", "+".join(r.found_by),
                ", ".join(f"{n}#{v}" for n, v in sorted(r.ranks.items())),
            )
        console.print(stage2)

    # Stage 3 + 4
    final = retrieve(question, k=k, candidate_k=candidate_k,
                     rerank_backend=rerank, where=where)
    kept = grade(final, rerank_backend=rerank)
    kept_keys = {id(c.document) for c in kept}

    console.print()
    for i, c in enumerate(final, start=1):
        is_kept = id(c.document) in kept_keys
        moved = ""
        for j, r in enumerate(fused, start=1):
            if r.document.page_content == c.document.page_content:
                delta = j - i
                moved = f" · fusion #{j} {'↑' if delta > 0 else '↓' if delta < 0 else '='}"
                break
        console.print(Panel(
            c.document.page_content.strip()[:600],
            title=f"[{i}] {c.citation} · score {c.score:.3f} · {c.provenance}{moved}"
                  f"{' [green]KEPT[/green]' if is_kept else ' [red]DROPPED[/red]'}",
            border_style="green" if is_kept else "red",
        ))

    console.print(
        f"\n[dim]Stages 3-4: reranked to {len(final)}, {len(kept)} cleared the floor. "
        f"Arrows show movement between fusion and rerank order.[/dim]"
    )


@app.command()
def chat(
    k: int = typer.Option(None, "--k"),
    threshold: float = typer.Option(None, "--threshold"),
    mode: str = typer.Option(None, "--mode", help="hybrid | dense | lexical"),
    rerank: str = typer.Option(None, "--rerank", help="llm | cross-encoder | none"),
    no_memory: bool = typer.Option(False, "--no-memory", help="Treat each question independently"),
) -> None:
    """Interactive loop with conversation memory. Ctrl-C or 'exit' to quit.

    Follow-ups work: prior turns are used to rewrite "what about the second one?"
    into a standalone query before retrieval. Type 'clear' to reset the history.
    """
    from app.graph import ask as run_ask

    console.print("[bold]RAG Knowledge Assistant[/bold]  ('exit' to quit, 'clear' to reset)\n")
    history: list[dict] = []

    while True:
        try:
            question = console.input("[cyan]you >[/cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nBye.")
            return
        if not question:
            continue
        if question.lower() in {"exit", "quit", ":q"}:
            console.print("Bye.")
            return
        if question.lower() == "clear":
            history = []
            console.print("[dim]conversation cleared[/dim]\n")
            continue

        with console.status("Thinking..."):
            result = run_ask(
                question,
                history=None if no_memory else history,
                k=k,
                threshold=threshold,
                mode=mode,
                rerank_backend=rerank,
            )

        console.print()
        if result.get("rewritten"):
            console.print(f"[dim]searched for: {result['search_query']}[/dim]")
        console.print(Markdown(result["answer"]))
        if result["sources"]:
            refs = "  ".join(
                f"[{s['n']}] {s['source']} p.{s['page']}" for s in result["sources"]
            )
            console.print(f"\n[dim]{refs}[/dim]")
        console.print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


@app.command("eval")
def evaluate(
    mode: str = typer.Option(None, "--mode", help="hybrid | dense | lexical"),
    rerank: str = typer.Option(None, "--rerank", help="llm | cross-encoder | none"),
    candidate_k: int = typer.Option(None, "--candidates", help="Candidates per retriever"),
    k: int = typer.Option(None, "--k", help="Chunks kept after reranking"),
    retrieval_only: bool = typer.Option(
        False, "--retrieval-only", help="Skip generation and the judge (cheap)"
    ),
    matrix: bool = typer.Option(
        False, "--matrix", help="Retrieval-only A/B of every mode x reranker"
    ),
    gate: bool = typer.Option(False, "--gate", help="Exit 1 below the success criteria"),
    golden: Path = typer.Option(None, "--golden", help="Golden set (.jsonl)"),
    limit: int = typer.Option(None, "--limit", help="Only the first N questions"),
) -> None:
    """Score retrieval and answers against the golden question set."""
    _setup_logging(False)
    from app import evaluate as ev

    items = ev.load_golden(golden or ev.GOLDEN_PATH)[:limit]
    configs = ev.MATRIX if matrix else [
        ev.EvalConfig(mode=mode, rerank_backend=rerank, candidate_k=candidate_k, k=k,
                      retrieval_only=retrieval_only)
    ]

    reports = []
    for config in configs:
        with console.status(f"Evaluating {config.label}...") as status:
            report = ev.run(config, items, progress=status.update)
        path = ev.save(report)
        reports.append(report)
        console.print(f"[dim]{config.label}: saved {path}[/dim]")

    metrics = ["hit_rate", "mrr", "ndcg", "context_precision", "context_recall",
               "abstention", "false_refusals", "faithfulness", "relevance", "correctness"]
    table = Table(title=f"Evaluation · {len(items)} questions")
    table.add_column("config")
    shown = [m for m in metrics if any(r["summary"].get(m) is not None for r in reports)]
    for m in shown:
        table.add_column(m.replace("_", " "), justify="right")
    table.add_column("tokens", justify="right")
    for r in reports:
        table.add_row(
            r["label"],
            *[f"{r['summary'][m]:.3f}" if r["summary"].get(m) is not None else "-"
              for m in shown],
            str(r["usage"]["total_tokens"]),
        )
    console.print(table)

    if gate:
        failures = [f"{r['label']}: {f}" for r in reports for f in ev.gate_failures(r)]
        for f in failures:
            console.print(f"[red]GATE FAILED[/red] {f}")
        if failures:
            raise typer.Exit(1)
        console.print("[green]All gates passed.[/green]")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
) -> None:
    """Run the FastAPI service. Docs at http://HOST:PORT/docs"""
    import uvicorn

    console.print(f"API docs: [cyan]http://{host}:{port}/docs[/cyan]")
    uvicorn.run("app.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
