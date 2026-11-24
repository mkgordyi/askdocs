"""Quality check for the whole answer flow. Used by people and by the pipeline.

    python -m evals.run              measure and compare with evals/thresholds.json
    python -m evals.run --calibrate  also show which refusal threshold separates best
    python -m evals.run --live       measure with the configured LLM instead of quoting

Exit code 1 means quality dropped below a threshold: the pipeline stops the deploy.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from askdocs.config import settings
from askdocs.generate import ExtractiveGenerator, make_generator
from askdocs.index import Index, load
from askdocs.service import AskService

HERE = Path(__file__).parent


def load_golden() -> tuple[list[dict], list[dict]]:
    rows = [json.loads(line) for line in open(HERE / "golden.jsonl", encoding="utf-8")]
    return [r for r in rows if r["expected"]], [r for r in rows if not r["expected"]]


def retrieval_metrics(index: Index, answerable: list[dict], mode: str) -> dict:
    hit1 = hit5 = rr = 0.0
    misses = []
    for row in answerable:
        articles = [h.chunk.article for h in index.search(row["question"], k=5, mode=mode)]
        rank = next((i for i, a in enumerate(articles, 1) if a in row["expected"]), None)
        hit1 += rank == 1
        hit5 += rank is not None
        rr += 1 / rank if rank else 0
        if rank is None:
            misses.append(row["id"])
    n = len(answerable)
    return {"hit@1": hit1 / n, "hit@5": hit5 / n, "mrr": rr / n, "misses": misses}


def flow_metrics(service: AskService, answerable: list[dict], unanswerable: list[dict]) -> dict:
    """End to end: does the service answer what it should and refuse what it should?"""
    good = [service.ask(r["question"]) for r in answerable]
    bad = [service.ask(r["question"]) for r in unanswerable]
    cited_right = sum(
        a.answered and any(h.chunk.article in r["expected"] for h in a.sources)
        for a, r in zip(good, answerable)
    )
    return {
        "false_refusal_rate": sum(not a.answered for a in good) / len(good),
        "refusal_recall": sum(not a.answered for a in bad) / len(bad),
        "correct_source_rate": cited_right / len(good),
        "wrongly_answered": [r["id"] for a, r in zip(bad, unanswerable) if a.answered],
        "confidence_answerable": [round(a.confidence, 3) for a in good],
        "confidence_unanswerable": [round(a.confidence, 3) for a in bad],
    }


def calibrate(flow: dict) -> list[tuple[float, float, float]]:
    """For each threshold: (threshold, share of good questions refused, share of bad ones refused)."""
    good, bad = flow["confidence_answerable"], flow["confidence_unanswerable"]
    table = []
    for step in range(50, 81):
        t = step / 100
        table.append((t, sum(c < t for c in good) / len(good), sum(c < t for c in bad) / len(bad)))
    return table


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--live", action="store_true", help="use the configured LLM provider")
    args = parser.parse_args()

    index = load()
    answerable, unanswerable = load_golden()
    retrieval = {mode: retrieval_metrics(index, answerable, mode) for mode in ("bm25", "dense", "hybrid")}
    generator = make_generator(settings) if args.live else ExtractiveGenerator()
    flow = flow_metrics(AskService(index, generator), answerable, unanswerable)
    thresholds = json.loads((HERE / "thresholds.json").read_text())

    measured = {
        "hit@5": retrieval["hybrid"]["hit@5"],
        "mrr": retrieval["hybrid"]["mrr"],
        "correct_source_rate": flow["correct_source_rate"],
        "refusal_recall": flow["refusal_recall"],
        "false_refusal_rate": flow["false_refusal_rate"],
    }
    failures = []
    for name, rule in thresholds.items():
        value = measured[name]
        ok = value >= rule["min"] if "min" in rule else value <= rule["max"]
        if not ok:
            failures.append(name)

    lines = [
        "## AskDocs quality report",
        "",
        f"{len(answerable)} questions with an answer in the text, {len(unanswerable)} without. "
        f"Answer mode: {generator.mode}. Refusal threshold: {settings.min_confidence}.",
        "",
        "### Search: is the right article in the top results?",
        "",
        "| search | hit@1 | hit@5 | MRR |",
        "|---|---|---|---|",
    ]
    for mode, m in retrieval.items():
        lines.append(f"| {mode} | {m['hit@1']:.2f} | {m['hit@5']:.2f} | {m['mrr']:.2f} |")
    lines += ["", "### Gate", "", "| metric | value | rule | result |", "|---|---|---|---|"]
    for name, rule in thresholds.items():
        bound = f">= {rule['min']}" if "min" in rule else f"<= {rule['max']}"
        lines.append(f"| {name} | {measured[name]:.2f} | {bound} | {'FAIL' if name in failures else 'ok'} |")
    lines += [
        "",
        f"Not found by hybrid search: {retrieval['hybrid']['misses'] or 'none'}",
        f"Answered although it should refuse: {flow['wrongly_answered'] or 'none'}",
    ]
    if args.calibrate:
        lines += ["", "### Threshold sweep", "", "| threshold | good refused | bad refused |", "|---|---|---|"]
        lines += [f"| {t:.2f} | {g:.2f} | {b:.2f} |" for t, g, b in calibrate(flow)]
    report = "\n".join(lines)
    print(report)

    (HERE / "report.json").write_text(json.dumps({"retrieval": retrieval, "flow": flow, "measured": measured}, indent=2))
    summary = os.getenv("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(report + "\n")
    if failures:
        print(f"\nQuality gate FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
