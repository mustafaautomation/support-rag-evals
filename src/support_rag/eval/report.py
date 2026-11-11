"""Markdown and HTML reports for a baseline-versus-improved comparison."""

from html import escape

from support_rag.eval.gate import GateResult
from support_rag.eval.metrics import MetricReport

OFFLINE_METRICS: tuple[tuple[str, str], ...] = (
    ("non_llm_context_recall", "Non-LLM context recall"),
    ("string_presence", "String presence"),
    ("refusal_correctness", "Refusal correctness"),
    ("retrieval_hit_rate", "Retrieval hit rate"),
    ("mrr", "MRR"),
)


def render_markdown(
    *,
    generated_on: str,
    variants: dict[str, MetricReport],
    gate: GateResult,
    note: str,
) -> str:
    lines = [
        "# Sableboard support RAG evaluation",
        "",
        f"Generated on {generated_on} from the offline pipeline. No LLM judge scores are included.",
        "",
        note,
        "",
        "## Metric averages",
        "",
        _markdown_table(variants, gate),
        "",
        "The improved column is the FAISS retriever with the guardrail prompt. "
        "The bad-retriever column always returns four unrelated help articles. "
        "The no-guardrail column uses the same retriever as improved and the prompt "
        "that tells the model it may ignore the help articles.",
        "",
        "## Improved gate",
        "",
    ]
    if gate.passed:
        lines.append("The improved run passed every offline threshold.")
    else:
        lines.append("The improved run missed at least one offline threshold:")
        lines.extend(f"- {failure}" for failure in gate.failures)
    lines.extend(["", "## Per-question failures", ""])
    for name, report in variants.items():
        lines.append(f"### {name}")
        lines.append("")
        if not report.failures:
            lines.append("No per-question failures.")
        else:
            for item in report.failures:
                reason = "; ".join(item.failures)
                lines.append(f"- `{item.id}`: {reason}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_html(
    *,
    generated_on: str,
    variants: dict[str, MetricReport],
    gate: GateResult,
    note: str,
) -> str:
    status = "passed" if gate.passed else "failed"
    status_class = "pass" if gate.passed else "fail"
    body = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        "<title>Sableboard support RAG evaluation</title>",
        "<style>",
        "body { font-family: Georgia, serif; margin: 2rem auto; max-width: 960px; "
        "color: #1c1c1c; }",
        "table { border-collapse: collapse; width: 100%; }",
        "th, td { border: 1px solid #ccc; padding: 0.4rem 0.6rem; text-align: left; }",
        "th { background: #f4f1ea; }",
        ".pass { color: #0b6b2a; } .fail { color: #8c1d1d; }",
        "code { font-family: ui-monospace, monospace; }",
        "</style>",
        "</head>",
        "<body>",
        "<h1>Sableboard support RAG evaluation</h1>",
        f"<p>Generated on {escape(generated_on)} from the offline pipeline. "
        "No LLM judge scores are included.</p>",
        f"<p>{escape(note)}</p>",
        "<h2>Metric averages</h2>",
        _html_table(variants, gate),
        "<p>The improved column is the FAISS retriever with the guardrail prompt. "
        "The bad-retriever column always returns four unrelated help articles. "
        "The no-guardrail column uses the same retriever as improved and the prompt "
        "that tells the model it may ignore the help articles.</p>",
        f'<h2>Improved gate: <span class="{status_class}">{status}</span></h2>',
    ]
    if gate.failures:
        body.append("<ul>")
        body.extend(f"<li>{escape(failure)}</li>" for failure in gate.failures)
        body.append("</ul>")
    body.append("<h2>Per-question failures</h2>")
    for name, report in variants.items():
        body.append(f"<h3>{escape(name)}</h3>")
        if not report.failures:
            body.append("<p>No per-question failures.</p>")
            continue
        body.append("<ul>")
        for item in report.failures:
            reason = "; ".join(item.failures)
            body.append(f"<li><code>{escape(item.id)}</code>: {escape(reason)}</li>")
        body.append("</ul>")
    body.append("</body></html>\n")
    return "\n".join(body)


def _markdown_table(variants: dict[str, MetricReport], gate: GateResult) -> str:
    header = "| Metric | " + " | ".join(variants) + " | Threshold | Improved |"
    rule = "| --- | " + " | ".join("---" for _ in variants) + " | --- | --- |"
    rows = [header, rule]
    for key, label in OFFLINE_METRICS:
        cells = [f"{report.scores[key]:.3f}" for report in variants.values()]
        limit = gate.thresholds.get(key)
        limit_text = f"{limit:.3f}" if limit is not None else ""
        improved = variants["improved"].scores[key]
        mark = "pass" if limit is not None and improved >= limit else "fail"
        rows.append(f"| {label} | " + " | ".join(cells) + f" | {limit_text} | {mark} |")
    return "\n".join(rows)


def _html_table(variants: dict[str, MetricReport], gate: GateResult) -> str:
    head = "".join(f"<th>{escape(name)}</th>" for name in variants)
    rows = [
        "<table>",
        f"<tr><th>Metric</th>{head}<th>Threshold</th><th>Improved</th></tr>",
    ]
    for key, label in OFFLINE_METRICS:
        cells = "".join(f"<td>{report.scores[key]:.3f}</td>" for report in variants.values())
        limit = gate.thresholds.get(key)
        limit_text = f"{limit:.3f}" if limit is not None else ""
        improved = variants["improved"].scores[key]
        mark = "pass" if limit is not None and improved >= limit else "fail"
        rows.append(
            f"<tr><td>{escape(label)}</td>{cells}<td>{limit_text}</td>"
            f'<td class="{mark}">{mark}</td></tr>'
        )
    rows.append("</table>")
    return "\n".join(rows)
