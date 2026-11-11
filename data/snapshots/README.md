These JSONL files are outputs of the offline pipeline in this repository.

They are not LLM judge scores. Each line is one labelled question, the reply
the deterministic generator produced, and the chunks the retriever returned.
`improved.jsonl` is the run the offline CI gate scores. `bad-retriever.jsonl`
and `no-guardrail.jsonl` are the broken variants the tests expect to fail
that gate.

No OpenAI or Anthropic key was available when these files were written, so
there is no file of faithfulness, answer relevancy, context precision, or
context recall scores.
