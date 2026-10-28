# Evaluation

The 20 cases in `dataset.json` refer to the three fictional engineering documents in `data/`.
They test retrieval, citations, selected facts, refusal, and optional model-graded faithfulness.
The fixtures describe a fictional Atlas service. Its targets are not StackLens measurements.

Run migrations, ingest the sample documents into the default CLI collection, then run:

```sh
stacklens eval --dataset evals/dataset.json --limit 20 --max-cost 1 --min-score 0.8
```

`--max-cost` is an estimated post-response stop condition, not a provider-enforced spending cap.
Fact matching and numeric groundedness are heuristics; they do not establish semantic correctness.
The model judge can make mistakes. Review failures and quotes manually. Local reports stay ignored;
only publish a report after reviewing it for document text and private information.
No old benchmark results or performance claims are included.
