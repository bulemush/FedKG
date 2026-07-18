# CWQ/WebQSP component ablation plan

## Fixed protocol

- Client task and partition: CWQ (`cwq@llm`), three clients, IID split.
- Server-held alignment task: WebQSP (`webquestionssp@llm`).
- All optimization, federation, model, seed, evaluation, and data settings are
  inherited unchanged from
  `cwq_client_iid_webqsp_align_kg_adpt2_dp2.yaml`.
- Every run writes to `checkpoints/ablations/cwq_webqsp/<variant>/` and
  `exp/ablations/cwq_webqsp/<run-tag>/`; it cannot overwrite the original
  paper checkpoint or log directory.

## Single-factor variants

| Variant | Exact intervention | What it tests | Expected if important | Priority |
|---|---|---|---|---:|
| `full` | All four switches enabled | Same-protocol control for fair comparison | Best or tied-best CWQ Hit@1 | 1 |
| `no_hybrid_embedding` | Use ID embeddings only; remove node/relation text-description mixing | Whether semantic-symbolic initialization helps sparse and long-tail graph elements | Hit@1 drops, especially for lexical/long-tail entities | 1 |
| `no_initial_graph_token_injection` | Keep graph states for downstream reasoning but remove the initial aligned graph residual and input cross-attention from the token stream | Whether early graph access is needed beyond later joint reasoning | Hit@1 drops on entity-linking and early disambiguation cases | 1 |
| `no_gnn` | Skip relation-aware graph message passing while retaining graph inputs, triples, and token-graph fusion | Whether relational-neighborhood refinement matters | Hit@1 drops most on multi-hop/path-dependent questions | 1 |
| `no_joint_reasoning` | Skip adapter-layer token-graph co-attention while retaining initial injection and graph refinement | Whether iterative text-structure exchange matters | Hit@1 drops on compositional questions | 1 |

These interventions deliberately preserve the remaining modules. In
particular, disabling initial injection does not clear the runtime graph state,
and disabling GNN does not disable triple encoding or joint reasoning.

The current paper draft lists `no_trips` where this requested protocol uses
`no_initial_graph_token_injection`. Before submission, either update the paper
ablation table and discussion to these four modules or run `no_trips` as an
additional fifth component ablation; `no_gnn` is not a substitute for it.

## Run order and reporting

Run `full` first, then `no_initial_graph_token_injection`, `no_joint_reasoning`,
`no_gnn`, and `no_hybrid_embedding`. Use at least three matched seeds before
making a component-importance claim. Report mean, standard deviation, and the
paired change from `full`; do not select seeds or tune only the full model.

Example validation without training:

```bash
python fedbiot_script/fedbiot/cwq/run_ablations.py all --run-tag seed0 --seed 0 --dry-run
```

One-round path smoke test (use a disposable run tag):

```bash
python fedbiot_script/fedbiot/cwq/run_ablations.py all --run-tag smoke-seed0 --seed 0 --smoke-test
```

Example full execution:

```bash
python fedbiot_script/fedbiot/cwq/run_ablations.py all --run-tag seed0 --seed 0 --evaluate
```

The launcher materializes one resolved YAML per variant. Both training and
evaluation use that file, so evaluation cannot silently re-enable an ablated
module. Repeat with matched `seed1` and `seed2`. Each seed schedules 90,000
client local-update batches (200 rounds x 3 clients x 30 batches x 5
variants); three seeds schedule 270,000. Measure the wall time of `seed0/full`
before estimating absolute GPU-hours. The configured `initial_update_rounds`
must not be assumed to equal executed alignment steps; verify optimizer-step
counts from the runtime log.

## Data-overlap limitation

The requested protocol is distribution alignment, not a strictly held-out
cross-dataset test. A repository audit found that 19,058/27,639 CWQ training
examples share a `webqsp_ID` with WebQSP training alignment data (1,831/2,754
unique base IDs overlap). Alignment also uses teacher-forced sequences that
contain answer tokens. This overlap is fixed identically across all variants,
so the within-protocol component comparison remains controlled, but the paper
must not describe WebQSP alignment as leakage-free or fully independent. A
CWQ validation audit also found 2,631/3,519 examples overlapping WebQSP train,
so CWQ validation is not leakage-free under this protocol. A stronger
follow-up is to precompute one WebQSP ID-disjoint alignment subset (1,021 of
3,098 WebQSP train examples remain after excluding CWQ train+dev base IDs),
save its manifest/hash, and reuse it unchanged for `full` plus all four
ablations; do not mix that stricter protocol into the primary table silently.

## Interpretation guardrail

The code makes the requested removals real; it does not and should not force a
performance drop. A module is supported as important only if the matched-seed
results show a stable degradation with uncertainty reported. Negative or
near-zero effects must also be recorded.
