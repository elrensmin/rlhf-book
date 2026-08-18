# Rejection Sampling "Choice Axis" Ablation

This folder contains a matched-baseline experiment campaign for Chapter 9
(Rejection Sampling) of the RLHF Book.

## Commands

All commands run from `code/`. Set `WANDB_PROJECT` to your own project (or
disable logging with `WANDB_MODE=disabled`). Run one training job at a time;
for long runs, background each invocation and monitor it.

### E1: `num_completions_per_prompt` sweep

```bash
cd code/
export WANDB_PROJECT=rlhf-book

uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e1_n04_top_per_prompt.yaml
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e1_n04_random_per_prompt.yaml
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e1_n16_top_per_prompt.yaml
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e1_n16_random_per_prompt.yaml
```

N=8 is the existing baseline in `rejection_sampling/configs/top_per_prompt.yaml`
and `random_per_prompt.yaml`.

### E2: `selection.top_k` sweep

```bash
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e2_k500_top_k_overall.yaml
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e2_k500_random_k_overall.yaml
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e2_k2000_top_k_overall.yaml
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e2_k2000_random_k_overall.yaml
```

K=1000 is the existing baseline in `rejection_sampling/configs/top_k_overall.yaml`
and `random_k_overall.yaml`.

### E3: smaller policy model

```bash
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e3_qwen06b_top_per_prompt.yaml
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e3_qwen06b_random_per_prompt.yaml
```

### E4: small reward model

E4 uses a self-trained Qwen3-0.6B outcome reward model (ORM) instead of
AceMath-7B-RM. First train the ORM and save a checkpoint, then run rejection
sampling with it as the reward model:

```bash
# Step 1: train the small ORM and save a checkpoint
uv run python -m reward_models.train_orm \
  --config rejection_sampling/ablations/configs/orm_qwen3_0.6b.yaml

# Step 2: run rejection sampling with the small ORM as the reward model
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e4_small_rm_top_per_prompt.yaml
uv run python -m rejection_sampling.train \
  --config rejection_sampling/ablations/configs/e4_small_rm_random_per_prompt.yaml
```

The rejection-sampling pipeline detects the ORM checkpoint (via its
`orm_checkpoint.json` marker) and scores with the ORM path automatically.

## Diagnostics

After all caches exist, run `rejection_sampling.diagnostics` on each cache to
get `decidable_fraction`, per-row RM winrate, and best-of-N coverage:

```bash
cd code/
for cache in rejection_sampling/output/rollouts/*.jsonl; do
  uv run python -m rejection_sampling.diagnostics \
    --cache "$cache" \
    --out-dir rejection_sampling/ablations/diagnostics/$(basename "$cache" .jsonl)
done
```

## Files

- `configs/` — 12 YAML files for E1, E2, E3, and E4.
- `analyze_results.py` — W&B CSV → result table + gap plots.
- `2026-08-17-rs-choice-sweep.md` — full experiment log and write-up.
