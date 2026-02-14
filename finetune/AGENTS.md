# FINETUNE KNOWLEDGE BASE

## OVERVIEW
`finetune/` trains and evaluates the query-expansion model used by QMD hybrid retrieval. It is a Python/uv workflow, separate from Bun runtime code.

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| End-to-end training entry | `train.py` | `sft` and `grpo` subcommands |
| Reward/scoring logic | `reward.py` | single source of truth for quality score |
| Eval runner | `eval.py` | local/HF model evaluation |
| Model conversion | `convert_gguf.py` | export deployment artifacts |
| Hyperparameters | `configs/*.yaml` | SFT/GRPO config files |
| Dataset tooling | `dataset/` | schema, generation, cleaning, prep |
| HF job scripts | `jobs/` | self-contained cloud jobs |
| GEPA prompt optimization | `gepa/` | DSPy-based optimization flow |

## CONVENTIONS (LOCAL)
- Use `uv run ...` commands; avoid ad hoc environment-specific wrappers.
- Treat `reward.py` as scoring authority for both RL reward and eval interpretations.
- Training inputs are JSONL files under `data/`; schema should match dataset helpers.
- Push/update model repos only when eval improves; include eval evidence.

## ANTI-PATTERNS
- Never upload model artifacts without running eval first.
- Do not introduce version-suffixed model repos for routine updates.
- Do not bypass schema validation/quality checks before training.

## COMMANDS
```bash
# Data checks
uv run dataset/validate_schema.py
uv run dataset/score_data.py
uv run dataset/prepare_data.py

# Training
uv run train.py sft --config configs/sft.yaml
uv run train.py grpo --config configs/grpo.yaml

# Evaluation + conversion
uv run eval.py --model ./outputs/grpo -o eval_results.json
uv run reward.py
uv run convert_gguf.py --size 1.7B
```

## NOTES
- `jobs/` scripts are intentionally self-contained for HF Jobs execution.
- `outputs/` is local and gitignored; avoid writing training artifacts elsewhere.
- `Justfile` provides repeatable local validation/training helpers.
