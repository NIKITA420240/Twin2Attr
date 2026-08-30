# Bidirectional category-conditioned head

The benchmark contains one trainable head over an immutable pair Transformer.
The backbone is evaluated for `(A, B)` and `(B, A)`. One shared directional
head uses mean/max card pools, symmetric interactions (`sum`, absolute
difference and product), and category-conditioned FiLM. The final logit is the
average of both directions.

The backbone is read-only, excluded from the optimizer and never saved. The
head is trained for 30 epochs; `head.pt` contains the epoch with the best
validation Macro PR-AUC.

Run tests:

```powershell
bench\.venv\Scripts\python.exe -m unittest discover -s bench -p "test_*.py"
```

Run the full training with the checkpoint currently stored as a Git LFS object:

```powershell
bench\.venv\Scripts\python.exe -X utf8 bench\train_heads.py `
  --config bench\config.yaml `
  --checkpoint-file .git\lfs\objects\32\fc\32fc6b735c8ee55bc5deadcd0af9c17020427d2f8b9f134b496aa97574e1b5fd
```

If `model.safetensors` exists at the path configured in `bench/config.yaml`,
omit `--checkpoint-file`. For a quick check, add `--max-pairs 2000 --seeds 42`.

Each run creates `bench/runs/<timestamp>` with:

- `report.json`, `results.parquet` and `aggregates.parquet`;
- `bidirectional_category_conditioned/seed-*/head.pt`;
- per-seed `metrics.json` and validation probabilities.

Forward and reverse frozen features are reused from `bench/cache`.
