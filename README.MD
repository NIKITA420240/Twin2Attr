# Twin2Attr

Сопоставление товарных карточек с помощью cross-encoder и классификатора.

## Установка

```bash
uv sync --extra notebooks
```

Для запуска без зависимостей ноутбуков достаточно `uv sync`.

## Запуск

Значения по умолчанию находятся в `configs/config.yaml`:

```bash
uv run python run.py
```

Любой параметр можно переопределить через Hydra в формате `ключ=значение`:

```bash
uv run python run.py \
  items_path=data/items.parquet \
  matches_path=data/matches.parquet \
  output_path=predictions.csv \
  classifier_path=baseline_logreg_l12.joblib \
  model_path=models/cross-encoder-ms-marco-MiniLM-L12-v2
```

Префикс `--` используется для собственных флагов Hydra, поэтому параметры
приложения передаются без него.

## Сборка

```bash
uv build
```
