# Twin2Attr

Сопоставление товарных карточек с помощью cross-encoder и классификатора.

## Установка

```bash
uv sync --extra notebooks
```

Для запуска без зависимостей ноутбуков достаточно `uv sync`.

## Запуск

Решение принимает обязательные именованные аргументы:

```bash
uv run python run.py \
  --items_path data/items.parquet \
  --matches_path data/matches.parquet \
  --output-path submit.csv
```

Короткие формы: `-i` для товаров, `-m` для пар и `-o` для результата.

## Сборка

```bash
uv build
```
