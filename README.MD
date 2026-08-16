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

## Нормализация атрибутов

Публичная функция принимает и возвращает `polars.DataFrame`. По умолчанию она
читает JSON из столбца `attributes` и добавляет столбец
`normalized_attributes`, не изменяя исходный DataFrame:

```python
import polars as pl

from match import normalize_attributes

items = pl.read_parquet("items.parquet")
normalized_items = normalize_attributes(
    items,
    "data/synonyms_df.parquet",
    "data/untouched_stats.parquet",
    n_jobs=4,
    chunk_size=5_000,
)
```

Имена столбцов можно изменить через параметры `source_column` и
`output_column`. По умолчанию нормализация выполняется в четырёх отдельных
процессах с чанками по 5000 строк, как в notebook. `n_jobs=1` отключает
multiprocessing; если оперативной памяти мало, используйте `n_jobs=2`.

## Сборка

```bash
uv build
```
