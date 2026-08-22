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

## Полный pipeline

Новый Hydra-конфиг находится в `configs/pipeline.yaml`. В нём задаются пути,
нормализация, токены полей, ограничения длины, обучение и артефакты инференса.

```bash
# Обучение
uv run python pipeline_cli.py action=train

# Инференс
uv run python pipeline_cli.py action=inference

# Только безопасно оценить max_length, ничего не обучая и не сохраняя
uv run python pipeline_cli.py action=inspect
```

Любое поле можно переопределить через Hydra, например:

```bash
uv run python pipeline_cli.py \
  action=train \
  'pipelines.train=[read_data,normalize,split_data,prepare_data,pair_encoding,transformer]' \
  pair_encoding.max_attribute_value_tokens=48 \
  pair_encoding.max_length=256
```

Разделение данных настраивается через `split`. В режиме `predefined` pipeline
читает готовые `paths.train_matches` и `paths.validation_matches` и проверяет
утечки. В режиме `auto` файл `paths.train_matches` считается полным набором
размеченных пар, после чего создаётся group-aware split с сохранением
распределения `category × target`:

```bash
uv run python pipeline_cli.py \
  action=train \
  split.mode=auto \
  split.leakage_scope=pair \
  split.validation_fraction=0.2
```

`leakage_scope=pair` не разносит одинаковые и обратные пары по разным частям.
`leakage_scope=item` не допускает ни одного общего товара, но из-за связанных
компонент может менее точно сохранять требуемое распределение.

Порядок и состав `pipelines.train` и `pipelines.inference` — единственный
переключатель стадий. Например, добавление `maxpooling` обучает или применяет
MaxPooling, а последовательность `transformer`, `maxpooling`, `fusion`
обучает или применяет общую fusion-голову. Стадия `pair_encoding` разрешает
настройки токенизации и `max_length`, но не материализует токены для всего
датасета: токенизация остаётся динамической внутри батчей.

Полный resolved-конфиг обучения сохраняется рядом с моделью в
`pipeline_config.yaml`. NER оставлен явной точкой расширения и должен
оставаться выключенным до реализации provider-а.

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
