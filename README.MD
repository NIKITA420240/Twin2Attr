# Twin2Attr

Сопоставление товарных карточек с помощью cross-encoder и классификатора.

## Установка

```bash
uv sync --extra notebooks
```

Для запуска без зависимостей ноутбуков достаточно `uv sync`.

## Единая точка запуска

`run.py` используется и для локального обучения, и для финального инференса.
Вызов без подкоманды совместим с evaluator и всегда означает инференс:

```bash
uv run python run.py \
  --items_path data/items.parquet \
  --matches_path data/matches.parquet \
  --output_path submit.csv
```

Результат сохраняется как CSV со столбцами `id1`, `id2`, `predict`. Выбранная
модель и пути к её артефактам фиксируются в `solution.json`, поэтому evaluator
не должен передавать дополнительный параметр модели. Короткие формы: `-i`,
`-m` и `-o`.

## Обучение и диагностика

Основной конфиг находится в `configs/pipeline.yaml`. В нём задаются пути,
нормализация, токены полей, ограничения длины и параметры обучения.

```bash
# Обучение
uv run python run.py train

# Только безопасно оценить max_length, ничего не обучая и не сохраняя
uv run python run.py inspect
```

Поля можно переопределять через Hydra/OmegaConf dotlist:

```bash
uv run python run.py train \
  normalization.enabled=true \
  pair_encoding.max_attribute_value_tokens=48 \
  pair_encoding.max_length=256
```

Разделение данных настраивается через `split`. В режиме `predefined` pipeline
читает готовые `paths.train_matches` и `paths.validation_matches` и проверяет
утечки. В режиме `auto` файл `paths.train_matches` считается полным набором
размеченных пар, после чего создаётся group-aware split с сохранением
распределения `category × target`:

```bash
uv run python run.py train \
  paths.items=data/items_human.parquet \
  paths.train_matches=data/matches.parquet \
  split.mode=auto \
  split.leakage_scope=item \
  split.validation_fraction=0.2
```

`leakage_scope=pair` не разносит одинаковые и обратные пары по разным частям.
`leakage_scope=item` не допускает ни одного общего товара, но из-за связанных
компонент может менее точно сохранять требуемое распределение.

Порядок обучения задан явно в `workflows.train()`: подготовка данных, вычисление
`max_length` и обучение Transformer. Дополнительные компоненты включаются
флагами `features.maxpooling.enabled` и `fusion.enabled`. Для fusion необходимо
также включить MaxPooling. Нормализация управляется флагом
`normalization.enabled`. Например:

```bash
uv run python run.py train \
  features.maxpooling.enabled=true \
  fusion.enabled=true
```

Конфиг задаёт параметры компонентов, но не конструирует порядок выполнения из
строк. Финальный predict запускается отдельно через `run.py predict` и
`solution.json`. Токенизация остаётся динамической внутри батчей.

Полный resolved-конфиг обучения сохраняется рядом с моделью в
`pipeline_config.yaml`. NER оставлен явной точкой расширения и должен
оставаться выключенным до реализации provider-а.

OmegaConf используется только на границе загрузки YAML и dotlist overrides.
Внутри workflow передаётся неизменяемый типизированный `AppConfig`, поэтому
этапы обучения не зависят от формата конфигурационного файла.

## Конфигурация финального решения

По умолчанию `solution.json` выбирает обученный transformer:

```json
{
  "predictor": "transformer",
  "model_directory": "models/twin2attr",
  "batch_size": 64
}
```

Для fusion в manifest указываются `predictor: fusion`, `maxpooling_path` и
`fusion_path`. `metadata.json` задаёт evaluator entrypoint `python -u run.py`.

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
