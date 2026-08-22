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

## Сборка архива для evaluator

После обучения архив собирается непосредственно из `configs/pipeline.yaml`:

```bash
uv run python -m build_submission
```

`training.model` определяет стратегию обучения, а `inference.model` — модель,
которая попадёт в финальный архив. Эти параметры независимы: например, после
обучения `fusion` можно собрать более компактное решение только с Transformer.

Путь результата задаётся параметром `submission.output_path` и по умолчанию
равен `dist/twin2attr_submission.zip`. В корень ZIP попадают `run.py`,
`metadata.json`, автоматически сгенерированный `solution.json`, код из
`src/match` и только необходимые артефакты:

- `inference.model=transformer` — Transformer;
- `inference.model=maxpooling` — MaxPooling;
- `inference.model=fusion` — Transformer, MaxPooling и fusion head;
- при включённой нормализации — оба parquet-файла правил;
- при включённом NER — NER-модель и cluster centers;
- physical provider не требует отдельных файлов весов.

Сборщик проверяет наличие каждого обязательного файла до создания ZIP и не
добавляет `__pycache__`, `.cache` и промежуточные `checkpoint-*`. Конфигурацию,
выходной путь и dotlist-параметры можно переопределить:

```bash
uv run python -m build_submission \
  --config configs/pipeline.yaml \
  --output dist/fusion_v1.zip \
  inference.model=fusion \
  features.physical.enabled=true
```

Если модель ещё не обучена, команда завершится ошибкой с путём отсутствующего
артефакта. Базовый `rubert-base-cased` содержит предобученные веса языка, но не
обученную голову мэтчинга, поэтому не является заменой результата `train`.

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

Итоговая модель выбирается одним параметром `training.model`. Значения
`transformer` и `maxpooling` обучают соответствующую модель, а `fusion`
автоматически обучает Transformer, MaxPooling и объединяющий классификатор.
Модель для последующей упаковки выбирается отдельно через `inference.model`.
Нормализация отдельно управляется флагом `normalization.enabled`. Например:

```bash
uv run python run.py train \
  training.model=fusion
```

Гиперпараметры лежат в `models_parameters.transformer`,
`models_parameters.maxpooling` и `models_parameters.fusion`. Выбранный trainer
возвращает единый `TrainingArtifacts`. Финальный predict запускается отдельно
через `run.py predict`; совместимый `solution.json` создаётся после успешного
обучения автоматически. Токенизация остаётся динамической внутри батчей.

Для Transformer отдельно задаются optimizer, train/eval batch size, gradient
accumulation, warmup, gradient clipping и early stopping. Например, запуск без
HPO с фиксированными параметрами выглядит так:

```bash
uv run python run.py train \
  models_parameters.transformer.hpo_trials=1 \
  models_parameters.transformer.learning_rate=0.00001 \
  models_parameters.transformer.weight_decay=0.01 \
  models_parameters.transformer.train_batch_size=32 \
  models_parameters.transformer.gradient_accumulation_steps=2
```

При `hpo_trials > 1` итоговые `learning_rate` и `weight_decay` выбирает Optuna
из диапазонов `hpo_learning_rate_min/max` и `hpo_weight_decay_min/max`.
Поле `batch_size` относится только к финальному Transformer-инференсу.

Пути Transformer, MaxPooling, Fusion, resolved-конфига и `solution.json`
задаются отдельно в секции `artifacts`. NER по умолчанию выключен, пока в
решение не добавлены его артефакты.

NER работает как опциональный item-enricher перед выбранной моделью. Он
извлекает из `name` бренд, тип, модель, материал, цвет, размер, назначение и
артикул, сохраняет результат в `ner_attributes`, а затем создаёт
`enriched_attributes`. Значения исходной карточки имеют приоритет над
предсказаниями NER.

```yaml
features:
  ner:
    enabled: true
    provider: word_ner
    model_dir: models/ner/rubert_tiny2_word_ner
    cluster_centers_path: models/ner/cluster_centers.pt
    source_column: name
    output_column: ner_attributes
    enriched_column: enriched_attributes
    merge_policy: missing_only
    batch_size: 512
    max_length: 100
    use_amp: true
    semantic_cleanup: true
```

Директория модели должна содержать `word_ner_config.json`, `model.pt`,
локальный Hugging Face `config.json` и tokenizer-файлы. Если включён
`semantic_cleanup`, также необходим `cluster_centers.pt`. Один и тот же
preprocessing применяется в `train`, `inspect` и финальном `predict`, а его
настройки автоматически записываются в `solution.json`.

Feature-провайдеры создаются через `ItemEnricherFactory` и последовательно
применяются объектом `FeaturePipeline`. Каждый провайдер включается собственным
флагом `enabled`; порядок зафиксирован как `NER → physical`.

Physical provider переносит алгоритмическую часть parser из notebook. Он
извлекает из `name` размеры, массу, объём, мощность, напряжение, память,
частоту и другие значения с единицами измерения. Конструкции `AxB` и `AxBxC`
разбиваются на длину, ширину и высоту. После извлечения используется общий
нормализатор физических единиц, поэтому, например, `120x60x75 см` становится
значениями длины, ширины и высоты в миллиметрах.

```yaml
features:
  physical:
    enabled: true
    source_column: name
    output_column: physical_attributes
    enriched_column: feature_attributes
    merge_policy: missing_only
    normalize_units: true
    n_jobs: 1
    chunk_size: 10000
```

Если включены оба провайдера, порядок по умолчанию выглядит так:
`normalization → NER → physical → model`. Исходные атрибуты карточки на каждом
шаге имеют приоритет над автоматически извлечёнными значениями.

OmegaConf используется только на границе загрузки YAML и dotlist overrides.
Внутри workflow передаётся неизменяемый типизированный `AppConfig`, поэтому
этапы обучения не зависят от формата конфигурационного файла.

## Конфигурация финального решения

По умолчанию `solution.json` выбирает обученный transformer:

```json
{
  "predictor": "transformer",
  "model_directory": "models/twin2attr/transformer",
  "batch_size": 64
}
```

Для fusion в manifest указываются `predictor: fusion`, `maxpooling_path` и
`fusion_path`. `metadata.json` задаёт evaluator entrypoint `python -u run.py`.
MaxPooling также можно использовать самостоятельно, указав
`predictor: maxpooling` и `maxpooling_path`.

Модели подключаются к workflow через небольшие интерфейсы `PairEncoder` и
`MatchPredictor`. Transformer и MaxPooling могут извлекать признаки и делать
предсказания, а Fusion композиционно объединяет их признаки.

Реализации разделены по модели и ответственности:

```text
src/match/models/
├── contracts.py
├── artifacts.py
├── factory.py
├── transformer/
│   ├── config.py
│   ├── model.py
│   ├── training.py
│   ├── predictor.py
│   └── metrics.py
├── maxpooling/
│   ├── model.py
│   ├── features.py
│   ├── training.py
│   ├── predictor.py
│   └── serialization.py
└── fusion/
    ├── model.py
    ├── training.py
    ├── predictor.py
    └── serialization.py
```

`factory.py` — единственная точка выбора конкретного trainer или predictor.
Workflow работает через контракты, а файлы каждой модели не смешивают
определение сети, обучение, инференс и сохранение артефактов.

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
