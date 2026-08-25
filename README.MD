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
Пути готовых весов задаются отдельно через `inference.transformer_dir`,
`inference.maxpooling_path`, `inference.fusion_path` и
`inference.boosting_dir`, поэтому сборка не зависит от директорий, куда
training workflow сохраняет новые артефакты.

Путь результата задаётся параметром `submission.output_path` и по умолчанию
равен `dist/twin2attr_submission.zip`. В корень ZIP попадают `run.py`,
`metadata.json`, автоматически сгенерированный `solution.json`, код из
`src/match` и только необходимые артефакты:

- `inference.model=transformer` — Transformer;
- `inference.model=maxpooling` — MaxPooling;
- `inference.model=fusion` — Transformer, MaxPooling и fusion head;
- `inference.model=boosting` — CatBoost и сохранённая схема его фич;
- `inference.model=cascade` — CatBoost и Transformer;
- при включённой нормализации — оба parquet-файла правил;
- при включённом NER — NER-модель и cluster centers;
- physical provider не требует отдельных файлов весов.

Polars wheels из `build_submission/vendor_wheels` также автоматически
добавляются в архив. Если Polars отсутствует в evaluator image, `run.py`
устанавливает эти wheels во временную директорию до импорта inference-кода.
CatBoost wheel добавляется и устанавливается только для `boosting` и `cascade`.
При включённой нормализации или physical provider сборщик аналогично добавляет
полный runtime предобработки: `pymorphy3`, русские словари, `joblib`, `loguru`,
`pint` и их офлайн-зависимости.

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
артефакта. Сырой pretrained checkpoint содержит языковые веса, но не обученную
голову мэтчинга, поэтому не является заменой результата `train`.
Для Transformer сборщик также проверяет classification architecture, сохранённые
`match_*` настройки и специальные токены, чтобы не упаковать сырой language model.

Для быстрой технической проверки новой архитектуры можно создать совместимый
артефакт без обучения:

```powershell
uv run python run.py initialize
```

Команда читает `models_parameters.transformer.pretrained_model_path`, создаёт
голову из `models_parameters.transformer.head` и сохраняет результат в
`artifacts.transformer_dir`. Артефакт помечается `match_initialized_only=true`
и файлом `initialization_metadata.json`: его предсказания случайны, поэтому он
не годится для итогового score. При `pair_encoding.max_length=null` используется
значение 128.

## Обучение и диагностика

Основной конфиг находится в `configs/pipeline.yaml`. В нём задаются пути,
нормализация, токены полей, ограничения длины и параметры обучения.

Каждый train-запуск обязан получить имя эксперимента от пользователя. Имя не
версионируется автоматически: перед новым экспериментом его нужно поменять.

```bash
export EXPERIMENT_NAME=mmarco-l12-human-last3-special-tokens
```

Все сгенерированные модели, checkpoints, resolved-конфиг, `solution.json`,
`experiment.json` и лог сохраняются в `experiments/$EXPERIMENT_NAME/`. После
успешного обучения строка добавляется в `experiments/experiments.csv`. Если имя
использовать повторно, папка эксперимента будет перезаписана, а в CSV появится
ещё одна строка — уникальность имени контролирует пользователь. В карточку
эксперимента записываются размеры train/validation и стабильный hash точного
набора validation-пар, поэтому одинаковость split можно проверить без сохранения
отдельных parquet-файлов.

```bash
# Обучение
uv run python run.py train

# Только безопасно оценить max_length, ничего не обучая и не сохраняя
uv run python run.py inspect
```

Поля можно переопределять через Hydra/OmegaConf dotlist:

```bash
uv run python run.py train \
  features.normalization.enabled=true \
  pair_encoding.max_attribute_value_tokens=48 \
  pair_encoding.max_length=256
```

Источник данных выбирается через `training.data_model`. `base_dataset` обучается
только на человеческой разметке и создаёт group-aware validation. Для него пути
и параметры разделения находятся в `data_model_description.base_dataset`:

```bash
uv run python run.py train \
  training.data_model=base_dataset \
  data_model_description.base_dataset.items=data/items_human.parquet \
  data_model_description.base_dataset.matches=data/matches.parquet \
  data_model_description.base_dataset.leakage_scope=item \
  data_model_description.base_dataset.validation_fraction=0.2
```

`leakage_scope=pair` не разносит одинаковые и обратные пары по разным частям.
`leakage_scope=item` не допускает ни одного общего товара, но из-за связанных
компонент может менее точно сохранять требуемое распределение.

`mix_dataset` добавляет к human train уверенные LLM-метки, но validation всегда
строит только из `validation_source`. Веса источников умножают loss каждого
обучающего примера, а validation использует нейтральный вес:

```bash
uv run python run.py train training.data_model=mix_dataset
```

Настройки источников, пороги голосов и ограничение размера LLM-выборки находятся
в `data_model_description.mix_dataset.sources`.

Итоговая модель выбирается одним параметром `training.model`. Значения
`transformer`, `maxpooling` и `boosting` обучают соответствующую модель, а `fusion`
автоматически обучает Transformer, MaxPooling и объединяющий классификатор.
`cascade` доступен только в `inference.model`: он использует уже независимо
обученные CatBoost и Transformer и сам не обучается.
Нормализация является первым этапом feature pipeline и управляется флагом
`features.normalization.enabled`. Например:

```bash
uv run python run.py train \
  training.model=fusion
```

Гиперпараметры лежат в `models_parameters.transformer`,
`models_parameters.maxpooling`, `models_parameters.fusion` и
`models_parameters.boosting`. Состав и пороги каскада задаются в
`models_parameters.cascade`. Выбранный trainer
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

Базовые пути Transformer, MaxPooling, Fusion, resolved-конфига и `solution.json`
задаются в секции `artifacts`. При запуске `run.py train` они автоматически
перенаправляются внутрь текущего `experiments/$EXPERIMENT_NAME`. Общие входные
данные, pretrained-модели и NER-артефакты не копируются в каждый эксперимент;
`solution.json` ссылается на них относительными путями.

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

Порядок этапов задаётся явно в `features.execution_order`:

```yaml
features:
  execution_order:
    - normalization
    - ner
    - physical
```

Каждый известный этап должен встретиться ровно один раз, а `enabled` определяет,
будет ли он запущен. Порядок также задаёт приоритет автоматически извлечённых
значений: результат более раннего этапа не заменяется более поздним при
`merge_policy: missing_only`.

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
MaxPooling и Boosting также можно использовать самостоятельно, указав
соответственно `predictor: maxpooling`/`maxpooling_path` или
`predictor: boosting`/`boosting_directory`.

Модели подключаются к workflow через небольшие интерфейсы `PairEncoder` и
`MatchPredictor`. Transformer и MaxPooling могут извлекать признаки и делать
предсказания, Boosting строит свою табличную схему, Fusion объединяет
эмбеддинги, а Cascade маршрутизирует неуверенные ответы быстрой модели в
основную.

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
├── fusion/
    ├── model.py
    ├── training.py
    ├── predictor.py
    └── serialization.py
├── boosting/
│   ├── features.py
│   ├── training.py
│   ├── predictor.py
│   └── serialization.py
└── cascade/
    └── predictor.py
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
