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
`-m` и `-o`. Локально путь берётся из `inference.solution_path`; явный
`--solution` имеет приоритет. В собранном архиве используется `solution.json`
рядом с `run.py`.

## Сборка архива для evaluator

После обучения архив собирается непосредственно из `configs/pipeline.yaml`:

```bash
uv run python -m build_submission
```

`training.model` определяет стратегию обучения, а `inference.model` — модель,
которая попадёт в финальный архив. Эти параметры независимы: например, после
обучения `fusion` можно собрать более компактное решение только с Transformer.
Параметры и путь артефакта каждой модели находятся вместе в соответствующей
секции `model_description`. `inference` содержит имя выбранной модели и путь к
её `solution.json`.

Путь результата задаётся параметром `submission.output_path` и по умолчанию
равен `dist/twin2attr_submission.zip`. В корень ZIP попадают `run.py`,
`metadata.json`, автоматически сгенерированный `solution.json`, код из
`src/match` и только необходимые артефакты:

- `inference.model=transformer` — Transformer;
- `inference.model=maxpooling` — MaxPooling;
- `inference.model=fusion` — Transformer, MaxPooling и fusion head;
- `inference.model=boosting` — CatBoost и сохранённая схема его фич;
- `inference.model=cascade` — CatBoost и Transformer;
- `inference.model=stacking` — Transformer и CatBoost второго уровня;
- при включённой нормализации — оба parquet-файла правил;
- при включённом NER — NER-модель и cluster centers;
- physical provider не требует отдельных файлов весов.

Polars wheels из `build_submission/vendor_wheels` также автоматически
добавляются в архив. Если Polars отсутствует в evaluator image, `run.py`
устанавливает эти wheels во временную директорию до импорта inference-кода.
CatBoost wheel добавляется и устанавливается для `boosting`, `cascade` и
`stacking`.
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

Команда читает `model_description.transformer.pretrained_model_path`, создаёт
голову из `model_description.transformer.head` и сохраняет результат в
`model_description.transformer.artifact_dir`. Артефакт помечается
`match_initialized_only=true`
и файлом `initialization_metadata.json`: его предсказания случайны, поэтому он
не годится для итогового score. При
`model_description.transformer.pair_encoding.max_length=null` используется
значение 128.

## Обучение и диагностика

Основной конфиг находится в `configs/pipeline.yaml`. В нём задаются пути,
нормализация, токены полей, ограничения длины и параметры обучения.

Каждый источник mixed dataset может независимо выбирать модель веса примеров.
`constant` сохраняет базовый `source.weight`, а `transitivity` ищет полностью
наблюдаемые противоречивые треугольники (`A=B`, `B=C`, `A!=C`) до sampling и
уменьшает вес затронутых пар. Итоговый вес равен
`source.weight * weight_multiplier`. Учитываются только наблюдаемые связи:
отсутствующая пара не считается отрицательной.

```yaml
weight_model:
  type: transitivity
  enabled: true
  penalty_strength: 1.0
  min_weight_multiplier: 0.25
  min_comparable_neighbors: 2
  confidence_weighted_violations: true
```

При `confidence_weighted_violations=true` менее уверенное ребро получает более
сильный штраф относительно уверенных противоречащих ему LLM-рёбер. Модель
сохраняет диагностические колонки `transitivity_comparable_neighbors`,
`transitivity_violations`, `transitivity_violation_rate` и
`weight_multiplier`; при отключении используется множитель `1.0`.

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

# Посчитать важность атрибутов по attention pooling обученного Transformer
uv run python run.py analyze
```

## Накопительная LLM-разметка

Команда `label` выбирает пары из диапазона вероятностей
`labeling.lower_p..labeling.upper_p`, исключает уже успешно обработанные пары и
отправляет детерминированный sample в LLM:

```bash
export LLM_PROXY_TOKEN="..."
uv run python run.py label --config configs/pipeline.yaml
```

Пути, размер sample, параметры запросов и число retry-раундов находятся в
секции `labeling` файла `configs/pipeline.yaml`. Пара канонизируется как
`(min(id1, id2), max(id1, id2))`, поэтому обратная запись той же пары также не
будет выбрана повторно. Исключение выполнено точным Polars anti-join по двум
идентификаторам; отдельный потенциально коллизионный hash не является ключом.
Hash с фиксированным `seed` используется только для стабильного порядка
sampling.

Успешные результаты накапливаются в одном файле
`data/llm_annotations/annotations.parquet`. Он обновляется атомарно каждые
`checkpoint_every_batches` батчей. Строки, для которых все попытки завершились
ошибкой, в файл не попадают и могут быть повторены следующим запуском.

Fingerprint автоматически учитывает prompt, JSON Schema, LLM, temperature и
ограничение длины полей. Повторный запуск с тем же fingerprint выбирает
следующую ещё не размеченную часть. После изменения prompt fingerprint меняется,
и тот же детерминированный sample можно разметить заново для сравнения версий.

В retry-раундах обрабатываются только оставшиеся ошибки. Concurrency каждого
следующего раунда делится на два, но не опускается ниже
`labeling.llm.min_concurrency`. При настройках по умолчанию получается
`64 → 32 → 32`.

ML Core-запуск описан в `label-dataset.ml-job-preset.yml`. Перед submit нужно
заполнить `job.env.LLM_PROXY_TOKEN`; реальный токен нельзя коммитить. Результат
синхронизируется с
`/hotel_pricing/experiments/ET-1016/TWIN2ATTR/data/llm_annotations/` и при
следующем запуске загружается вместе с остальными данными.

Режим `analyze` не обучает и не изменяет модель. Он загружает Transformer из
`model_description.transformer.artifact_dir`, один раз инференсит выборку и
агрегирует attention-веса по исходным именам атрибутов обеих карточек. Для
каждого появления атрибута используется средний attention по его токенам,
умноженный на фактическую длину последовательности: так оценка не уменьшается
только из-за более длинной карточки. Затем значения усредняются по примерам.

Настройки анализа разделены на выбор режима и описание самого анализатора:

```yaml
analysis:
  analysis_model: attribute_importance
  data_model: base_dataset
  augmentation_model: attribute_shuffle
  data_postprocessing_model: null

augmentation_models:
  attribute_shuffle:
    shuffled_copies: 1
    keep_original: false
    seed: ${runtime.seed}
    shuffle_cards_independently: true
    skip_oversized: true

  attribute_word_dropout:
    pair_probability: 0.5
    attribute_dropout_probability: 0.15
    word_dropout_probability: 0.03
    keyboard_typo_probability: 0.01
    word_shuffle_probability: 0.0
    seed: ${runtime.seed}

data_postprocessing_models:
  attribute_sort:
    priorities_path: ${analysis_models.attribute_importance.output_dir}/${analysis_models.attribute_importance.output_file}

analysis_models:
  attribute_importance:
    model: transformer
    output_dir: analysis/attribute_importance
    # null означает анализ всей выбранной выборки
    sample_size: null
    group_by_category: true
    min_occurrences: 30
    score_type: normalized_mean_attention
    output_file: attribute_importance.parquet
    metadata_file: attribute_importance_metadata.json
```

При `group_by_category=true` итог содержит глобальные оценки и оценки внутри
категорий. Если пара «категория + атрибут» реально попала в Transformer менее
`min_occurrences` раз, для неё используется глобальная оценка атрибута. В
Parquet также записываются количество наблюдений и доля секций, не попавших в
Transformer из-за ограничения длины; рядом сохраняется JSON с параметрами
расчёта. `sample_size: null` отключает ограничение и анализирует всю выборку;
положительное число оставляет возможность сделать быстрый пробный расчёт.

`attribute_shuffle` работает после feature pipeline и до токенизации. При
`shuffled_copies: 1` и `keep_original: false` каждая исходная пара заменяется
одной детерминированно перемешанной версией, поэтому число Transformer-вызовов
не увеличивается. Аугментированные пары сохраняют заданный порядок атрибутов;
обычные пары при `augmentation_model: null` продолжают использовать прежнюю
сортировку от коротких полей к длинным. `skip_oversized: true` позволяет после
не поместившегося поля проверить следующие поля.

Для обучения `training.augmentation_model: attribute_word_dropout` динамически
создаёт новый вариант пары при каждом её чтении. С вероятностью
`pair_probability` аугментация применяется к паре; ключи атрибутов и категория
сохраняются, а названия товаров и значения получают word dropout, опечатки и
опциональную перестановку слов. Атрибуты могут удаляться целиком. На каждом
шаге Transformer получает только один вариант пары, поэтому второго forward
pass и смешивания двух loss нет. Validation и inference не аугментируются.

Анализ добавляет целочисленный `priority`, где `1` означает самый важный
атрибут. Приоритет рассчитывается отдельно для глобальных строк и внутри каждой
категории. На inference готовую таблицу можно включить так:

```yaml
inference:
  augmentation_model: null
  data_postprocessing_model: attribute_sort
```

`attribute_sort` получает только `priorities_path`: сначала использует
категориальный priority, затем глобальный, а неизвестные таблице атрибуты
оставляет в конце в исходном относительном порядке. Этап выполняется после
augmentation и перед `PairEncoding`. В training и analysis по умолчанию стоит
`data_postprocessing_model: null`, чтобы сохранить текущее обучение и не
создавать циклическую зависимость при построении самой таблицы.

Для анализа нужна уже обученная наша pooling-голова с `attention` в
`model_description.transformer.head.poolings`. Обычный
`BertForSequenceClassification`, модель только после `initialize` или голова
без attention будут отклонены с явной ошибкой. `max_length` и правила
токенизации читаются из сохранённого Transformer-артефакта, поэтому отдельно
дообучать модель для анализа не требуется.

Поля можно переопределять через Hydra/OmegaConf dotlist:

```bash
uv run python run.py train \
  features.normalization.enabled=true \
  model_description.transformer.pair_encoding.max_attribute_value_chars=256 \
  model_description.transformer.pair_encoding.max_attribute_value_tokens=48 \
  model_description.transformer.pair_encoding.max_length=256
```

`max_attribute_value_chars` ограничивает значение атрибута до токенизации,
а `max_attribute_value_tokens` — после неё. Значение `null` отключает
соответствующий предел. Обе настройки сохраняются в Transformer-артефакте и
одинаково применяются при обучении и инференсе.

Батчевое кодирование отдельных полей настраивается отдельно от правил сборки
пары и не меняет итоговые token IDs:

```yaml
model_description:
  transformer:
    tokenizer:
      batch_fields:
        enabled: true
        chunk_size: 16384
```

`chunk_size` здесь означает число названий, категорий, ключей и значений в
одном вызове fast tokenizer, а не число товарных пар для Transformer.

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

При `training.model=stacking` human-выборка делится на три непересекающиеся
части: Transformer train, CatBoost stacking train и validation. Доли последних
двух задаются через `stacking_train_fraction` и `validation_fraction`. CatBoost
получает 85 структурных признаков базового бустинга и один фиксированный
`transformer_logit_margin`.

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
`transformer`, `maxpooling` и `boosting` обучают соответствующую модель, `fusion`
автоматически обучает Transformer, MaxPooling и объединяющий классификатор, а
`stacking` — Transformer и CatBoost второго уровня.
`cascade` доступен только в `inference.model`: он использует уже независимо
обученные CatBoost и Transformer и сам не обучается.
Нормализация является первым этапом feature pipeline и управляется флагом
`features.normalization.enabled`. Например:

```bash
uv run python run.py train \
  training.model=fusion
```

Полное описание модели — гиперпараметры и путь её артефакта — лежит в
`model_description.transformer`,
`model_description.maxpooling`, `model_description.fusion` и
`model_description.boosting`. Состав и пороги каскада задаются в
`model_description.cascade`, а модели двух уровней — в
`model_description.stacking`. Выбранный trainer
возвращает единый `TrainingArtifacts`. Финальный predict запускается отдельно
через `run.py predict`; совместимый `solution.json` создаётся после успешного
обучения автоматически. Токенизация остаётся динамической внутри батчей.

Для Transformer отдельно задаются optimizer, train/eval batch size, gradient
accumulation, warmup, gradient clipping и early stopping. Например, запуск без
HPO с фиксированными параметрами выглядит так:

```bash
uv run python run.py train \
  model_description.transformer.hpo_trials=1 \
  model_description.transformer.learning_rate=0.00001 \
  model_description.transformer.weight_decay=0.01 \
  model_description.transformer.train_batch_size=32 \
  model_description.transformer.gradient_accumulation_steps=2
```

При `hpo_trials > 1` итоговые `learning_rate` и `weight_decay` выбирает Optuna
из диапазонов `hpo_learning_rate_min/max` и `hpo_weight_decay_min/max`.

### Частичный fine-tuning Transformer

Для BGE reranker по умолчанию включён P0-режим: обучаются только три верхних
encoder layer и classifier head. Input embeddings, включая добавленные
`[KEY]`/`[VAL]`, и более ранние слои backbone заморожены. Это уменьшает
потребление памяти optimizer/gradient и не строит backward-граф через нижнюю
часть модели.

```yaml
model_description:
  transformer:
    learning_rate: 3.0e-5
    train_last_n_layers: 3
    train_new_token_embeddings_only: false
    embeddings_learning_rate: null
    special_token_initialization:
      enabled: true
      key_seed_texts: [key, attribute, ключ, поле]
      value_seed_texts: [value, значение, характеристика]
    head_learning_rate: 1.0e-4
    layerwise_lr_decay: 0.9
```

`train_last_n_layers` одновременно является переключателем и числом
обучаемых верхних блоков:

- `null` — полный fine-tune: обучаются embeddings, все encoder layers и head;
- `3` — P0: обучаются top-3 и head;
- `5` или `6` — абляции с большей адаптационной способностью;
- значение больше количества layers модели завершается понятной ошибкой.

При численном `train_last_n_layers` значение
`train_new_token_embeddings_only: false` оставляет всю embedding matrix frozen.
Не включайте `train_new_token_embeddings_only: true` в P0: хотя optimizer
обновляет только две новые строки, autograd всё равно должен пройти до
embedding layer, что заметно ухудшает экономию памяти и времени. Этот режим
оставлен только для отдельного эксперимента с адаптацией `[KEY]`/`[VAL]`; в
нём `embeddings_learning_rate` задаёт LR этих строк.

При `special_token_initialization.enabled=true` новые `[KEY]` и `[VAL]`
инициализируются до freeze. Каждый текст из `key_seed_texts` или
`value_seed_texts` разбирается текущим BGE tokenizer на subword ids; среднее
их существующих embedding-векторов записывается соответственно в `[KEY]` или
`[VAL]`. Это обычные слова-доноры, а не ручные номера vocabulary. Например,
`значение` может состоять из нескольких subword-токенов, и все они войдут в
среднее. Пустой список или текст, дающий только special/unknown tokens,
завершает запуск ошибкой, чтобы не получить случайную инициализацию.

Для сравнимых экспериментов меняйте только глубину и сохраняйте одинаковыми
датасет, seed, `learning_rate` и effective batch:

```bash
# P0: top-3 + head
uv run python run.py train \
  model_description.transformer.train_last_n_layers=3

# Абляция: top-6 + head
uv run python run.py train \
  model_description.transformer.train_last_n_layers=6

# Полный fine-tune
uv run python run.py train \
  model_description.transformer.train_last_n_layers=null
```

### Скорость обучения Transformer

Без дополнительных флагов train использует динамический padding, length-aware
sampler и неблокирующую передачу батчей на GPU. Sampler сначала случайно
перемешивает примеры, затем упорядочивает только внутри `mega-batch`; поэтому
длинные и короткие карточки реже попадают в один batch, но глобальный порядок
не превращается в сортировку по длине. В конце каждой train-эпохи пишется ровно
одна компактная строка с `examples/s`, `real_tokens/s`, `padded_tokens/s`,
эффективностью padding и пиком CUDA-памяти. Те же записи сохраняются в
`training_metadata.json` как `performance_history`.

В `model_description.transformer.training_runtime` автоматически включены:

- `length_bucketing.enabled=true` и `mega_batch_multiplier=50`;
- четыре worker'а DataLoader, prefetch, persistent workers и pinned memory;
- `non_blocking_transfer=true`;
- эпоховое performance-логирование.

Обычно менять их не нужно. Для машины с другой CPU-производительностью можно
переопределить workers, например:

```bash
uv run python run.py train \
  model_description.transformer.training_runtime.dataloader.num_workers=8
```

Две опции намеренно выключены по умолчанию, поскольку требуют замера на
конкретной модели и H100:

```yaml
model_description:
  transformer:
    training_runtime:
      length_bucketing:
        # Ограничивает число форм тензоров; последний bucket равен max_length.
        padding_length_buckets: [128, 192, 256]
      torch_compile:
        enabled: true
        mode: reduce-overhead
```

`padding_length_buckets` полезен прежде всего вместе с `torch_compile`: он
округляет длину batch до ближайшей указанной длины, снижая число перекомпиляций
ценой дополнительного padding. Перед постоянным включением сравните
`real_tokens/s`, `padded_tokens/s`, peak CUDA memory и PR-AUC в metadata хотя
бы на одной полной эпохе. Чтобы откатить sampler для A/B, задайте
`length_bucketing.enabled=false`.

Параметры финального Transformer-инференса вынесены в секцию `inference`:
`inference.transformer.batch_size` задаёт размер батча, а
`inference.transformer.dtype` — точность вычислений (`float32`, `float16` или
`bfloat16`). Для H100 рекомендуется `bfloat16`. Параметры `num_workers`,
`prefetch_factor`, `pin_memory` и `non_blocking_transfer` управляют параллельной
подготовкой и передачей батчей. `length_bucketing.enabled` группирует пары
близкой длины и восстанавливает исходный порядок предсказаний после инференса.
Опциональный `length_bucketing.padding_length_buckets` округляет длину каждого
батча до ближайшего указанного значения, чтобы ограничить число форм тензоров;
последнее значение должно совпадать с `max_length` артефакта модели. Значение
`null` сохраняет обычный динамический padding.
`torch_compile` включает оптимизированный PyTorch-граф; если compilation backend
не поддерживает операцию, predictor автоматически продолжает в eager mode.

Готовый Transformer можно дополнительно экспортировать в два ONNX-графа:
`onnx/classifier.onnx` возвращает logits, а `onnx/encoder.onnx` — CLS embedding
для fusion. Stacking использует classifier-граф для logit margin. Экспорт
выполняется после сохранения обученной модели:

```yaml
model_description:
  transformer:
    export:
      onnx:
        enabled: true
        opset: 18
        precision: float16
        dynamic_batch: true
        dynamic_sequence_length: true
        export_classifier: true
        export_encoder: true
```

Существующий артефакт можно экспортировать без повторного обучения:

```bash
uv run --extra onnx-cpu python -m match.models.transformer.onnx_export \
  models/twin2attr/transformer --precision float16
```

На машине с NVIDIA GPU вместо `onnx-cpu` устанавливается extra `onnx-gpu`.
Backend выбирается независимо от экспорта:

```yaml
inference:
  transformer:
    backend: onnxruntime
    batch_size: 2048
    onnxruntime:
      provider: tensorrt
      device_id: 0
      io_binding: true
      graph_optimization: all
      fallback_to_pytorch: true
      tensorrt:
        engine_cache:
          enabled: true
          path: onnx/trt_cache
        timing_cache:
          enabled: true
        profiles:
          min_batch_size: 1
          opt_batch_size: 2048
          max_batch_size: 2048
          sequence_lengths: [64, 96, 128]
```

`device_id` выбирает GPU (для единственной H100 это `0`). `io_binding` передаёт
CUDA-тензоры по адресу без промежуточного преобразования в NumPy. При
недоступном ONNX Runtime/provider или ошибке загрузки графа
`fallback_to_pytorch: true` возвращает predictor к PyTorch. `torch_compile`
используется только при PyTorch backend. Для TensorRT после проверки CUDA
backend можно выбрать `provider: tensorrt`; TensorRT engine и timing cache
создаются в `onnx/trt_cache`. Для TensorRT список `sequence_lengths`
преобразуется в один поддерживаемый ONNX Runtime optimization profile:
первое значение становится `min`, среднее — `opt`, последнее — `max`.
Имена входов (`input_ids`, `attention_mask` и при наличии `token_type_ids`)
добавляются автоматически. Значения профиля должны покрывать все формы,
которые реально создаёт length bucketing.

Чистый TensorRT выбирается отдельным backend и не создаёт сессию ONNX
Runtime:

```yaml
inference:
  transformer:
    backend: tensorrt
    batch_size: 512
    dtype: float16
    tensorrt:
      device_id: 0
      workspace_size_gb: 8
      builder_optimization_level: 3
      fallback_to_onnxruntime: true
      engine_cache:
        enabled: true
        path: onnx/native_trt_cache
      timing_cache:
        enabled: true
        path: onnx/native_trt_cache/timing.cache
      profiles:
        min_batch_size: 1
        opt_batch_size: 512
        max_batch_size: 512
        sequence_lengths: [1, 256, 472]
```

Нативный backend строит engine из ONNX на целевом GPU и затем исполняет его
через TensorRT Python API, используя CUDA-адреса PyTorch-тензоров. Имя файла
engine содержит fingerprint ONNX-графа, профиля, версии TensorRT и GPU.
Инициализация имеет две явные стадии: `prepare()` загружает либо строит engine,
а `warmup()` выполняет короткий пробный проход до начала обработки датасета.
Построение engine, его постоянный кеш и исполнение разделены на независимые
компоненты; при ошибке native TensorRT ONNX Runtime создаётся напрямую без
повторного разбора и изменения solution manifest.
TensorRT engine аппаратно-зависим, поэтому собранный на другой видеокарте
файл не переносится; первая сборка может быть заметно дольше последующих
запусков. `builder_optimization_level` принимает значения `0..5`: меньшие
значения уменьшают время первой сборки, большие дают builder больше времени
на поиск быстрых kernels.

Пути моделей задаются рядом с их параметрами через `artifact_dir` или
`artifact_path`. `training.resolved_config_path` задаёт место для полного
resolved-конфига, а `training.solution_path` — место записи inference manifest.
`inference.solution_path` по умолчанию ссылается на тот же файл через
интерполяцию. При запуске `run.py train` все генерируемые пути автоматически
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
  "batch_size": 512,
  "dtype": "bfloat16",
  "torch_compile": {
    "enabled": true,
    "mode": "reduce-overhead",
    "dynamic": true
  }
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
