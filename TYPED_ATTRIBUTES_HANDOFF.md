# Typed Attribute Metrics — handoff

Дата состояния: 2026-08-30.

Этот документ нужен для продолжения работы в новой Codex-задаче без повторного
объяснения контекста. Репозиторий: `/home/andrey/Documents/Twin2Attr`.

## Цель работы

Мы проверяем идею из ноутбука
`/home/andrey/Downloads/Telegram Desktop/ozon-e-cup-model-llm (10).ipynb`:
сравнивать значения одинаковых атрибутов разными детерминированными способами в
зависимости от их семантического типа.

Поддерживаемые типы:

- `CODE` — артикулы, OEM, SKU, model/part numbers;
- `PHYSICAL` — вес, размеры, мощность, напряжение и другие физические значения;
- `NUMERIC` — обычные числовые значения;
- `SET` — цвет, материал, комплектация и другие множества значений;
- `TEXT` — оставшиеся текстовые атрибуты.

Главная идея: Transformer продолжает понимать текст и общий смысл карточек, а
typed metrics дают точные результаты структурированного сравнения — совпадение
кодов, относительную разницу чисел, Jaccard/containment множеств и признаки
отсутствия значений.

На первом этапе эти признаки подключены не к Transformer, а к CatBoost и
stacking. Это позволяет сначала доказать их полезность дешёвой абляцией. Новый
Transformer head следует делать только после подтверждённого прироста.

## Важные архитектурные решения

### Pair-level, а не item-level

Typed metrics вычисляются только после образования пары карточек. Например,
признак отсутствия существует относительно второй карточки. Поэтому они
находятся в отдельном `pair_features`, а не в существующем item-level блоке
`features`.

### Тип сохраняется при отсутствии значения

В исходном ноутбуке отсутствующий атрибут превращался в отдельный тип
`MISSING`. Здесь сделано иначе:

```text
semantic_type = SET
one_missing = 1
```

То есть модель знает одновременно, что отсутствует значение и что это, например,
цвет, а не OEM или физический размер.

### Симметрия обязательна

Все новые признаки инвариантны относительно перестановки `id1` и `id2`.
Направленные `only_left`/`only_right` не используются; вместо них используется
симметричный `one_missing`. Это защищено тестами.

### Нормализация происходит раньше

Comparator получает `PreparedPair`, построенный из уже выбранной колонки
атрибутов. В нормальной конфигурации это нормализованные атрибуты. Typed
comparator не должен заменять существующую нормализацию единиц и имён ключей.

## Что уже реализовано

### Общий comparator

Основной код:

```text
src/match/pair_features/__init__.py
src/match/pair_features/typed_attributes.py
```

Ключевые классы и функции:

- `TypedAttributeOptions` — стабильные настройки и manifest contract;
- `TypedAttributeComparator` — выравнивание ключей, определение типов и метрики;
- `TypedAttributeComparison` — результат сравнения одного ключа;
- `aggregate_typed_attribute_features` — перевод переменного числа сравнений в
  фиксированную CatBoost-схему.

Реализованные метрики:

- `CODE`: normalized exact intersection, set exact, SequenceMatcher similarity,
  совпадение цифровой части, Jaccard, containment;
- `PHYSICAL`/`NUMERIC`: exact, relative difference, min/max ratio;
- `SET`: exact, Jaccard, Dice, containment;
- `TEXT`: exact, token Jaccard, char-trigram Jaccard;
- для всех типов: `both_present`, `one_missing`, exact, similarity и strong
  conflict.

Также есть отдельные агрегаты для:

- model codes;
- OEM;
- бренда.

### CatBoost и stacking

Изменённые файлы:

```text
src/match/models/boosting/features.py
src/match/models/boosting/training.py
src/match/models/boosting/predictor.py
src/match/models/boosting/serialization.py
src/match/models/stacking/features.py
src/match/models/stacking/training.py
src/match/models/stacking/predictor.py
src/match/models/stacking/serialization.py
```

Что сделано:

- `BoostingFeatureBuilder` принимает `TypedAttributeOptions`;
- typed features добавляются только при `enabled=true`;
- без typed features остаётся прежняя схема из 85 колонок;
- со всеми typed features получается 137 колонок;
- stacking автоматически использует тот же structured feature builder;
- настройки feature builder сохраняются в `manifest.json` артефакта;
- predictor восстанавливает настройки из manifest и проверяет точное совпадение
  списка колонок.

`FEATURE_SCHEMA_VERSION` повышена с `1` до `2`. Старые CatBoost, cascade и
stacking-артефакты со schema version 1 намеренно не загружаются новым кодом — их
нужно переобучить.

### Конфигурация

Изменены:

```text
src/match/config.py
configs/pipeline.yaml
configs/benchmark_pipeline.yaml
```

Добавлен блок:

```yaml
pair_features:
  typed_attributes:
    enabled: true
    detector: rules
    preserve_semantic_type_when_missing: true
    symmetric: true
    types:
      code: true
      physical: true
      numeric: true
      set: true
      text: true
```

Для Transformer этот блок сейчас ничего не меняет. Он используется CatBoost и
stacking.

### Подготовленная абляция

Файл:

```text
configs/benchmark_tests/typed_attribute_quality.yaml
```

Варианты:

1. `typed_attributes_off` — baseline без typed metrics;
2. `code_only`;
3. `code_physical`;
4. `code_physical_set`;
5. `all_typed_attributes`.

В `configs/benchmark.yaml` добавлена задача `typed_attribute_quality`, но она
намеренно оставлена `enabled: false`. Обучение CatBoost ещё не запускалось.

Screening сейчас настроен на seed `42`. Для окончательного решения нужно
подтвердить лучший вариант на seed `42`, `43`, `44`.

## Выполненные проверки

Добавлены/изменены тесты:

```text
tests/test_typed_attribute_features.py
tests/test_boosting_cascade.py
tests/test_config.py
tests/test_benchmark_runner.py
```

Проверяется:

- type detection;
- type-specific metrics;
- сохранение семантического типа при missing value;
- инвариантность к перестановке карточек;
- отсутствие `NaN`/`inf`;
- включение и отключение отдельных типов;
- round-trip настроек через manifest;
- стабильность CatBoost-схемы;
- валидность всех пяти ablation-конфигураций.

Последний результат: 50 тестов прошли. Также прошли `compileall` и
`git diff --check`. CatBoost training/benchmark не запускался.

Обычный `uv run` в sandbox сначала не смог записать lock в
`/home/andrey/.cache/uv`, а с cache в `/tmp` попытался скачать отсутствующую
зависимость при закрытой сети. Поэтому проверки были выполнены локальным Python
с read-only пакетами из существующего uv cache. Это ограничение окружения, не
ошибка проекта.

## Что делать дальше

### 1. Перед запуском проверить diff

```bash
git status --short
git diff --check
git diff --stat
```

Особенно внимательно проверить:

- правила `_CODE_MARKERS`, `_PHYSICAL_MARKERS`, `_NUMERIC_MARKERS`,
  `_SET_MARKERS`;
- смысл strong conflict для кодов;
- что входная `attributes_column` действительно нормализована;
- приемлемость 52 дополнительных признаков и скорости их вычисления.

### 2. Запустить только screening typed attributes

В `configs/benchmark.yaml`:

- установить `typed_attribute_quality.enabled: true`;
- отключить другие training-quality jobs, особенно текущий
  `soft_label_confidence_quality.enabled: true`, чтобы не запустить лишние
  обучения.

Затем:

```bash
uv run python run.py benchmark
```

Это первый ещё не выполненный шаг. Перед ним пользователь специально попросил
остановиться.

### 3. Сравнить screening-результаты

Основная метрика:

```text
validation_macro_pr_auc
```

Нужно сравнить каждый вариант с `typed_attributes_off` и проверить:

- знак и величину `delta_validation_macro_pr_auc`;
- время построения признаков и полное `training_seconds`;
- нет ли падения из-за `TEXT`, который может дублировать старые признаки;
- даёт ли `CODE` основную часть прироста;
- добавляет ли `PHYSICAL` сигнал при текущей нормализации;
- полезен ли `SET` или создаёт шум.

Ожидаемый вероятный сценарий: основную пользу дадут `CODE` и `PHYSICAL`, а
`TEXT` может оказаться избыточным.

### 4. Провести ручной error analysis

Для лучшего варианта желательно выгрузить:

- уверенные FP с `typed_model_code_conflict=1`;
- FN с `typed_model_code_exact_match=1`;
- случаи `typed_oem_exact_match=1`;
- ошибки в категориях с худшим normalized PR-AUC.

Нужно проверить ложные определения типов:

- model code ошибочно распознан как physical/numeric;
- обычное число ошибочно распознано как code;
- строковый цвет/материал неправильно разбит на множество;
- несколько OEM-кодов трактуются как конфликт, хотя есть общее значение.

### 5. Подтвердить лучший вариант на трёх seed

В `configs/benchmark_tests/typed_attribute_quality.yaml` заменить:

```yaml
seeds:
  - 42
```

на:

```yaml
seeds:
  - 42
  - 43
  - 44
```

Для финального решения смотреть среднюю парную дельту, стандартное отклонение и
одинаковость validation hash.

### 6. Проверить stacking

Если typed CatBoost стабильно выигрывает baseline, следующий дешёвый шаг:

```text
Transformer logit margin
+
CatBoost typed features
→ stacking
```

Поддержка typed features в stacking уже добавлена. Потребуется отдельное
обучение stacking, но менять архитектуру больше не нужно.

### 7. Только после доказанного прироста рассматривать Transformer fusion

Возможный следующий этап:

```text
typed comparisons
    → type-specific MLP
    → attention pooling
    → structured vector

Transformer pooled embedding + structured vector
    → fusion head
```

Для этого понадобятся новый collator contract, дополнительные tensors,
сериализация нового head, predictor и ONNX/TensorRT работа. Этого пока нет и до
CatBoost-абляции делать не следует.

## Текущие ограничения реализации

- Type detector пока основан только на правилах и подстроках нормализованного
  ключа.
- Physical comparator сам не конвертирует единицы; он рассчитывает метрики по
  значениям после существующей normalization pipeline.
- Для нескольких чисел выбирается наиболее похожая пара, поэтому часть
  дополнительных конфликтующих чисел может теряться.
- `SequenceMatcher` и char n-grams могут заметно увеличить CPU-время на полном
  LLM-датасете; это нужно измерить в screening.
- В агрегатах пока нет отдельных признаков для каждого имени атрибута. Это
  намеренно: иначе схема стала бы нестабильной для неизвестных ключей.
- Normalized category PR-AUC и автоматическая выгрузка FP/FN пока не
  реализованы.

## Критерий принятия идеи

Typed metrics имеет смысл оставлять, если:

- средняя дельта macro PR-AUC положительна на трёх seed;
- улучшение не объясняется одним seed или одной категорией;
- худшие категории не деградируют систематически;
- inference/training overhead остаётся приемлемым;
- ручной анализ показывает, что code/physical conflicts действительно
  исправляют ошибки, а не создают случайную корреляцию.

Если прироста нет, typed branch нужно оставить выключенным или сократить до
доказавших пользу групп, например только `CODE`.
