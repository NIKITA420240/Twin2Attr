# Оптимизации обучения Transformer

Этот документ — журнал всех изменений Twin2Attr, направленных на ускорение
обучения Transformer. Для каждой оптимизации здесь фиксируются статус,
конфигурация, принцип работы, ожидаемый эффект, ограничения и результат
benchmark. Оценки ускорения являются ориентировочными, зависят от GPU, состава
данных и длины последовательностей и не складываются напрямую.

Статусы:

- **Включено** — реализовано и активно в `configs/pipeline.yaml`.
- **Условно** — реализовано, но активируется только в подходящем режиме.
- **Запланировано** — кандидат, который ещё не реализован.

## Сводная таблица

| Оптимизация | Статус | Ожидаемый эффект | Влияние на качество |
|---|---|---:|---|
| BF16/FP16 mixed precision | Включено автоматически | 1.3–2.5× против FP32 | Обычно отсутствует |
| Partial fine-tuning top-3 | Включено | 20–50% для backward | Возможна разница против full fine-tuning |
| DataLoader pipeline | Включено | 5–20% при CPU bottleneck | Нет |
| Length bucketing | Включено | 5–25% | Нет |
| Ограниченные padding shapes | Включено | 3–15%, особенно с compile | Нет |
| `torch.compile` | Включено | 5–25% после прогрева | Нет |
| Sharded token cache | Включено | До устранения почти всей повторной токенизации | Нет |
| SDPA attention | Включено | 5–20% | Только численная погрешность |
| Fused AdamW | Включено на CUDA с fallback | 2–8% | Нет |
| Кэш профиля batch size | Запланировано | Убирает повторный autotuning | Нет |
| Throughput-based batch tuner | Запланировано | 5–20% | Нет при сохранении effective batch |
| Batch size по length bucket для eval/inference | Запланировано | 10–40% eval/inference | Нет |
| Удаление дублирующего final validation | Включено | До стоимости одного validation pass | Нет |
| Остальная оптимизация validation | Запланировано | 5–30% общего wall time | Нет |
| Автоподбор padding buckets | Запланировано | 3–15% | Нет |
| Кэшируемая аугментация | Запланировано | 5–25% при CPU bottleneck | Требует проверки |
| Token-level augmentation | Запланировано | 10–30% при CPU bottleneck | Средний риск; нельзя повреждать prompt |
| Асинхронное сохранение checkpoint | Запланировано | 1–10% общего wall time | Нет |
| Multi-GPU DDP | Запланировано | Около 1.7–1.9× на двух GPU | Нет |
| FP8 training | Эксперимент | 10–40% на H100/новее | Требует проверки стабильности |
| 8-bit optimizer states | Условно | 0–10%; может замедлить | Низкий–средний риск |
| Gradient checkpointing | Только при OOM | Сам замедляет | Нет |
| Уменьшение `max_length` | Эксперимент | 10–30% | Может снизить качество |
| Top-1/top-2 fine-tuning | Эксперимент | 10–25% | Может снизить качество |
| Progressive unfreezing | Эксперимент | 5–20% wall time | Требует проверки |
| `torch.compile` mode autotuning | Эксперимент | 0–15% поверх текущего | Нет |
| CUDA Graphs | Долгосрочно | 3–15% | Нет |
| FSDP/ZeRO, LoRA/QLoRA | Только при нехватке VRAM | Ускорение не гарантировано | LoRA требует quality ablation |
| Distillation | Долгосрочно | 2–5× | Может снизить качество |

## Реализованные оптимизации

### 1. Mixed precision BF16/FP16

Training автоматически использует BF16 на совместимом CUDA GPU, иначе FP16.
Это ускоряет матричные операции и уменьшает объём activation tensors.

Текущая политика находится в `src/match/models/transformer/training.py`:

```python
fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
```

На CPU оба режима отключены.

### 2. Partial fine-tuning верхних слоёв

По умолчанию обучаются только три верхних encoder layer и classifier head:

```yaml
model_description:
  transformer:
    train_last_n_layers: 3
```

Нижние слои и embeddings заморожены. Это сокращает backward-граф, память под
градиенты и optimizer state. Для полного fine-tuning используется `null`.

### 3. DataLoader pipeline

Текущая конфигурация:

```yaml
training_runtime:
  dataloader:
    num_workers: 4
    prefetch_factor: 2
    persistent_workers: true
    pin_memory: true
    non_blocking_transfer: true
```

Workers готовят следующие batch параллельно с GPU. Pinned memory и
non-blocking transfer позволяют перекрывать часть копирования CPU → GPU с
вычислениями.

### 4. Length bucketing и ограниченные padding shapes

```yaml
training_runtime:
  length_bucketing:
    enabled: true
    mega_batch_multiplier: 50
    padding_length_buckets: [32, 64, 96, 128]
```

Примеры сортируются по приблизительной длине только внутри случайно
перемешанных megabatch. Это уменьшает padding, не превращая всю эпоху в
глобально отсортированный проход.

Длина batch округляется до одного из четырёх значений. Это немного увеличивает
padding, но ограничивает число tensor shapes и повторных компиляций.

Выключение:

```yaml
training_runtime:
  length_bucketing:
    enabled: false
    padding_length_buckets: null
```

### 5. `torch.compile`

```yaml
training_runtime:
  torch_compile:
    enabled: true
    mode: reduce-overhead
```

PyTorch компилирует повторяющиеся участки training graph. Первые batch могут
быть медленнее из-за compilation warmup, поэтому эффект нужно измерять на
полной эпохе, а не на одном холодном вызове.

Выключение:

```yaml
training_runtime:
  torch_compile:
    enabled: false
```

### 6. Sharded token cache

```yaml
training_runtime:
  token_cache:
    enabled: true
    directory: .cache/tokenized_pairs
    build_chunk_size: 4096
```

Token IDs сохраняются на диск без padding и читаются через memory mapping.
Кэш разделяется по `data_source`, например `human`, `llm` и `hard_negative`.
Каждый source получает отдельный fingerprint, включающий:

- tokenizer и vocabulary;
- параметры pair encoding и `max_length`;
- тексты карточек;
- состав и порядок пар внутри source.

Если изменился только один источник, повторно токенизируется только его shard.
Labels и sample weights не входят в fingerprint и всегда берутся из текущего
dataset.

Поведение с аугментацией:

| Режим | Train cache |
|---|---|
| Без аугментации | Используется |
| Материализованный `attribute_shuffle` | Используется |
| Динамический `attribute_word_dropout` | Обходится |
| Validation | Используется всегда |

### 7. SDPA attention

SDPA включён при создании PyTorch-модели для train и inference:

```yaml
attention:
  implementation: sdpa
```

Доступные режимы:

- `sdpa` — явно использовать PyTorch Scaled Dot Product Attention;
- `eager` — полностью отключить SDPA;
- `auto` — оставить выбор библиотеке Transformers.

Training переключатель:

```yaml
model_description:
  transformer:
    training_runtime:
      attention:
        implementation: sdpa
```

Inference переключатель сохраняется в `solution.json`:

```json
{
  "attention": {
    "implementation": "sdpa"
  }
}
```

Для честного A/B старый speed-suite удалён. Новый
`configs/benchmark_tests/speed_optimizations.yaml` содержит только
`eager_reference` и `sdpa`. Реальный GPU benchmark пока не выполнен.

### 8. Performance logging

После каждой train-эпохи логируются и сохраняются в
`training_metadata.json`:

- `examples_per_second`;
- `real_tokens_per_second`;
- `padded_tokens_per_second`;
- `padding_efficiency`;
- peak CUDA memory.

Основной показатель для сравнения runtime-оптимизаций —
`real_tokens_per_second`. Итоговый PR-AUC проверяется отдельно, чтобы ускорение
не скрывало ухудшение качества.

### 9. Fused AdamW

Включается конфигурацией:

```yaml
training_runtime:
  optimizer:
    fused: true
```

На CUDA и при наличии аргумента `fused` в текущем PyTorch используется
`torch.optim.AdamW(..., fused=True)`. На другом device или в старом PyTorch
автоматически выбирается обычный AdamW. Остаётся выполнить GPU A/B
при одинаковых seed, dataset и effective batch.

### 10. Удаление дублирующего final validation

`Trainer` уже выполняет validation после каждой эпохи, сохраняет
`macro_pr_auc` лучшей checkpoint в `state.best_metric` и загружает эту
модель при `load_best_model_at_end=True`. Поэтому повторный полный
`trainer.evaluate()` после `trainer.train()` удалён. Если best metric нет,
training завершается явной ошибкой вместо сохранения непроверенной
модели.

## Очередь следующих оптимизаций

### P0. Throughput-based batch tuner

`auto_find_batch_size` умеет уменьшать batch после OOM, но не ищет максимум
throughput. Планируется короткий прогрев нескольких размеров, например
`64, 128, 192, 256, 320`, с выбором максимального `real_tokens_per_second`.

Результат нужно кэшировать по fingerprint окружения, модели и runtime:
GPU/VRAM, PyTorch/CUDA/Transformers, checkpoint, число обучаемых слоёв,
precision, SDPA, compile, optimizer, `max_length`, buckets, micro/effective batch.
Хранить следует `best_batch_size` и `safe_batch_size` с 10–15% запасом
VRAM. При точном совпадении fingerprint поиск можно пропустить.

Для validation и inference можно профилировать отдельный batch size для
каждого length bucket: короткие последовательности позволяют безопасно
увеличить batch и лучше загрузить GPU.

### P1. Оптимизация validation

Кандидаты:

- BF16 для полного validation;
- независимый больший `eval_batch_size`;
- потоковый расчёт метрик;
- отказ от хранения лишних logits;
- опциональный повторный full eval после `trainer.train()`.

### P1. Автоподбор padding buckets

Строить гистограмму фактических token lengths из token cache и выбирать
небольшой набор bucket по квантилям. Решение должно балансировать padding ratio
и число compiled graphs.

### P1. Кэшируемая динамическая аугментация

Возможные реализации:

1. Заранее создать несколько детерминированных аугментированных вариантов и
   кэшировать каждый по отдельному seed.
2. Применять dropout непосредственно к cached token IDs, сохраняя границы
   служебных частей prompt и атрибутов.

### P2. Multi-GPU DDP

Если модель помещается на одну GPU, использовать DDP. FSDP и gradient
checkpointing рассматривать только при нехватке памяти: они не являются
безусловными оптимизациями скорости.

### P2. Асинхронные checkpoint и FP8

Если профиль показывает паузы GPU на записи checkpoint, можно снимать
CPU snapshot и записывать его асинхронно. Это требует запаса CPU RAM и
атомарной финализации файла.

FP8 имеет смысл только на поддерживаемом GPU после исчерпания безопасных BF16
оптимизаций. Он требует отдельной проверки convergence и PR-AUC.

## Рекомендуемый порядок

1. Замерить breakdown wall time: train, validation, tokenization, checkpoint I/O.
2. Выполнить GPU A/B обычного и Fused AdamW.
3. Добавить кэш профиля и throughput-based batch tuner.
4. Подобрать отдельный batch по length bucket для validation/inference.
5. Ускорить validation через BF16, больший eval batch и streaming metrics.
6. Автоматически подобрать padding buckets по квантилям token lengths.
7. После профилирования решать, нужны ли async checkpoint, кэш аугментации или DDP.
8. FP8, CUDA Graphs, уменьшение context и архитектурные изменения оставить
   на отдельные quality ablation.

## Правила benchmark

Каждая новая runtime-оптимизация должна сравниваться с изолированным reference:

1. Один checkpoint, dataset, sample order и seed.
2. Одинаковые batch size, precision и sequence shapes, если они не являются
   предметом теста.
3. Отдельный warmup до измерения.
4. Не менее трёх measured runs.
5. Синхронизация CUDA перед замером времени.
6. Сравнение `real_tokens/s`, времени эпохи и peak VRAM.
7. Проверка PR-AUC или эквивалентности predictions.

## Журнал изменений

| Дата | Изменение | Проверка |
|---|---|---|
| 2026-08-30 | Length bucketing и buckets `[32, 64, 96, 128]` | Unit tests |
| 2026-08-30 | `torch.compile` в режиме `reduce-overhead` | Unit tests |
| 2026-08-30 | Sharded token cache по `data_source` | Unit tests, cache reuse test |
| 2026-08-30 | SDPA для train/inference с `eager`-выключателем | 249 tests |
| 2026-08-30 | Speed benchmark пересобран как eager vs SDPA | Configuration tests |
| 2026-08-30 | Fused AdamW с CUDA/PyTorch fallback | Unit tests; GPU A/B ожидается |
| 2026-08-30 | Удалён повторный full validation после train | Integration tests |
