# Оптимизации скорости обучения

Этот документ хранит список уже используемых и потенциальных оптимизаций
обучения Twin2Attr. Оценки ускорения ориентировочные: они зависят от GPU,
версий PyTorch/CUDA, длины последовательностей и состава датасета. Эффекты
разных оптимизаций могут перекрываться, поэтому их нельзя складывать напрямую.

## Что уже используется

| Оптимизация | Статус | Назначение |
|---|---|---|
| BF16/FP16 mixed precision | Реализовано | Ускорение tensor-core операций и уменьшение activation memory |
| Partial fine-tuning | Реализовано | Обучение только верхних Transformer layers и head |
| DataLoader workers/prefetch/pinned memory | Реализовано | Перекрытие CPU-подготовки и передачи данных с работой GPU |
| Length bucketing | Реализовано | Уменьшение padding внутри batch |
| Ограниченные padding buckets | Реализовано | Стабильные tensor shapes для GPU и `torch.compile` |
| `torch.compile` | Реализовано | Компиляция и объединение операций training graph |
| Sharded token cache | Реализовано | Исключение повторной токенизации неизменившихся источников данных |
| SDPA | Реализовано | Оптимизированное вычисление attention |
| Performance logging | Реализовано | Измерение examples/s, tokens/s, padding efficiency и peak VRAM |
| Fused AdamW | Реализовано, выключено по умолчанию | Уменьшение overhead optimizer step на CUDA |
| Отказ от повторного final eval | Реализовано | Использование метрики уже выбранного лучшего checkpoint |

## Что ещё можно сделать

| Приоритет | Оптимизация | Что ускоряет | Потенциальный эффект | Память | Риск для качества | Сложность | Когда имеет смысл |
|---|---|---|---:|---|---|---|---|
| P0 | Накопительный batch-profile cache | Старт повторных запусков и autotuner | Убирает повторный полный поиск; косвенно 5–20% через лучший batch | Без изменений | Нет | Средняя | Среда обучения и проверки совпадает |
| P0 | Throughput-based batch tuner | Загрузку GPU | 5–20% | Автоматически выбирает безопасный предел | Нет при сохранении effective batch | Средняя | После появления стабильного benchmark |
| P0 | Batch size по length bucket для eval/inference | Offline validation и проверку | 10–40% eval/inference | Эффективнее использует VRAM | Нет | Средняя | Длины входов сильно различаются |
| P0 | Оптимизация validation | Общее wall time обучения | 5–30% wall time | Может уменьшить peak eval memory | Нет | Низкая–средняя | Validation занимает заметную долю эпохи |
| P1 | Автоподбор padding buckets | Attention и compiled shapes | 3–15% | Меньше padding | Нет | Средняя | `padding_efficiency` ниже 90–95% |
| P1 | Кэшируемые варианты аугментации | CPU preprocessing и tokenization | 5–25% при CPU bottleneck | Требует больше диска | Низкий; проверить разнообразие | Средняя | Используется динамический `attribute_word_dropout` |
| P1 | Token-level augmentation | CPU preprocessing и tokenization | 10–30% при CPU bottleneck | Почти без изменений | Средний: нельзя повреждать prompt | Высокая | Материализованные варианты слишком велики |
| P1 | BF16 full evaluation | Validation | 5–20% validation | Ниже | Обычно отсутствует | Низкая | GPU поддерживает BF16 |
| P1 | Потоковый расчёт validation metrics | Validation и перенос logits | 2–10% wall time | Ниже | Нет | Средняя | Validation-набор большой |
| P1 | Асинхронное сохранение checkpoint | Паузы на запись модели | 1–10% wall time | Нужна CPU RAM для snapshot | Нет | Средняя | Сохранение 1B checkpoint останавливает GPU |
| P1 | DDP | Обучение на нескольких GPU | Около 1.7–1.9× на двух GPU | Полная копия модели на каждой GPU | Нет | Средняя | Доступны несколько одинаковых GPU |
| P1 | Настройка DDP communication buckets | Синхронизацию gradients | 2–10% поверх DDP | Небольшое влияние | Нет | Средняя | Только после профилирования multi-GPU |
| P2 | FP8 training | Matmul и память на H100/новее | 10–40% | Существенно ниже | Средний: проверить стабильность | Высокая | BF16-оптимизации исчерпаны |
| P2 | 8-bit optimizer states | Optimizer memory | Обычно 0–10%; иногда медленнее | Существенно ниже | Низкий–средний | Средняя | Optimizer state ограничивает batch |
| P2 | Gradient checkpointing | Позволяет увеличить batch | Сам замедляет; возможен итоговый выигрыш через больший batch | Сильно ниже activation memory | Нет | Низкая | Только при OOM |
| P2 | Уменьшение `max_length` 128 → 96 | Attention и MLP | 10–30% | Ниже | Возможна потеря важных данных | Низкая | После анализа truncation и PR-AUC |
| P2 | Top-1/top-2 вместо top-3 fine-tuning | Backward | 10–25% | Ниже | Возможна потеря качества | Низкая | Как отдельная quality ablation |
| P2 | Progressive unfreezing | Первые эпохи или шаги | 5–20% wall time | Ниже в начале | Средний | Средняя | Top-3 недостаточно, а full tune слишком дорог |
| P2 | `torch.compile` mode autotuning | Compiled graph | 0–15% поверх текущего режима | Зависит от режима | Нет | Средняя | Сравнить `reduce-overhead` и `max-autotune` |
| P2 | CUDA Graphs со статическими shapes | Kernel launch overhead | 3–15% | Требует статические buffers | Нет | Высокая | Shapes стабилизированы length buckets |
| P3 | FSDP/ZeRO | Модели, не помещающиеся на одну GPU | Не гарантирует ускорение | Сильно снижает GPU memory | Нет | Высокая | Текущий режим перестал помещаться |
| P3 | LoRA/QLoRA | Память trainable parameters | Не гарантирует ускорение текущего top-3 | Существенно ниже | Средний | Средняя | Нужно разморозить больше слоёв |
| P3 | Distillation в меньший reranker | Train и inference | 2–5× | Существенно ниже | Возможна потеря качества | Высокая | Долгосрочная архитектурная оптимизация |

## Рекомендуемый порядок

1. Выполнить GPU A/B обычного и Fused AdamW.
2. Реализовать performance-profile cache.
3. Добавить throughput-based batch tuner, использующий сохранённую статистику.
4. Подбирать отдельный безопасный batch для каждого length bucket при
   validation и inference.
5. Измерить долю времени validation и убрать лишние расходы.
6. Автоматически подобрать padding buckets по распределению token lengths.
7. Ускорить динамическую аугментацию.
8. Добавить DDP при появлении второй GPU.
9. После безопасных оптимизаций исследовать FP8, уменьшение контекста и
   архитектурные изменения.

## Какие данные сохранять для batch-profile cache

Профиль должен разделять training, validation и inference и содержать:

| Группа | Поля |
|---|---|
| Окружение | GPU, объём VRAM, PyTorch, CUDA, Transformers |
| Модель | Hash checkpoint, head, число обучаемых слоёв |
| Runtime | BF16/FP16, SDPA/eager, `torch.compile`, optimizer |
| Формы | `max_length`, padding buckets, квантили длин p50/p90/p95/p99 |
| Batch | Micro batch, gradient accumulation, число GPU, effective batch |
| Результат | Real tokens/s, examples/s, step latency, peak VRAM, OOM status |

Для каждого профиля следует хранить два результата:

- `best_batch_size` — максимальный измеренный throughput;
- `safe_batch_size` — немного меньший batch с запасом 10–15% VRAM.

В проверяющей системе нужно использовать `safe_batch_size`. Если fingerprint
полностью совпал, повторный поиск можно пропустить. При частичном совпадении
сохранённый batch используется только как стартовая точка короткой проверки.

## Правила benchmark

Каждая новая runtime-оптимизация сравнивается с изолированным reference:

1. Одинаковые checkpoint, dataset, seed и порядок примеров.
2. Одинаковые precision, shapes и effective batch, если они не являются
   предметом теста.
3. Compilation и CUDA warmup выполняются до измерения.
4. Перед началом и завершением замера вызывается CUDA synchronization.
5. Выполняется не менее трёх measured runs.
6. Сравниваются `real_tokens/s`, время эпохи и peak VRAM.
7. Проверяется PR-AUC либо эквивалентность predictions.
