# Docker-образы Twin2Attr

В проекте используются отдельные evaluator-образы для ONNX Runtime CUDA,
Native TensorRT, PyTorch CUDA INT8 и гибридной упаковки FP16-весов. Образы
содержат только runtime-зависимости и, в одном случае, вынесенную из submission
часть весов. Точка входа (`python -u run.py`), код решения и оставшиеся модельные
артефакты поставляются в submission ZIP.

## Опубликованные образы

Публичный репозиторий: `c0ffee103/twin2attr-runtime` на Docker Hub.
Состояние тегов на 30 августа 2026 года:

| Тег | Назначение | Размер в registry | Digest |
| --- | --- | ---: | --- |
| `qwen3-trt10.9-ort1.23.2` | ONNX Runtime CUDA и Native TensorRT | 10,176,857,712 B (9.48 GiB) | `sha256:c6ad55102c43a8142a6e70020348d1dd0c45076b7d5a41844733278a189fa8fc` |
| `qwen3-pytorch-int8` | PyTorch CUDA с weight-only INT8-весами в submission | 6,746,708,406 B (6.28 GiB) | `sha256:4c4f73508635502d85c2ed4f51f1809a54649aba2b34ad77276bd2c87bfd523e` |
| `qwen3-fp16-hybrid` | ORT/TensorRT runtime и часть FP16 ONNX-весов внутри образа | 11,999,503,318 B (11.18 GiB) | `sha256:a042b8d75ca189da58512469142acdcfc070e2095186c825aa17af3e1260f478` |

Размеры выше возвращены Docker Hub API и не равны объёму слоёв после их
распаковки на машине evaluator.

## Базовый ORT и TensorRT runtime

Файл: `Dockerfile.runtime`.

Образ наследуется от `odsai/ecup26-matching-baseline:1.0` и добавляет:

- `onnxruntime-gpu==1.23.2`;
- `tensorrt-cu12==10.9.0.34`;
- фиксированные версии Polars, pymorphy3, joblib, Loguru, Pint и orjson;
- smoke-проверку импорта TensorRT и доступных ONNX Runtime providers.

Флаг `--break-system-packages` необходим из-за системного Python внутри
baseline-контейнера. Этот образ используется как для `onnxruntime` с CUDA
provider, так и для backend `tensorrt`, где ONNX служит исходным графом для
построения engine.

Сборка и публикация:

```bash
docker build --pull -f Dockerfile.runtime \
  -t c0ffee103/twin2attr-runtime:qwen3-trt10.9-ort1.23.2 .
docker push c0ffee103/twin2attr-runtime:qwen3-trt10.9-ort1.23.2
```

Проверка на Linux-хосте с NVIDIA Container Toolkit:

```bash
docker run --rm --gpus all \
  c0ffee103/twin2attr-runtime:qwen3-trt10.9-ort1.23.2 \
  python -c "import onnxruntime as ort, tensorrt as trt; print(ort.get_available_providers()); print(trt.__version__)"
```

## PyTorch CUDA INT8 runtime

Файл: `Dockerfile.cuda-int8`.

Это облегчённый runtime без ONNX Runtime и TensorRT. Он использует CUDA-сборку
PyTorch из baseline-образа, устанавливает preprocessing-зависимости и проверяет
наличие CUDA, Transformers, Accelerate и Safetensors. Квантованные weight-only
INT8-веса не встроены в образ: они остаются в submission ZIP. Такое разделение
позволяет независимо обновлять код/веса и не пересобирать runtime.

```bash
docker build --pull -f Dockerfile.cuda-int8 \
  -t c0ffee103/twin2attr-runtime:qwen3-pytorch-int8 .
docker push c0ffee103/twin2attr-runtime:qwen3-pytorch-int8
```

Проверка:

```bash
docker run --rm --gpus all \
  c0ffee103/twin2attr-runtime:qwen3-pytorch-int8 \
  python -c "import torch, transformers, accelerate, safetensors; print(torch.__version__, torch.version.cuda, transformers.__version__)"
```

## Гибридный FP16-образ

Файлы:

- `Dockerfile.fp16-hybrid`;
- `build_submission/fp16_image_weights_manifest.json`;
- правила выборочного Docker context в `.dockerignore`.

Образ наследуется от `qwen3-trt10.9-ort1.23.2` и переносит 33 самых крупных
внешних файла FP16 ONNX в `/opt/twin2attr/fp16-weights`. Их суммарный размер —
2,370,380,800 байт (2.208 GiB). Submission builder может исключить эти файлы из
ZIP и записать ссылку на каталог внутри образа. Manifest фиксирует имя и точный
размер каждого файла; Docker build завершается ошибкой при любом несовпадении.

Собирать образ нужно из корня репозитория после подготовки каталога
`models/twin2attr/qwen3-reranker-4b-fp16/onnx`:

```bash
docker build -f Dockerfile.fp16-hybrid \
  -t c0ffee103/twin2attr-runtime:qwen3-fp16-hybrid .
docker push c0ffee103/twin2attr-runtime:qwen3-fp16-hybrid
```

`.dockerignore` использует allowlist и передаёт Docker daemon только Dockerfile,
manifest и перечисленные 33 файла весов. Остальная модель и рабочее дерево в
контекст сборки не попадают.

## Использование образа в submission

Тег задаётся в конфигурации:

```yaml
submission:
  custom_image: c0ffee103/twin2attr-runtime:qwen3-trt10.9-ort1.23.2
```

Для воспроизводимого production-сабмита вместо изменяемого тега можно указывать
digest:

```text
c0ffee103/twin2attr-runtime@sha256:c6ad55102c43a8142a6e70020348d1dd0c45076b7d5a41844733278a189fa8fc
```

## GitHub Container Registry

Workflow `.github/workflows/build-evaluator-image.yml` собирает
`Dockerfile.runtime` вручную (`workflow_dispatch`) или при изменениях runtime на
ветке `qwen-3`. Он публикует Linux AMD64-образы в:

```text
ghcr.io/nikita420240/twin2attr-runtime:qwen3-trt10.9-ort1.23.2
ghcr.io/nikita420240/twin2attr-runtime:qwen3-<git-sha>
```

Текущие submission-конфигурации используют Docker Hub, а GHCR остаётся
альтернативным CI registry.

## Локальная среда сборки

На Windows системное восстановление через `sfc`/`DISM` потребовало повышенных
прав, поэтому для Docker была подготовлена Linux VM в VirtualBox. Параметры VM:

- 6 GiB RAM;
- 4 vCPU;
- динамический диск 50 GiB;
- Docker Engine внутри Linux-гостя.

VM использовалась для Linux AMD64 build, проверки слоёв, `docker login` и
публикации в Docker Hub. VirtualBox не предоставляет используемый нами NVIDIA
GPU passthrough, поэтому CUDA, ONNX Runtime CUDA и TensorRT engine нельзя полноценно
проверить в этой VM. GPU-инференс проверяется уже в evaluator; локально доступны
структурные проверки архивов, импортов и состава образов.

## Порядок обновления

1. Изменить соответствующий Dockerfile и закрепить версии зависимостей.
2. Собрать образ из корня репозитория.
3. Выполнить smoke-проверку; для CUDA runtime — на машине с NVIDIA GPU.
4. Опубликовать новый неизменяемый тег, не перезаписывая ранее проверенный.
5. Получить digest через `docker inspect` или registry API.
6. Обновить эту таблицу и `submission.custom_image`.
7. Пересобрать submission ZIP и проверить `metadata.json`, CRC и SHA-256.
