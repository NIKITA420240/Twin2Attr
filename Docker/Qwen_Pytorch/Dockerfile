ARG BASE_IMAGE=pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /work/Twin2Attr

COPY requirements.txt /tmp/qwen-pytorch-requirements.txt

RUN python -m pip install \
        --no-cache-dir \
        --requirement /tmp/qwen-pytorch-requirements.txt \
    && python -c "import torch, transformers; print(f'torch={torch.__version__} cuda={torch.version.cuda} transformers={transformers.__version__}')" \
    && rm /tmp/qwen-pytorch-requirements.txt

CMD ["python", "--version"]
