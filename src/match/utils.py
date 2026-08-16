import os

os.environ.setdefault("TRITON_CACHE_DIR", "/dev/shm/.triton_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/.inductor_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")
os.environ.setdefault("RAYON_NUM_THREADS", "32")

import numpy as np
import pandas as pd

import json
import joblib
from loguru import logger
from tqdm import tqdm
from typing import List, Tuple, Optional

import torch
from sentence_transformers import CrossEncoder
from transformers import AutoModel, AutoTokenizer, logging as hf_logging

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


# Make a simple string for each product
def _create_product_text(row):

    name = str(row["name"])
    category = str(row["category"])
    try:
        attrs = json.loads(row["attributes"])
        attr_text = " ".join([f"{k}: {v}" for k, v in attrs.items()])
    except:
        attr_text = ""
    return f"Name: {name} Category: {category} Attributes: {attr_text}"


# Pack loading and making a big string into one function
def _load_and_process_data(data_path):

    df = pd.read_parquet(data_path)
    df["text"] = df.apply(_create_product_text, axis=1)
    return df


# Merge matches with processes items to create text pairs
def _build_pairs(match_df, data_df):

    id_to_text = dict(zip(data_df["id"], data_df["text"]))
    pairs, labels, pair_ids = [], [], []
    has_target = "target" in match_df.columns

    for _, row in match_df.iterrows():
        id1, id2 = row["id1"], row["id2"]
        text1, text2 = id_to_text.get(id1, ""), id_to_text.get(id2, "")
        if not text1 or not text2:
            continue
        pairs.append((text1, text2))
        pair_ids.append((id1, id2))
        if has_target:
            labels.append(row["target"])

    return pairs, pair_ids, (np.array(labels) if has_target else None)


# Utils — mind that you can use macs too
def _get_best_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# (Partially optimised) Cross-encoder loading with a few tweaks and whistles
def _load_cross_encoder_opt(
    model_path: str,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,  # bf16 is safe on Ampere+
    attn_implementation: str = "sdpa",  # fused attention kernel
    backend: str = "pytorch",  # "pytorch" | "onnx"
    use_compile: bool = False,
    compile_mode: str = "reduce-overhead",  # CUDA graphs
    onnx_provider: str = "cuda",  # "cuda" | "tensorrt"
):

    if device is None:
        device = _get_best_device()

    logger.info(
        f"[CE] device={device}  dtype={dtype}  attn={attn_implementation}  "
        f"backend={backend}  compile={use_compile}({compile_mode})  "
        f"onnx_provider={onnx_provider}"
    )

    if backend == "onnx":
        return _load_onnx_backend(model_path, device, onnx_provider)

    # We only need last_hidden_state[:, 0] (CLS), so the classification head is dead weight
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Temporarily suppress warnings about the dropped classifier head
    default_verbosity = hf_logging.get_verbosity()
    hf_logging.set_verbosity_error()

    model = (
        AutoModel.from_pretrained(
            model_path,
            dtype=dtype,
            attn_implementation=attn_implementation,
        )
        .to(device)
        .eval()
    )

    # Restore default logging level for the rest of the script
    hf_logging.set_verbosity(default_verbosity)

    if use_compile:
        # Compile the bare transformer — fullgraph=True enables CUDA graph
        model = torch.compile(model, mode=compile_mode, fullgraph=True)
        logger.info(
            "[CE] torch.compile active — first 2-3 batches will be slow (graph capture)"
        )

    # Lightweight container to preserve the original .model/.tokenizer interface
    class ModelWrapper:
        __slots__ = (
            "model",
            "tokenizer",
            "device",
            "max_seq_length",
            "_backend",
            "_use_compile",
            "_compile_mode",
        )

        def __init__(self):
            self.model = model
            self.tokenizer = tokenizer
            self.device = device
            self.max_seq_length = 256
            self._backend = "pytorch"
            self._use_compile = use_compile
            self._compile_mode = compile_mode

    return ModelWrapper()


# Experimental ONNX backend option, use at own risk;
# Not supported in baseline docker image
def _load_onnx_backend(model_path, device, provider):

    # from optimum.onnxruntime import ORTModelForFeatureExtraction
    # from transformers import AutoTokenizer

    provider_map = {
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }
    ep = provider_map.get(provider, "CUDAExecutionProvider")

    provider_options = {}
    if provider == "tensorrt":
        cache_dir = "/tmp/trt_cache_ce"
        os.makedirs(cache_dir, exist_ok=True)
        provider_options = {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_timing_cache_enable": True,
        }
        logger.info("[CE] TensorRT EP with fp16 — first run builds the engine (slow)")

    ort_model = ORTModelForFeatureExtraction.from_pretrained(
        model_path,
        export=True,
        provider=ep,
        provider_options=provider_options,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    class ORTWrapper:
        __slots__ = (
            "model",
            "tokenizer",
            "device",
            "max_seq_length",
            "_backend",
            "_use_compile",
            "_compile_mode",
        )

        def __init__(self):
            self.model = ort_model
            self.tokenizer = tokenizer
            self.device = device
            self.max_seq_length = 256
            self._backend = "onnx"
            self._use_compile = False
            self._compile_mode = ""

    return ORTWrapper()


# Warmup with a dummy forward pass so compilation / TRT cache is warm
def _warmup_model(model, max_seq_length=256):

    tokenizer = model.tokenizer
    device = model.device

    dummy = tokenizer(
        text="warmup query",
        text_pair="warmup document",
        padding="max_length",
        truncation=True,
        max_length=max_seq_length,
        return_tensors="pt",
    )
    dummy = {k: v.to(device, non_blocking=True) for k, v in dummy.items()}

    with torch.inference_mode():
        _ = model.model(**dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()
    logger.info("[CE] Warmup complete — kernels/engines cached.")


# V2 somewhat optimised [CLS] token embeddings extractor from a cross-encoder
def _extract_embeddings_opt_v2(
    model,
    pairs: List[Tuple[str, str]],
    batch_size: int = 128,
    pooling: str = "cls",
    use_bucketing: bool = True,
    pad_to_max: Optional[bool] = None,
    show_progress: bool = True,
    warmup: bool = True,
) -> np.ndarray:

    if pooling != "cls":
        raise ValueError("Only 'cls' pooling is implemented.")

    # Resolve config from the model wrapper
    backend = getattr(model, "_backend", "pytorch")
    use_compile = getattr(model, "_use_compile", False)
    compile_mode = getattr(model, "_compile_mode", "")
    tokenizer = model.tokenizer
    device = model.device
    max_seq_length = getattr(model, "max_seq_length", 256)

    # If torch.compile is using CUDA graphs, we MUST have static shapes
    if pad_to_max is None:
        pad_to_max = use_compile and compile_mode in ("reduce-overhead", "max-autotune")

    if pad_to_max:
        if use_bucketing:
            use_bucketing = False

    # Warmup
    if warmup and (use_compile or backend == "onnx"):
        _warmup_model(model, max_seq_length)

    # 1. Determine hidden dimension and PRE-ALLOCATE result array
    # This prevents the massive RAM spike caused by np.vstack at the end
    hidden_dim = model.model.config.hidden_size
    result = np.empty((len(pairs), hidden_dim), dtype=np.float32)

    # 2. Length bucketing using string length (Zero RAM overhead)
    # Instead of tokenizing the whole dataset (which OOMs RAM) use character count as a proxy for token length
    if use_bucketing and len(pairs) > batch_size:
        lengths = [len(p[0]) + len(p[1]) for p in pairs]
        sorted_indices = np.argsort(lengths)  # shortest → longest
        pairs = [pairs[i] for i in sorted_indices]
    else:
        sorted_indices = np.arange(len(pairs))

    # Padding strategy
    padding_strategy = "max_length" if pad_to_max else True

    # Inference loop
    iterator = range(0, len(pairs), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Extracting embeddings")

    if backend == "onnx":
        # ONNX Runtime path
        for i in iterator:
            batch_pairs = pairs[i : i + batch_size]
            features = tokenizer(
                text=[p[0] for p in batch_pairs],
                text_pair=[p[1] for p in batch_pairs],
                padding=padding_strategy,
                truncation=True,
                max_length=max_seq_length,
                return_tensors="pt",
            )
            outputs = model.model(**features)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()

            # Write directly to pre-allocated array in the original order
            batch_idx = 0
            for offset in range(i, min(i + batch_size, len(pairs))):
                orig_idx = sorted_indices[offset]
                result[orig_idx] = embeddings[batch_idx]
                batch_idx += 1

    else:
        # PyTorch path
        underlying_model = model.model
        underlying_model.eval()

        with torch.inference_mode():
            for i in iterator:
                batch_pairs = pairs[i : i + batch_size]

                # Tokenise with dynamic (or fixed) padding
                features = tokenizer(
                    text=[p[0] for p in batch_pairs],
                    text_pair=[p[1] for p in batch_pairs],
                    padding=padding_strategy,
                    truncation=True,
                    max_length=max_seq_length,
                    return_tensors="pt",
                )

                # H2D transfer
                features = {
                    k: v.to(device, non_blocking=True) for k, v in features.items()
                }

                # Forward
                outputs = underlying_model(**features)
                last_hidden_state = outputs.last_hidden_state
                embeddings = last_hidden_state[:, 0, :]

                # Cast to fp32 and move to CPU
                if embeddings.dtype != torch.float32:
                    embeddings = embeddings.float()

                # Write directly to pre-allocated array
                embeddings_np = embeddings.cpu().numpy()
                batch_idx = 0
                for offset in range(i, min(i + batch_size, len(pairs))):
                    orig_idx = sorted_indices[offset]
                    result[orig_idx] = embeddings_np[batch_idx]
                    batch_idx += 1

    return result


# a pipeline woirth a 1000 words instead of jupyter baseline
# not that useful within competition evaluation as there will be no target
def train_pipeline(data_path, match_path, model_path, output_model_path, device=None, batch_size=32):

    logger.info("Starting training pipeline")
    logger.info("[1/6] Loading and processing items data")
    data_df = _load_and_process_data(data_path)

    logger.info("[2/6] Building text pairs from matches data")
    match_df = pd.read_parquet(match_path)
    pairs, _, labels = _build_pairs(match_df, data_df)

    logger.info("[3/6] Loading cross-encoder")
    ce_model = _load_cross_encoder_opt(model_path, device=device)

    logger.info("[4/6] Extracting embeddings")
    embeddings = _extract_embeddings_opt_v2(ce_model, pairs, batch_size=batch_size)

    logger.info("[5/6] Training Logistic Regression classifier")
    clf_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=1234
                ),
            ),
            # ('classifier', MLPClassifier(hidden_layer_sizes=(256, 128), early_stopping=True, random_state=42))
        ]
    )
    clf_pipeline.fit(embeddings, labels)

    logger.info("[6/6] Saving model to {}", output_model_path)
    joblib.dump(clf_pipeline, output_model_path)
    logger.info("Training pipeline complete")

    return clf_pipeline


# (OPTIMIZERED) hey we can now fit a minilm-l12 CE model into our 20 minute time limit!
def predict_pipeline(
    data_path,
    match_path,
    model_path,
    logreg_path,
    output_csv_path,
    device=None,
    batch_size=32,
    backend="pytorch",
):

    logger.info("Starting prediction pipeline")
    logger.info("[1/6] Loading and processing new product data")
    data_df = _load_and_process_data(data_path)

    logger.info("[2/6] Building text pairs from match data")
    match_df = pd.read_parquet(match_path)
    pairs, pair_ids, _ = _build_pairs(match_df, data_df)

    logger.info("[3/6] Loading cross-encoder")
    ce_model = _load_cross_encoder_opt(model_path, device=device, backend=backend, use_compile=False)

    logger.info("[4/6] Extracting embeddings with V2")
    embeddings = _extract_embeddings_opt_v2(ce_model, pairs, batch_size=batch_size, warmup=True)

    logger.info("[5/6] Loading saved classifier from {}", logreg_path)
    clf_pipeline = joblib.load(logreg_path)
    predictions = clf_pipeline.predict_proba(embeddings)[:, 1]

    logger.info("[6/6] Saving predictions to {}", output_csv_path)
    results_df = pd.DataFrame(
        {
            "id1": [p[0] for p in pair_ids],
            "id2": [p[1] for p in pair_ids],
            "predict": predictions,
        }
    )
    results_df.to_csv(output_csv_path, index=False)

    logger.info("Prediction pipeline complete")
    return results_df
