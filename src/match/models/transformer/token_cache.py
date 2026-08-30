"""Compact reusable tokenization cache for Transformer training datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from ...pair_encoding import PairEncodingCollator, PreencodedPreparedPair
from ...prepare_data import PreparedCard, PreparedPair


_CACHE_SCHEMA_VERSION = 1
_INPUT_IDS_FILE = "input_ids.bin"
_TOKEN_TYPE_IDS_FILE = "token_type_ids.bin"
_OFFSETS_FILE = "offsets.npy"
_METADATA_FILE = "metadata.json"


def _hash_text(digest: Any, value: object) -> None:
    encoded = str(value).encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)


def _hash_card(digest: Any, card: PreparedCard) -> None:
    _hash_text(digest, card.item_id)
    _hash_text(digest, card.name)
    _hash_text(digest, card.category)
    digest.update(len(card.attributes).to_bytes(8, "little"))
    for key, value in card.attributes:
        _hash_text(digest, key)
        _hash_text(digest, value)


def token_cache_fingerprint(
    pairs: Sequence[PreparedPair],
    tokenizer: PreTrainedTokenizerBase,
    collator: PairEncodingCollator,
) -> str:
    """Hash every input that can affect the unpadded token ids."""
    digest = hashlib.sha256()
    settings = {
        "schema_version": _CACHE_SCHEMA_VERSION,
        "tokenizer_class": type(tokenizer).__qualname__,
        "tokenizer_name": str(getattr(tokenizer, "name_or_path", "")),
        "tokenizer_size": len(tokenizer),
        "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
        "profile": collator.profile,
        "max_length": collator.max_length,
        "use_field_tokens": collator.use_field_tokens,
        "max_attribute_value_chars": collator.max_attribute_value_chars,
        "max_attribute_value_tokens": collator.max_attribute_value_tokens,
    }
    digest.update(
        json.dumps(settings, ensure_ascii=False, sort_keys=True).encode("utf-8")
    )
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    if backend_tokenizer is not None and hasattr(backend_tokenizer, "to_str"):
        _hash_text(digest, backend_tokenizer.to_str())
    elif hasattr(tokenizer, "get_vocab"):
        vocabulary = sorted(tokenizer.get_vocab().items())
        digest.update(
            json.dumps(vocabulary, ensure_ascii=False).encode("utf-8")
        )
    seen_cards: set[tuple[type, str]] = set()
    for pair in pairs:
        for card in (pair.left, pair.right):
            card_key = (type(card.item_id), str(card.item_id))
            if card_key not in seen_cards:
                seen_cards.add(card_key)
                _hash_card(digest, card)
        _hash_text(digest, pair.left.item_id)
        _hash_text(digest, pair.right.item_id)
        digest.update(bytes((pair.preserve_attribute_order, pair.skip_oversized_attributes)))
    return digest.hexdigest()


class TokenCacheDataset(Dataset[PreencodedPreparedPair]):
    """Read compact unpadded encodings through memory-mapped arrays."""

    def __init__(self, pairs: Sequence[PreparedPair], cache_directory: Path) -> None:
        self.pairs = pairs
        metadata = json.loads(
            (cache_directory / _METADATA_FILE).read_text(encoding="utf-8")
        )
        if int(metadata["rows"]) != len(pairs):
            raise ValueError("token cache row count does not match the dataset")
        self.offsets = np.load(
            cache_directory / _OFFSETS_FILE,
            mmap_mode="r",
        )
        self.input_ids = np.memmap(
            cache_directory / _INPUT_IDS_FILE,
            mode="r",
            dtype=np.uint32,
        )
        token_type_path = cache_directory / _TOKEN_TYPE_IDS_FILE
        self.token_type_ids = (
            np.memmap(token_type_path, mode="r", dtype=np.uint8)
            if bool(metadata["has_token_type_ids"])
            else None
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> PreencodedPreparedPair:
        start = int(self.offsets[index])
        stop = int(self.offsets[index + 1])
        inputs = {
            "input_ids": self.input_ids[start:stop].astype(np.int64).tolist(),
            "attention_mask": [1] * (stop - start),
        }
        if self.token_type_ids is not None:
            inputs["token_type_ids"] = (
                self.token_type_ids[start:stop].astype(np.int64).tolist()
            )
        return PreencodedPreparedPair(self.pairs[index], inputs)


class ShardedTokenCacheDataset(Dataset[PreencodedPreparedPair]):
    """Present independently cached source shards in original row order."""

    def __init__(
        self,
        shards: Sequence[TokenCacheDataset],
        shard_indices: np.ndarray,
        local_indices: np.ndarray,
    ) -> None:
        if shard_indices.shape != local_indices.shape:
            raise ValueError("token cache shard mappings must have equal shapes")
        self.shards = tuple(shards)
        self.shard_indices = shard_indices
        self.local_indices = local_indices

    def __len__(self) -> int:
        return int(self.shard_indices.size)

    def __getitem__(self, index: int) -> PreencodedPreparedPair:
        shard = self.shards[int(self.shard_indices[index])]
        return shard[int(self.local_indices[index])]


def _cache_is_complete(directory: Path, *, rows: int) -> bool:
    metadata_path = directory / _METADATA_FILE
    offsets_path = directory / _OFFSETS_FILE
    input_ids_path = directory / _INPUT_IDS_FILE
    if not all(path.is_file() for path in (metadata_path, offsets_path, input_ids_path)):
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        offsets = np.load(offsets_path, mmap_mode="r")
        token_count = int(offsets[-1])
        token_type_path = directory / _TOKEN_TYPE_IDS_FILE
        return (
            int(metadata.get("schema_version", -1)) == _CACHE_SCHEMA_VERSION
            and int(metadata.get("rows", -1)) == rows
            and offsets.shape == (rows + 1,)
            and token_count == int(metadata.get("tokens", -1))
            and input_ids_path.stat().st_size == token_count * 4
            and (
                not bool(metadata.get("has_token_type_ids", False))
                or (
                    token_type_path.is_file()
                    and token_type_path.stat().st_size == token_count
                )
            )
        )
    except (OSError, ValueError, KeyError, IndexError, json.JSONDecodeError):
        return False


def _build_cache(
    pairs: Sequence[PreparedPair],
    collator: PairEncodingCollator,
    destination: Path,
    *,
    chunk_size: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}-", dir=destination.parent)
    )
    offsets = np.zeros(len(pairs) + 1, dtype=np.uint64)
    token_count = 0
    has_token_type_ids: bool | None = None
    try:
        with (temporary / _INPUT_IDS_FILE).open("wb") as input_file, (
            temporary / _TOKEN_TYPE_IDS_FILE
        ).open("wb") as token_type_file:
            for start in range(0, len(pairs), chunk_size):
                chunk = pairs[start : start + chunk_size]
                encoded_chunk = collator.encode_pairs(chunk)
                for relative_index, encoded in enumerate(encoded_chunk):
                    input_ids = np.asarray(encoded["input_ids"], dtype=np.uint32)
                    attention_mask = encoded.get("attention_mask")
                    if attention_mask is not None and not all(attention_mask):
                        raise ValueError("unpadded token cache requires a full attention mask")
                    input_ids.tofile(input_file)
                    current_has_token_types = "token_type_ids" in encoded
                    if has_token_type_ids is None:
                        has_token_type_ids = current_has_token_types
                    elif has_token_type_ids != current_has_token_types:
                        raise ValueError("token_type_ids must be present for every cache row")
                    if current_has_token_types:
                        token_type_ids = np.asarray(
                            encoded["token_type_ids"], dtype=np.uint8
                        )
                        if token_type_ids.shape != input_ids.shape:
                            raise ValueError("token_type_ids must align with input_ids")
                        token_type_ids.tofile(token_type_file)
                    token_count += int(input_ids.size)
                    offsets[start + relative_index + 1] = token_count
        if not has_token_type_ids:
            (temporary / _TOKEN_TYPE_IDS_FILE).unlink()
        np.save(temporary / _OFFSETS_FILE, offsets, allow_pickle=False)
        metadata = {
            "schema_version": _CACHE_SCHEMA_VERSION,
            "rows": len(pairs),
            "tokens": token_count,
            "has_token_type_ids": bool(has_token_type_ids),
        }
        (temporary / _METADATA_FILE).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        try:
            os.rename(temporary, destination)
        except FileExistsError:
            if not _cache_is_complete(destination, rows=len(pairs)):
                raise
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_or_build_token_cache(
    pairs: Sequence[PreparedPair],
    tokenizer: PreTrainedTokenizerBase,
    collator: PairEncodingCollator,
    cache_root: str | Path,
    *,
    split_name: str,
    chunk_size: int,
) -> TokenCacheDataset:
    if not pairs:
        raise ValueError("cannot cache an empty pair dataset")
    if chunk_size < 1:
        raise ValueError("token cache chunk_size must be positive")
    fingerprint = token_cache_fingerprint(pairs, tokenizer, collator)
    directory = Path(cache_root) / f"{split_name}-{fingerprint[:20]}"
    if _cache_is_complete(directory, rows=len(pairs)):
        logger.info(
            "Using Transformer token cache: split={}, rows={}, path={!s}",
            split_name,
            len(pairs),
            directory,
        )
    else:
        if directory.exists():
            shutil.rmtree(directory)
        logger.info(
            "Building Transformer token cache: split={}, rows={}, path={!s}",
            split_name,
            len(pairs),
            directory,
        )
        _build_cache(pairs, collator, directory, chunk_size=chunk_size)
    return TokenCacheDataset(pairs, directory)


def _safe_shard_name(value: object) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-._")
    return (normalized or "dataset")[:64]


def load_or_build_sharded_token_cache(
    pairs: Sequence[PreparedPair],
    shard_keys: Sequence[object],
    tokenizer: PreTrainedTokenizerBase,
    collator: PairEncodingCollator,
    cache_root: str | Path,
    *,
    split_name: str,
    chunk_size: int,
) -> ShardedTokenCacheDataset:
    """Cache each source independently and retain the combined row order."""
    if len(shard_keys) != len(pairs):
        raise ValueError("token cache shard keys must align with pairs")
    grouped_indices: dict[str, list[int]] = {}
    for index, key in enumerate(shard_keys):
        grouped_indices.setdefault(str(key), []).append(index)
    if not grouped_indices:
        raise ValueError("cannot cache an empty sharded pair dataset")

    shard_indices = np.empty(len(pairs), dtype=np.uint16)
    local_indices = np.empty(len(pairs), dtype=np.uint64)
    shards: list[TokenCacheDataset] = []
    for shard_index, (key, indices) in enumerate(grouped_indices.items()):
        if shard_index > np.iinfo(np.uint16).max:
            raise ValueError("token cache has too many source shards")
        shard_pairs = [pairs[index] for index in indices]
        shard = load_or_build_token_cache(
            shard_pairs,
            tokenizer,
            collator,
            cache_root,
            split_name=f"{split_name}-{_safe_shard_name(key)}",
            chunk_size=chunk_size,
        )
        shards.append(shard)
        for local_index, global_index in enumerate(indices):
            shard_indices[global_index] = shard_index
            local_indices[global_index] = local_index
    logger.info(
        "Prepared sharded Transformer token cache: split={}, rows={}, shards={}",
        split_name,
        len(pairs),
        {key: len(indices) for key, indices in grouped_indices.items()},
    )
    return ShardedTokenCacheDataset(shards, shard_indices, local_indices)


__all__ = [
    "TokenCacheDataset",
    "ShardedTokenCacheDataset",
    "load_or_build_sharded_token_cache",
    "load_or_build_token_cache",
    "token_cache_fingerprint",
]
