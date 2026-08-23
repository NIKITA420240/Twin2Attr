"""Запускает CatBoost-решение для матчинга товаров."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "catboost.cbm"
DIGIT_RE, ALPHA_RE = re.compile(r"\d"), re.compile(r"[^\W\d_]", re.UNICODE)
SCORE_COEFFICIENTS = np.asarray(
    [0.2140903622, -0.3722740710, -0.1632695198, -0.1526790708, -0.7561550736, -0.1195358336],
    dtype=np.float32,
)


def _load_catboost():
    """Загружает CatBoost из образа или вложенного Linux-wheel."""

    try:
        from catboost import CatBoostClassifier
        return CatBoostClassifier
    except ModuleNotFoundError:
        tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        wheels = sorted((ROOT / "wheels").glob(f"catboost-*-{tag}-{tag}-*.whl"))
        if not wheels:
            raise RuntimeError(f"No bundled CatBoost wheel for Python {sys.version_info.major}.{sys.version_info.minor}")
        target = ROOT / ".catboost_vendor"
        if not (target / "catboost").exists():
            target.mkdir(exist_ok=True)
            with zipfile.ZipFile(wheels[-1]) as archive:
                archive.extractall(target)
        sys.path.insert(0, str(target))
        from catboost import CatBoostClassifier
        return CatBoostClassifier


CatBoostClassifier = _load_catboost()


def _pair_values(output: pd.DataFrame, name: str, left: np.ndarray, right: np.ndarray) -> None:
    """Добавляет медиану и знаковую разность значений двух карточек."""

    output[f"{name}_median"] = ((left + right) / 2).astype(np.float32)
    output[f"{name}_difference_1_2"] = (left - right).astype(np.float32)


def _minmax_ratio(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Считает отношение меньшего значения к большему."""

    largest = np.maximum(left, right)
    return np.divide(np.minimum(left, right), largest, out=np.ones_like(largest), where=largest != 0)


def _set_metrics(left: set[str], right: set[str]) -> tuple[float, float, float, float]:
    """Возвращает common, Jaccard, Dice и containment двух множеств."""

    common = len(left & right)
    union, total, smallest = len(left | right), len(left) + len(right), min(len(left), len(right))
    return (common, common / union if union else 1, 2 * common / total if total else 1,
            common / smallest if smallest else 1)


def _overlaps(left: pd.Series, right: pd.Series, kinds: tuple[str, ...]) -> np.ndarray:
    """Считает common, Jaccard, Dice и containment для наборов токенов."""

    rows = []
    for left_text, right_text in zip(left, right):
        base_left = set(left_text.casefold().split())
        base_right = set(right_text.casefold().split())
        numeric_left = numeric_right = None
        values = []
        for kind in kinds:
            if kind != "all" and numeric_left is None:
                numeric_left = {token for token in base_left if DIGIT_RE.search(token)}
                numeric_right = {token for token in base_right if DIGIT_RE.search(token)}
            if kind == "numeric":
                left_tokens, right_tokens = numeric_left, numeric_right
            elif kind == "model_code":
                left_tokens = {token for token in numeric_left if ALPHA_RE.search(token)}
                right_tokens = {token for token in numeric_right if ALPHA_RE.search(token)}
            else:
                left_tokens, right_tokens = base_left, base_right
            values.extend(_set_metrics(left_tokens, right_tokens))
        rows.append(values)
    return np.asarray(rows, dtype=np.float32)


def _json_dict(value: object) -> dict[str, str]:
    """Безопасно преобразует JSON-атрибуты в нормализованный словарь."""

    try:
        parsed = json.loads(str(value or "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    result = {}
    for key, item in parsed.items():
        key = " ".join(str(key).casefold().split())
        raw = item if isinstance(item, str) else json.dumps(item, ensure_ascii=False, sort_keys=True)
        result[key] = " ".join(str(raw).casefold().split())
    return result


def _name_char_metrics(left: pd.Series, right: pd.Series) -> np.ndarray:
    """Считает сходство названий по множествам символьных триграмм."""

    rows = []
    for left_name, right_name in zip(left, right):
        names = []
        for value in (left_name, right_name):
            normalized = "".join(char for char in value.casefold() if char.isalnum())
            names.append({normalized[index:index + 3] for index in range(max(1, len(normalized) - 2))}
                         if normalized else set())
        rows.append(_set_metrics(names[0], names[1]))
    return np.asarray(rows, dtype=np.float32)


def _structured_features(output: pd.DataFrame, left: pd.Series, right: pd.Series) -> None:
    """Добавляет признаки структуры JSON-атрибутов."""

    rows = []
    article_keys = ("артикул", "article", "sku", "код товара", "код модели")
    brand_keys = ("бренд", "brand", "производитель")
    for left_raw, right_raw in zip(left, right):
        left_dict, right_dict = _json_dict(left_raw), _json_dict(right_raw)
        left_keys, right_keys = set(left_dict), set(right_dict)
        common_keys = left_keys & right_keys
        exact = sum(left_dict[key] == right_dict[key] for key in common_keys)
        conflicts = len(common_keys) - exact
        left_articles = {value for key, value in left_dict.items() if any(x in key for x in article_keys)}
        right_articles = {value for key, value in right_dict.items() if any(x in key for x in article_keys)}
        left_brand = next((value for key, value in left_dict.items() if any(x in key for x in brand_keys)), "")
        right_brand = next((value for key, value in right_dict.items() if any(x in key for x in brand_keys)), "")
        rows.append((*_set_metrics(left_keys, right_keys),
                     *_set_metrics(set(left_dict.values()), set(right_dict.values())),
                     *_set_metrics(left_articles, right_articles), exact, conflicts,
                     exact / len(common_keys) if common_keys else 1,
                     conflicts / len(common_keys) if common_keys else 0,
                     float(bool(left_brand and right_brand and left_brand == right_brand))))
    values = np.asarray(rows, dtype=np.float32)
    for group, prefix in enumerate(("attribute_keys", "attribute_values", "article_tokens")):
        for index, suffix in enumerate(("common", "jaccard", "dice", "containment")):
            output[f"{prefix}_{suffix}"] = values[:, group * 4 + index]
    names = ("exact_common_attribute_values", "conflicting_common_attribute_values",
             "exact_value_share_among_common_keys", "conflict_share_among_common_keys",
             "brand_exact_match")
    for index, name in enumerate(names, start=12):
        output[name] = values[:, index]


def build_features(pairs: pd.DataFrame) -> pd.DataFrame:
    """Строит 55 признаков CatBoost для произвольного числа пар.

    Args:
        pairs: Таблица с полями двух присоединённых карточек.

    Returns:
        Таблица из одного категориального и 54 числовых признаков.
    """

    text = {f"{field}_{side}": pairs[f"{field}_{side}"].fillna("").astype(str)
            for field in ("name", "attributes") for side in (1, 2)}
    for side in (1, 2):
        text[f"card_{side}"] = text[f"name_{side}"] + " " + text[f"attributes_{side}"]

    output = pd.DataFrame(index=pairs.index)
    category_1 = pairs["category_1"].fillna("<MISSING>").astype(str)
    category_2 = pairs["category_2"].fillna("<MISSING>").astype(str)
    output["category"] = [a if a == b else " <> ".join(sorted((a, b)))
                          for a, b in zip(category_1, category_2)]

    for field in ("name", "attributes", "card"):
        left, right = text[f"{field}_1"], text[f"{field}_2"]
        left_chars, right_chars = left.str.len().to_numpy(np.float32), right.str.len().to_numpy(np.float32)
        left_words = left.str.split().str.len().to_numpy(np.float32)
        right_words = right.str.split().str.len().to_numpy(np.float32)
        _pair_values(output, f"{field}_chars", left_chars, right_chars)
        _pair_values(output, f"{field}_words", left_words, right_words)
        if field == "name":
            output["name_chars_minmax_ratio"] = _minmax_ratio(left_chars, right_chars)
            output["name_words_minmax_ratio"] = _minmax_ratio(left_words, right_words)
        elif field == "card":
            output["card_words_minmax_ratio"] = _minmax_ratio(left_words, right_words)

    card_1, card_2 = text["card_1"], text["card_2"]
    digits_1 = card_1.str.count(r"\d").to_numpy(np.float32)
    digits_2 = card_2.str.count(r"\d").to_numpy(np.float32)
    _pair_values(output, "digits_total", digits_1, digits_2)
    output["digits_total_minmax_ratio"] = _minmax_ratio(digits_1, digits_2)
    for digit in "0123456789":
        _pair_values(output, f"digit_{digit}", card_1.str.count(digit).to_numpy(np.float32),
                     card_2.str.count(digit).to_numpy(np.float32))

    groups = (
        (("name_words",), text["name_1"], text["name_2"], ("all",)),
        (("attribute_words",), text["attributes_1"], text["attributes_2"], ("all",)),
        (("card_words", "numeric_tokens", "model_code_tokens"), card_1, card_2,
         ("all", "numeric", "model_code")),
    )
    suffixes = ("common", "jaccard", "dice", "containment")
    for prefixes, left, right, kinds in groups:
        values = _overlaps(left, right, kinds)
        for group_index, prefix in enumerate(prefixes):
            for metric_index, suffix in enumerate(suffixes):
                output[f"{prefix}_{suffix}"] = values[:, group_index * 4 + metric_index]
    name_1, name_2 = text["name_1"].str.casefold(), text["name_2"].str.casefold()
    output["exact_name"] = name_1.eq(name_2).to_numpy(np.int8)
    output["name_contains_other"] = [float(bool(a and b) and (a in b or b in a))
                                      for a, b in zip(name_1, name_2)]
    output["model_code_overlap_present"] = output["model_code_tokens_common"].gt(0).to_numpy(np.int8)
    output["numeric_token_conflict"] = output["numeric_tokens_containment"].eq(0).to_numpy(np.int8)
    output["model_code_token_conflict"] = output["model_code_tokens_containment"].eq(0).to_numpy(np.int8)
    char_metrics = _name_char_metrics(text["name_1"], text["name_2"])
    for index, suffix in enumerate(("common", "jaccard", "dice", "containment")):
        output[f"name_char_trigrams_{suffix}"] = char_metrics[:, index]
    _structured_features(output, text["attributes_1"], text["attributes_2"])
    return output


def adjust_probability(probability: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """Корректирует score по явным конфликтам карточек."""

    p_false = np.clip(1 - probability, 1e-7, 1 - 1e-7)
    score = np.log(p_false / (1 - p_false))
    evidence = np.column_stack([
        features["numeric_token_conflict"], features["model_code_token_conflict"],
        np.minimum(features["conflicting_common_attribute_values"], 5) / 5,
        1 - features["name_char_trigrams_jaccard"], 1 - features["name_words_containment"],
        features["article_tokens_containment"].eq(0).astype(float),
    ])
    score += evidence @ SCORE_COEFFICIENTS
    return 1 / (1 + np.exp(np.clip(score, -30, 30)))


def load_pairs(items_path: str, matches_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Читает входные parquet и присоединяет обе карточки к каждой паре."""

    matches = pd.read_parquet(matches_path, columns=["id1", "id2"])
    items = pd.read_parquet(items_path, columns=["id", "name", "attributes", "category"])
    if items["id"].duplicated().any():
        raise ValueError("items.id must be unique")
    cards = items.set_index("id")
    pairs = matches.copy()
    for side in (1, 2):
        joined = cards.reindex(matches[f"id{side}"].to_numpy())
        if joined.index.hasnans or joined[["name", "attributes", "category"]].isna().all(axis=1).any():
            raise ValueError(f"Some id{side} values are absent from items")
        pairs[[f"name_{side}", f"attributes_{side}", f"category_{side}"]] = joined.to_numpy()
    return matches, pairs


def main() -> None:
    """Вычисляет вероятность совпадения и записывает submission CSV."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--items_path", "--items-path", required=True)
    parser.add_argument("--matches_path", "--matches-path", required=True)
    parser.add_argument("--output_path", "--output-path", required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    matches, pairs = load_pairs(args.items_path, args.matches_path)
    features = build_features(pairs)
    model = CatBoostClassifier()
    model.load_model(str(MODEL_PATH))
    predict = adjust_probability(model.predict_proba(features, thread_count=-1)[:, 1], features)
    if len(predict) != len(matches) or not np.isfinite(predict).all():
        raise RuntimeError("Invalid prediction result")

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    matches.assign(predict=predict).to_csv(output_path, index=False)
    print(f"Saved {len(matches):,} predictions to {output_path}; elapsed={time.perf_counter() - started:.2f}s")


if __name__ == "__main__":
    main()
