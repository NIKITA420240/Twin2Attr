"""Для получения maxpooling эмбедингов нужно вызвать функцию get_maxpooling_embeddings"""

from __future__ import annotations
from paths import resolve_project_path

from pathlib import Path
from typing import List

import numpy as np
import polars as pl

import re
import orjson
import yaml

from gensim.models import FastText

from loguru import logger
from tqdm import tqdm

_NON_ALNUM_RE = re.compile(r'[^а-яёa-z0-9]+')
_SPACE_RE = re.compile(r'\s+')

# в тексте оставляем только руссике/английские буквы и цифры
# удаляем лишние пробелы
def normalize_text(text):
    text = str(text).lower()
    text = _NON_ALNUM_RE.sub(' ', text)
    return _SPACE_RE.sub(' ', text).strip()

# просто из json в dict
def parse_attributes(attrs):
    if attrs is None:
        return {}
    attrs = orjson.loads(attrs)
    return attrs

def prepare_attributes(attrs):
    attrs = parse_attributes(attrs)
    return {normalize_text(key): normalize_text(val) for key, val in attrs.items()}

# словарь атрибутов товара -> в список слов для обучения
def attributes_to_tokens(attrs):
    tokens = []
    
    for key, val in attrs.items():
        tokens.extend(key.split())
        tokens.extend(val.split())
    
    return tokens

# создание корпуса слов для обучения FastText
def build_corpus(df):
    corpus = []

    for raw_attrs in tqdm(df['attributes'].drop_nulls(), desc='Building corpus'):
        attrs = prepare_attributes(raw_attrs)
        tokens = attributes_to_tokens(attrs)
            
        if tokens:
            corpus.append(tokens)
    return corpus


def train_fasttext(corpus: List[List], VECTOR_SIZE = 256) -> FastText:
    
    ft_model = FastText(
        vector_size=VECTOR_SIZE,
        window=5,
        min_count=2,
        workers=8,
        sg=1,
        min_n=2,
        max_n=5,
        epochs=10
    )

    ft_model.build_vocab(corpus_iterable=corpus)
    ft_model.train(corpus, total_examples=len(corpus), epochs=ft_model.epochs)

    return ft_model

# получение эмбединга текста = сумма эмбедингов слов / кол-во токенов
def text_embedding(text, model):

    # если текст пустой, то возвращаем вектор из 0
    if not text:
        return np.zeros(model.vector_size, dtype=np.float32)

    tokens = text.split()

    # итоговый вектор = сумма векторов всех токенов / количество токенов
    embedding = np.zeros(model.vector_size, dtype=np.float32)

    for token in tokens:
        idx = model.wv.key_to_index.get(token)
        if idx is None:
            embedding += model.wv.get_vector(token)
        else:
            embedding += model.wv.vectors[idx]
    return embedding * np.float32(1.0/len(tokens))

# получаем минимум и максимум по каждому эмбедингу ключа и значения 
# для подсчета maxpooling
def get_atribute_min_max(attrs, model):
    # возвращаем кортеж: название атрибута, 
    keys, vals, names = [], [], []

    for key, val in attrs.items():
        # считаем эмбединги
        key_emb = text_embedding(key, model)
        val_emb = text_embedding(val, model)

        # нормализуем эмбединги
        key_norm = np.linalg.norm(key_emb)
        val_norm = np.linalg.norm(val_emb)

        if key_norm > 0:
            key_emb = key_emb / key_norm
        if val_norm > 0:
            val_emb = val_emb / val_norm

        keys.append(key_emb)
        vals.append(val_emb)
        
        names.append(key)
    
    if not keys:
        empty = np.zeros(model.vector_size)
        return (names, empty, empty, empty, empty)
        
    keys = np.asarray(keys)
    vals = np.asarray(vals)

    max_keys = keys.max(axis=0)
    max_vals = vals.max(axis=0)
    min_keys = keys.min(axis=0)
    min_vals = vals.min(axis=0)
    
    return (names, min_keys, max_keys, min_vals, max_vals)

def max_pairwise_product(out, a_min, a_max, b_min, b_max, tmp):
    np.multiply(a_min, b_min, out=out)
    np.multiply(a_min, b_max, out=tmp)
    np.maximum(out, tmp, out=out)
    np.multiply(a_max, b_min, out=tmp)
    np.maximum(out, tmp, out=out)
    np.multiply(a_max, b_max, out=tmp)
    np.maximum(out, tmp, out=out)


# подготовливаем таблицу matches
# для каждой пары id в строке находим атрибуты
def prepare_matches(matches: pl.DataFrame, items_path: str):

    # извлекаем только нужные id
    needed_ids = pl.concat([matches.select(pl.col('id1').alias('id')),
                                matches.select(pl.col('id2').alias('id'))]).unique()

    if Path(items_path).exists():
        # сканируем items.parquet
        items_lazy = pl.scan_parquet(items_path).select(['id', 'attributes'])
    else:
        logger.error("Items path doesn't exist: ", {items_path})
        raise FileNotFoundError(f"Items file not found: {items_path}")

    # находим нужные товары 
    needed_items = items_lazy.join(needed_ids.lazy(), on='id', how='semi').collect()
    items_for_join = needed_items.select(['id', 'attributes'])

    # join товаров из items с matches по id
    matches = matches.join(items_for_join.rename(
        {'id': 'id1', 'attributes': 'attributes1'}), on='id1', how='left')
    
    matches = matches.join(items_for_join.rename(
        {'id': 'id2', 'attributes': 'attributes2'}), on='id2', how='left')

    return matches

# получение модели fasttext
def get_fasttext(use_pretrained_fasttext):
    config_path = resolve_project_path('configs/config.yaml')
    with open(config_path) as file:
        params = yaml.safe_load(file)
        fasttext_path = resolve_project_path(params['fasttext_path'])
        items_path = resolve_project_path(params['items_path'])

        # если модели нет по указанному пути или нужно ее обучить
        if not Path(fasttext_path).exists() or not use_pretrained_fasttext:
            logger.info("Start training FastText model")
            df = pl.read_parquet(items_path)
            corpus = build_corpus(df)
            ft_model = train_fasttext(corpus)
            logger.info("Finished training FastText model")
        # иначе подгружаем готовую модель
        else:
            logger.info("Load pretrained FastText model")
            ft_model = FastText.load(str(fasttext_path))
        return ft_model

def calculate_embedding(attrs1, attrs2, ft_model, feature_dim):
    # получаем min, max по каждому эмбедингу атрибутов
    _, min_keys_1, max_keys_1, min_vals_1, max_vals_1 = get_atribute_min_max(attrs1, ft_model)
    _, min_keys_2, max_keys_2, min_vals_2, max_vals_2 = get_atribute_min_max(attrs2, ft_model)
    
    d = len(min_keys_1)

    # считаем maxpooling
    out = np.zeros(feature_dim, dtype=np.float32)
    tmp = np.zeros(feature_dim, dtype=np.float32)

    max_pairwise_product(out[:d], min_keys_1, max_keys_1, min_keys_2, max_keys_2, tmp[:d])
    max_pairwise_product(out[d:], min_vals_1, max_vals_1, min_vals_2, max_vals_2, tmp[d:])

    return np.asarray(out, dtype=np.float32)


# эта функция получает на вход относительный путь к файлу matches.parquet и items.parquet
# также можно указать, нужно ли заново обучать fasttext (+указать новый размер эмбединга) 
# или же взять готовые веса
# функция вернет матрицу, каждая строка которой - это эмбединг пары карточек из matches
# размерность эмбединга - 2 * размерность эмбединга fasttext (по дефолту 256)
def get_maxpooling_embeddings(matches_path: str, items_path: str, use_pretrained_fasttext=True, vector_size=256):
    logger.info('Start obtaining embeddings via maxpooling')
    # список готовых эмбедингов для каждой пары карточек товаров
    embeddings = []
    
    matches_path = resolve_project_path(matches_path)
    items_path = resolve_project_path(items_path)

    if Path(matches_path).exists():
        matches = pl.read_parquet(matches_path)
    else:
        logger.error("Matches path doesn't exist: {}", matches_path)
        raise FileNotFoundError(f"Matches file not found: {items_path}")

    # для каждой пары карточек ищем их атрибуты
    logger.info("Start preparing matches dataset")
    matches = prepare_matches(matches, items_path)
    logger.info("Finished preparing matches dataset")

    # подгружаем (или обучаем) FastText
    ft_model = get_fasttext(use_pretrained_fasttext)

    # атрибуты карточек
    raw_attrs1 = matches['attributes1']
    raw_attrs2 = matches['attributes2']


    # итоговая размерность эмбединга
    feature_dim = 2 * ft_model.vector_size

    logger.info("Start calculate maxpooling embeddings")
    try:
        for raw_attr1, raw_attr2 in tqdm(zip(raw_attrs1, raw_attrs2), desc='Calculating embeddings'):
            # обработка атрибутов
            attr1 = prepare_attributes(raw_attr1)
            attr2 = prepare_attributes(raw_attr2)
        
            embedding = calculate_embedding(attr1, attr2, ft_model, feature_dim)
            embeddings.append(embedding)
    except Exception:
        logger.error("Something went wrong!")
    

    logger.info("Maxpooling embedding calculations completed successfully!")
    return np.asarray(embeddings)



