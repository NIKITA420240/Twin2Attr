"""Word-level NER model extracted from the parsing notebook."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoConfig, AutoModel
from transformers.modeling_outputs import TokenClassifierOutput

from .config import WordNerModelConfig


class WordNERModel(nn.Module):
    def __init__(
        self,
        config: WordNerModelConfig,
        *,
        architecture_source: str | Path | None = None,
    ) -> None:
        super().__init__()
        source = architecture_source or config.model_name
        if architecture_source is None:
            self.bert = AutoModel.from_pretrained(source)
        else:
            base_config = AutoConfig.from_pretrained(source, local_files_only=True)
            self.bert = AutoModel.from_config(base_config)
        if not hasattr(self.bert, "encoder") or not hasattr(
            self.bert.encoder,
            "layer",
        ):
            raise TypeError("WordNERModel requires a BERT-like encoder")
        self.bert.encoder.layer = nn.ModuleList(self.bert.encoder.layer[:2])
        self.bert.config.num_hidden_layers = 2
        hidden_size = int(self.bert.config.hidden_size)
        self.hidden_size = hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.max_subwords_per_word = config.max_subwords_per_word
        self.subword_pos_embedding = nn.Embedding(
            config.max_subwords_per_word,
            config.pos_dim,
        )
        self.word_attention = nn.Sequential(
            nn.Linear(hidden_size + config.pos_dim, config.attention_hidden),
            nn.Tanh(),
            nn.Linear(config.attention_hidden, config.num_attention_heads),
        )
        self.word_projection = nn.Sequential(
            nn.Linear(hidden_size * config.num_attention_heads, hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.sequence_encoder = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=config.sequence_heads,
            dim_feedforward=hidden_size * 2,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.classifier = nn.Linear(hidden_size, len(config.class_names))

    def pool_words(
        self,
        hidden: torch.Tensor,
        word_ids: torch.Tensor,
        subword_positions: torch.Tensor,
        *,
        max_words: int,
    ) -> torch.Tensor:
        batch_size, token_count, hidden_size = hidden.shape
        positions = subword_positions.clamp(
            max=self.max_subwords_per_word - 1
        )
        scores = self.word_attention(
            torch.cat((hidden, self.subword_pos_embedding(positions)), dim=-1)
        )
        head_count = scores.size(-1)
        valid = word_ids >= 0
        valid_flat = valid.reshape(-1)
        valid_word_ids = word_ids.reshape(-1)[valid_flat]
        valid_hidden = hidden.reshape(-1, hidden_size)[valid_flat]
        valid_scores = scores.reshape(-1, head_count)[valid_flat]
        batch_ids = (
            torch.arange(batch_size, device=hidden.device)[:, None]
            .expand(-1, token_count)
            .reshape(-1)[valid_flat]
        )
        group_ids = batch_ids * max_words + valid_word_ids
        group_count = batch_size * max_words
        pooled_heads = []
        for head in range(head_count):
            head_scores = valid_scores[:, head]
            group_max = torch.full(
                (group_count,),
                -torch.inf,
                dtype=head_scores.dtype,
                device=hidden.device,
            )
            group_max.scatter_reduce_(
                0,
                group_ids,
                head_scores,
                reduce="amax",
                include_self=True,
            )
            weights = torch.exp(head_scores - group_max[group_ids])
            group_sum = torch.zeros(
                group_count,
                dtype=weights.dtype,
                device=hidden.device,
            )
            group_sum.scatter_add_(0, group_ids, weights)
            weights /= group_sum[group_ids].clamp_min(1e-8)
            pooled = torch.zeros(
                (group_count, hidden_size),
                dtype=hidden.dtype,
                device=hidden.device,
            )
            pooled.index_add_(0, group_ids, valid_hidden * weights[:, None])
            pooled_heads.append(
                pooled.view(batch_size, max_words, hidden_size)
            )
        return self.word_projection(torch.stack(pooled_heads, dim=2).flatten(2))

    def encode_words(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids: torch.Tensor,
        subword_positions: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        word_lengths = word_ids.amax(dim=1).add(1).clamp_min(0)
        max_words = max(1, int(word_lengths.max().item()))
        semantic_hidden = self.pool_words(
            outputs.last_hidden_state,
            word_ids,
            subword_positions,
            max_words=max_words,
        )
        padding_mask = (
            torch.arange(max_words, device=input_ids.device)[None, :]
            >= word_lengths[:, None]
        )
        semantic_hidden = semantic_hidden.masked_fill(
            padding_mask[..., None],
            0.0,
        )
        contextual_hidden = self.sequence_encoder(
            semantic_hidden,
            src_key_padding_mask=padding_mask,
        )
        return contextual_hidden, semantic_hidden, word_lengths

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        word_ids: torch.Tensor,
        subword_positions: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **_: object,
    ) -> TokenClassifierOutput:
        contextual_hidden, _, _ = self.encode_words(
            input_ids,
            attention_mask,
            word_ids,
            subword_positions,
            token_type_ids,
        )
        logits = self.classifier(contextual_hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return TokenClassifierOutput(loss=loss, logits=logits)


__all__ = ["WordNERModel"]
