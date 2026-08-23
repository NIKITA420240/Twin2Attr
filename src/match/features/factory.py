"""Factory for configured item feature providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import (
    AppConfig,
    FeatureSettings,
    NerSettings,
    NormalizationSettings,
    PhysicalFeatureSettings,
)
from .contracts import ItemEnricher
from .manifest import feature_settings_from_manifest
from .pipeline import FeaturePipeline


@dataclass(frozen=True, slots=True)
class ItemEnricherFactory:
    """Create one concrete enricher without exposing its implementation."""

    device: str | None = None

    def create(
        self,
        provider: str,
        settings: NormalizationSettings | NerSettings | PhysicalFeatureSettings,
    ) -> ItemEnricher:
        if provider == "normalization":
            if not isinstance(settings, NormalizationSettings):
                raise TypeError("normalization factory requires NormalizationSettings")
            return self._create_normalization(settings)
        if provider == "ner":
            if not isinstance(settings, NerSettings):
                raise TypeError("NER factory requires NerSettings")
            return self._create_ner(settings)
        if provider == "physical":
            if not isinstance(settings, PhysicalFeatureSettings):
                raise TypeError(
                    "physical factory requires PhysicalFeatureSettings"
                )
            return self._create_physical(settings)
        raise ValueError(f"Unsupported feature provider: {provider!r}")

    @staticmethod
    def _create_normalization(
        settings: NormalizationSettings,
    ) -> ItemEnricher:
        from .normalization import NormalizationItemEnricher

        return NormalizationItemEnricher(
            synonyms_path=settings.synonyms_path,
            unique_attributes_path=settings.unique_attributes_path,
            source_column=settings.source_column,
            output_column=settings.output_column,
            n_jobs=settings.n_jobs,
            chunk_size=settings.chunk_size,
        )

    def _create_ner(self, settings: NerSettings) -> ItemEnricher:
        from .ner.enrichment import NerItemEnricher
        from .ner.serialization import load_word_ner_predictor

        if settings.model_dir is None:
            raise ValueError("enabled NER requires model_dir")
        extractor = load_word_ner_predictor(
            settings.model_dir,
            cluster_centers_path=settings.cluster_centers_path,
            batch_size=settings.batch_size,
            max_length=settings.max_length,
            use_amp=settings.use_amp,
            semantic_cleanup=settings.semantic_cleanup,
            device=self.device,
        )
        return NerItemEnricher(
            extractor=extractor,
            source_column=settings.source_column,
            output_column=settings.output_column,
            enriched_column=settings.enriched_column,
            merge_policy=settings.merge_policy,
        )

    def _create_physical(
        self,
        settings: PhysicalFeatureSettings,
    ) -> ItemEnricher:
        from .physical.enrichment import PhysicalItemEnricher
        from .physical.parser import PhysicalAttributeParser

        return PhysicalItemEnricher(
            parser=PhysicalAttributeParser(
                n_jobs=settings.n_jobs,
                chunk_size=settings.chunk_size,
            ),
            source_column=settings.source_column,
            output_column=settings.output_column,
            enriched_column=settings.enriched_column,
            merge_policy=settings.merge_policy,
            normalize_units=settings.normalize_units,
        )


def _pipeline(
    settings: FeatureSettings,
    *,
    device: str | None,
) -> FeaturePipeline:
    factory = ItemEnricherFactory(device=device)
    # Feature order is application behavior: normalize the source attributes,
    # then add semantic and deterministic values extracted from the item name.
    configured = (
        ("normalization", settings.normalization),
        ("ner", settings.ner),
        ("physical", settings.physical),
    )
    return FeaturePipeline(
        tuple(
            factory.create(name, provider_settings)
            for name, provider_settings in configured
            if provider_settings.enabled
        )
    )


def build_feature_pipeline(config: AppConfig) -> FeaturePipeline:
    return _pipeline(
        config.features,
        device=config.runtime.device,
    )


def build_manifest_feature_pipeline(
    solution: Mapping[str, Any],
    solution_root: Path,
) -> FeaturePipeline:
    values = solution.get("features")
    feature_values = values if isinstance(values, Mapping) else {}
    legacy_normalization = solution.get("normalization")
    features = feature_settings_from_manifest(
        feature_values,
        solution_root,
        legacy_normalization=(
            legacy_normalization
            if isinstance(legacy_normalization, Mapping)
            else None
        ),
    )
    return _pipeline(
        features,
        device=None if solution.get("device") is None else str(solution["device"]),
    )


__all__ = [
    "ItemEnricherFactory",
    "build_feature_pipeline",
    "build_manifest_feature_pipeline",
]
