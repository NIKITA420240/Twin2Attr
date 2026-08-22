"""Factory for configured item feature providers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..config import (
    AppConfig,
    FeatureSettings,
    NerSettings,
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
        settings: NerSettings | PhysicalFeatureSettings,
    ) -> ItemEnricher:
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
    # Feature order is part of application behavior: semantic extraction first,
    # deterministic physical values second.
    configured = (
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
    if not isinstance(values, Mapping):
        return FeaturePipeline()
    features = feature_settings_from_manifest(values, solution_root)
    return _pipeline(
        features,
        device=None if solution.get("device") is None else str(solution["device"]),
    )


__all__ = [
    "ItemEnricherFactory",
    "build_feature_pipeline",
    "build_manifest_feature_pipeline",
]
