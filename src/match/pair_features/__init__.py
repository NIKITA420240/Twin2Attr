"""Model-independent features computed from complete product pairs."""

from .typed_attributes import (
    ATTRIBUTE_TYPES,
    TypedAttributeComparison,
    TypedAttributeComparator,
    TypedAttributeOptions,
    aggregate_typed_attribute_features,
    typed_attribute_feature_names,
)

__all__ = [
    "ATTRIBUTE_TYPES",
    "TypedAttributeComparison",
    "TypedAttributeComparator",
    "TypedAttributeOptions",
    "aggregate_typed_attribute_features",
    "typed_attribute_feature_names",
]
