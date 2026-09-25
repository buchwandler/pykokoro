# PolyNorm benchmark status

This maintainer benchmark guide describes the previous pipeline-based PolyNorm harness,
including separate plain-document and SSMD modes. That harness has not been migrated to
the request-centric PyKokoro API and is not a supported verification path for the new
engine boundary.

The upstream PolyNorm dataset is not bundled with PyKokoro. Its terms and provenance
remain unchanged; do not treat the old document-pipeline reports as evidence for the
request API. For current engine behavior, see the [API reference](api_reference.md) and
[breaking-change note](breaking-change-0.10.0.md).
