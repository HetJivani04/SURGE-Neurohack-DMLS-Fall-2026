"""I/O utilities for dataset discovery, preprocessing, and schema contracts."""

from .contract import (
    ContractError,
    NPZ_SCHEMA,
    SCHEMA_VERSION,
    load_connectomes,
    load_embeddings,
    read_manifest,
    read_subject_run,
    subject_run_path,
    two_run_subjects,
    validate_subject_run,
    write_manifest,
    write_subject_run,
)

__all__ = [
    "ContractError",
    "NPZ_SCHEMA",
    "SCHEMA_VERSION",
    "subject_run_path",
    "write_subject_run",
    "read_subject_run",
    "validate_subject_run",
    "read_manifest",
    "write_manifest",
    "two_run_subjects",
    "load_connectomes",
    "load_embeddings",
]
