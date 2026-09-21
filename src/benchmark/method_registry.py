"""
method_registry.py
==================
Catalog of every annotation method in ``src/methods/``.

Each ``MethodSpec`` records the method category (which determines how the
benchmark runner invokes it), script path, and per-dataset artifact paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[2]
_METHODS = _REPO / "src" / "methods"


class MethodCategory(str, Enum):
    """How the benchmark runner executes a method."""

    SUPERVISED_KFOLD = "supervised_kfold"
    """Needs k-fold CSVs from ``run_kfold_creator`` (classic ML, MAPS)."""

    UNSUPERVISED_QUANT = "unsupervised_quant"
    """Clusters on the full quantification CSV; maps clusters → GT via greedy F1."""

    PRIOR_KNOWLEDGE = "prior_knowledge"
    """Uses a marker decision matrix / biological rules on the full quant CSV."""

    IMAGE_PIPELINE = "image_pipeline"
    """Needs raw images + segmentation (foundation models, CellSighter, etc.)."""


@dataclass(frozen=True)
class MethodSpec:
    id: str
    display_name: str
    category: MethodCategory
    description: str
    script: Optional[Path] = None
    interpreter: str = "python"
    requires_r: bool = False
    requires_gpu: bool = False
    requires_raw_images: bool = False
    # Per-dataset artifact keys → relative paths (resolved by DatasetContext.artifact)
    artifact_keys: List[str] = field(default_factory=list)
    # Default enabled for tabular benchmarking (no raw images needed)
    tabular_default: bool = False


def _p(*parts: str) -> Path:
    return _METHODS.joinpath(*parts)


# Per-dataset artifact tables (relative to repo root)
_IMMUCAN = {
    "decision_matrix_scyan": "configs/artifacts/IMMUcan/scyan_decision_matrix_level3.csv",
    "decision_matrix_tacit": "configs/artifacts/IMMUcan/tacit_decision_matrix_level3.csv",
    "decision_matrix_astir": "src/methods/astir/cell_types_IMMUcan.yml",
    "celllens_script": "src/methods/CellLens/IMMUcan_LITE_CellLENS.py",
}
_CRC_TMA = {
    "decision_matrix_scyan": "configs/artifacts/CRC_TMA/scyan_decision_matrix_level3.csv",
    "decision_matrix_tacit": "configs/artifacts/CRC_TMA/tacit_decision_matrix_level3.csv",
    "decision_matrix_astir": "src/methods/astir/cell_types_CRC_TMA.yml",
}

_DATASET_ARTIFACTS: Dict[str, Dict[str, str]] = {
    "IMMUcan": _IMMUCAN,
    "CRC_TMA": _CRC_TMA,
}


METHOD_REGISTRY: Dict[str, MethodSpec] = {}

def _register(spec: MethodSpec) -> MethodSpec:
    METHOD_REGISTRY[spec.id] = spec
    return spec


# ── Supervised (k-fold) ───────────────────────────────────────────────────
_register(MethodSpec(
    id="random_forest",
    display_name="Random Forest",
    category=MethodCategory.SUPERVISED_KFOLD,
    description="sklearn RandomForest on k-fold splits.",
    script=_p("classic_ml", "run_classic_ml_default.py"),
    tabular_default=True,
))
_register(MethodSpec(
    id="logistic_regression",
    display_name="Logistic Regression",
    category=MethodCategory.SUPERVISED_KFOLD,
    description="sklearn LogisticRegression on k-fold splits.",
    script=_p("classic_ml", "run_classic_ml_default.py"),
    tabular_default=True,
))
_register(MethodSpec(
    id="xgboost",
    display_name="XGBoost",
    category=MethodCategory.SUPERVISED_KFOLD,
    description="XGBoost classifier on k-fold splits.",
    script=_p("classic_ml", "run_classic_ml_default.py"),
    tabular_default=True,
))
_register(MethodSpec(
    id="svm",
    display_name="SVM",
    category=MethodCategory.SUPERVISED_KFOLD,
    description="sklearn LinearSVM (linear, class-balanced) on k-fold splits.",
    script=_p("classic_ml", "run_classic_ml_default.py"),
    tabular_default=True,
))
_register(MethodSpec(
    id="ribca_adapted",
    display_name="RIBCA (adapted)",
    category=MethodCategory.SUPERVISED_KFOLD,
    description="Reference-informed cell annotation: mean profiles + correlation + Hungarian assignment.",
    script=None,
    tabular_default=True,
))
_register(MethodSpec(
    id="maps",
    display_name="MAPS",
    category=MethodCategory.SUPERVISED_KFOLD,
    description="Deep learning MAPS cell phenotyping on k-fold splits.",
    script=_p("MAPS", "run_maps.py"),
    requires_gpu=True,
    tabular_default=False,
))
_register(MethodSpec(
    id="singler",
    display_name="SingleR",
    category=MethodCategory.SUPERVISED_KFOLD,
    description="Reference mapping via Pearson correlation to mean cell-type profiles.",
    script=_p("singler", "run_singler.py"),
    tabular_default=True,
))
_register(MethodSpec(
    id="scarches",
    display_name="scArches",
    category=MethodCategory.SUPERVISED_KFOLD,
    description="Reference label transfer via scanpy ingest (scArches-compatible).",
    script=_p("scarches", "run_scarches.py"),
    tabular_default=True,
))

# ── Unsupervised (full quant CSV) ─────────────────────────────────────────
_register(MethodSpec(
    id="leiden",
    display_name="Leiden",
    category=MethodCategory.UNSUPERVISED_QUANT,
    description="Leiden clustering + greedy F1 label mapping.",
    script=_p("leiden", "run_leiden_clustering.py"),
    tabular_default=True,
))
_register(MethodSpec(
    id="louvain",
    display_name="Louvain",
    category=MethodCategory.UNSUPERVISED_QUANT,
    description="Louvain clustering + greedy F1 label mapping.",
    script=_p("louvain", "run_louvain_clustering.py"),
    tabular_default=True,
))
_register(MethodSpec(
    id="spade",
    display_name="SPADE",
    category=MethodCategory.UNSUPERVISED_QUANT,
    description="SPADE-inspired density downsampling + hierarchical clustering.",
    script=_p("spade", "run_spade.py"),
    tabular_default=True,
))
_register(MethodSpec(
    id="starling",
    display_name="Starling",
    category=MethodCategory.UNSUPERVISED_QUANT,
    description="Starling probabilistic gating + greedy F1 mapping.",
    script=_p("starling", "run_starling.py"),
    tabular_default=True,
))
_register(MethodSpec(
    id="flowsom",
    display_name="FlowSOM",
    category=MethodCategory.UNSUPERVISED_QUANT,
    description="FlowSOM metaclustering (R) + greedy F1 mapping.",
    script=_p("FlowSOM", "run_flowsom.R"),
    interpreter="rscript",
    requires_r=True,
    tabular_default=True,
))
_register(MethodSpec(
    id="phenograph",
    display_name="Phenograph",
    category=MethodCategory.UNSUPERVISED_QUANT,
    description="Phenograph clustering (R) + greedy F1 mapping.",
    script=_p("Phenograph", "run_phenograph.R"),
    interpreter="rscript",
    requires_r=True,
    tabular_default=False,
))
_register(MethodSpec(
    id="fusesom",
    display_name="FuseSOM",
    category=MethodCategory.UNSUPERVISED_QUANT,
    description="FuseSOM clustering (R) + greedy F1 mapping.",
    script=_p("FuseSOM", "run_fusesom.R"),
    interpreter="rscript",
    requires_r=True,
    tabular_default=False,
))

# ── Prior-knowledge (decision matrix / rules) ─────────────────────────────
_register(MethodSpec(
    id="signature",
    display_name="Signature Rules",
    category=MethodCategory.PRIOR_KNOWLEDGE,
    description="Pure-Python marker threshold rules (Stage 2).",
    script=_REPO / "src" / "pseudo_labeling" / "run_pseudo_labeler.py",
    artifact_keys=["decision_matrix_tacit"],
    tabular_default=True,
))
_register(MethodSpec(
    id="tacit",
    display_name="TACIT",
    category=MethodCategory.PRIOR_KNOWLEDGE,
    description="TACIT prior-knowledge cell typing (R).",
    script=_REPO / "src" / "pseudo_labeling" / "run_tacit.py",
    requires_r=True,
    artifact_keys=["decision_matrix_tacit"],
    tabular_default=True,
))
_register(MethodSpec(
    id="scyan",
    display_name="Scyan",
    category=MethodCategory.PRIOR_KNOWLEDGE,
    description="Scyan Bayesian cell typing with signed marker rules.",
    script=_p("scyan", "run_scyan.py"),
    artifact_keys=["decision_matrix_scyan"],
    tabular_default=True,
))
_register(MethodSpec(
    id="astir",
    display_name="Astir",
    category=MethodCategory.PRIOR_KNOWLEDGE,
    description="Astir deep generative model with YAML marker definitions.",
    script=_p("astir", "run_astir.py"),
    artifact_keys=["decision_matrix_astir"],
    tabular_default=False,
))
_register(MethodSpec(
    id="tribus",
    display_name="Tribus",
    category=MethodCategory.PRIOR_KNOWLEDGE,
    description="Tribus logic-rule cell typing (requires external Excel matrix).",
    script=_p("tribus", "run_tribus.py"),
    artifact_keys=["decision_matrix_tribus"],
    tabular_default=False,
))

# ── Image / foundation-model pipelines ────────────────────────────────────
_register(MethodSpec(
    id="stellar",
    display_name="Stellar",
    category=MethodCategory.IMAGE_PIPELINE,
    description="Spatial GNN (STELLAR) — needs images, segmentation, folds.json.",
    script=_p("Stellar", "run_stellar.py"),
    requires_gpu=True,
    requires_raw_images=True,
))
_register(MethodSpec(
    id="cellsighter",
    display_name="CellSighter",
    category=MethodCategory.IMAGE_PIPELINE,
    description="CNN on single-cell image crops.",
    script=_p("CellSighter", "run_cellsighter.py"),
    requires_gpu=True,
    requires_raw_images=True,
))
_register(MethodSpec(
    id="eva_supervised",
    display_name="EVA (supervised)",
    category=MethodCategory.IMAGE_PIPELINE,
    description="EVA foundation model — supervised head.",
    script=_p("Eva", "run_eva_forked.py"),
    requires_gpu=True,
    requires_raw_images=True,
))
_register(MethodSpec(
    id="eva_leiden",
    display_name="EVA (Leiden)",
    category=MethodCategory.IMAGE_PIPELINE,
    description="EVA embeddings + Leiden clustering.",
    script=_p("Eva", "run_eva_forked.py"),
    requires_gpu=True,
    requires_raw_images=True,
))
_register(MethodSpec(
    id="kronos_supervised",
    display_name="KRONOS (supervised)",
    category=MethodCategory.IMAGE_PIPELINE,
    description="KRONOS foundation model — RF supervised.",
    script=_p("KRONOS", "run_kronos.py"),
    requires_gpu=True,
    requires_raw_images=True,
))
_register(MethodSpec(
    id="kronos_leiden",
    display_name="KRONOS (Leiden)",
    category=MethodCategory.IMAGE_PIPELINE,
    description="KRONOS embeddings + Leiden clustering.",
    script=_p("KRONOS", "run_kronos.py"),
    requires_gpu=True,
    requires_raw_images=True,
))
_register(MethodSpec(
    id="virtues_supervised",
    display_name="VirTues (supervised)",
    category=MethodCategory.IMAGE_PIPELINE,
    description="VirTues foundation model — supervised.",
    script=_p("VirTues", "run_virtues.py"),
    requires_gpu=True,
    requires_raw_images=True,
))
_register(MethodSpec(
    id="virtues_leiden",
    display_name="VirTues (Leiden)",
    category=MethodCategory.IMAGE_PIPELINE,
    description="VirTues embeddings + Leiden.",
    script=_p("VirTues", "run_virtues.py"),
    requires_gpu=True,
    requires_raw_images=True,
))
_register(MethodSpec(
    id="deepcelltypes",
    display_name="DeepCellTypes",
    category=MethodCategory.IMAGE_PIPELINE,
    description="Image-based prior-knowledge cell typing.",
    script=_p("deepcelltypes", "run_deepcelltypes.py"),
    requires_gpu=True,
    requires_raw_images=True,
))
_register(MethodSpec(
    id="nimbus",
    display_name="Nimbus",
    category=MethodCategory.IMAGE_PIPELINE,
    description="Nimbus marker inference + clustering pipeline.",
    script=_p("Nimbus", "nimbus.py"),
    requires_raw_images=True,
))
_register(MethodSpec(
    id="celllens",
    display_name="CellLENS",
    category=MethodCategory.IMAGE_PIPELINE,
    description="CellLENS unsupervised clustering on crops.",
    script=None,  # resolved per dataset via artifact
    requires_raw_images=True,
))


def list_methods(
    category: Optional[MethodCategory] = None,
    tabular_only: bool = False,
) -> List[MethodSpec]:
    specs = list(METHOD_REGISTRY.values())
    if category:
        specs = [s for s in specs if s.category == category]
    if tabular_only:
        specs = [s for s in specs if s.tabular_default]
    return sorted(specs, key=lambda s: s.id)


def get_dataset_artifacts(dataset_name: str) -> Dict[str, str]:
    return _DATASET_ARTIFACTS.get(dataset_name, {})


def resolve_artifact(dataset_name: str, key: str) -> Optional[str]:
    return get_dataset_artifacts(dataset_name).get(key)


DEFAULT_UNLABELED_METHODS: List[str] = ["signature", "scyan", "leiden"]

_UNLABELED_CATEGORIES = frozenset({
    MethodCategory.UNSUPERVISED_QUANT,
    MethodCategory.PRIOR_KNOWLEDGE,
})


def is_unlabeled_compatible(method_id: str) -> bool:
    """Return True if *method_id* does not require expert labels or k-folds."""
    spec = METHOD_REGISTRY.get(method_id)
    return spec is not None and spec.category in _UNLABELED_CATEGORIES


def filter_unlabeled_methods(methods: List[str]) -> List[str]:
    """Keep only prior-knowledge and unsupervised tabular methods."""
    filtered = [m for m in methods if is_unlabeled_compatible(m)]
    return filtered if filtered else list(DEFAULT_UNLABELED_METHODS)


def resolve_unlabeled_methods(
    ds_entry: dict,
    bench_cfg: dict,
    explicit: Optional[List[str]] = None,
) -> List[str]:
    """Default method list for ``--unlabeled`` benchmark runs."""
    if explicit:
        return filter_unlabeled_methods(explicit)
    custom = ds_entry.get("unlabeled_methods") or bench_cfg.get("defaults", {}).get(
        "unlabeled_methods"
    )
    if custom:
        return filter_unlabeled_methods(list(custom))
    return list(DEFAULT_UNLABELED_METHODS)
