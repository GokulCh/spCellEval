"""Tests for notebook-style reporting tables and plots."""

import sys
from pathlib import Path

import pandas as pd
import pytest

_REPO = Path(__file__).resolve().parents[1]
_EVAL = _REPO / "src" / "evaluation"
sys.path.insert(0, str(_EVAL))

from reporting import (  # noqa: E402
    build_confusion_dataframe,
    composition_table,
    export_ground_truth_composition,
    notebook_overall_score,
    stability_score,
)


def test_stability_score():
    assert stability_score(0.0) == pytest.approx(1.0)
    assert stability_score(0.05) == pytest.approx(0.5)
    assert stability_score(0.2) == pytest.approx(0.0)


def test_notebook_overall_score():
    row = pd.Series({
        "weighted_f1_mean": 0.8,
        "accuracy_mean": 0.9,
        "macro_f1_mean": 0.7,
        "mcc_mean": 0.6,
        "kappa_mean": 0.5,
        "r2_composition_mean": 0.4,
        "pearson_composition_mean": 0.3,
        "ari_mean": 0.2,
        "nmi_mean": 0.1,
        "stability": 1.0,
    })
    assert notebook_overall_score(row) == pytest.approx(0.55)


def test_composition_table():
    tbl = composition_table(pd.Series(["A", "A", "B"]), "test")
    assert len(tbl) == 2
    assert tbl.loc[tbl["phenotype"] == "A", "count"].iloc[0] == 2


def test_build_confusion_dataframe():
    yt = pd.Series(["A", "A", "B", "B"])
    yp = pd.Series(["A", "B", "B", "B"])
    cm, labels = build_confusion_dataframe(yt, yp)
    assert "A" in labels
    assert cm.loc["A", "A"] == 1


def test_export_ground_truth_composition(tmp_path):
    quant = tmp_path / "quant.csv"
    pd.DataFrame({
        "cell_type": ["A", "A", "B"],
        "Image_ID": ["i1", "i1", "i2"],
    }).to_csv(quant, index=False)
    out = export_ground_truth_composition(quant, tmp_path / "tables")
    assert out is not None
    df = pd.read_csv(out)
    assert "ground_truth" in df["label"].values


try:
    import matplotlib
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


@pytest.mark.skipif(not HAS_MPL, reason="matplotlib not installed")
def test_plot_helpers_write_files(tmp_path):
    matplotlib.use("Agg")
    from visualize import (  # noqa: WPS433
        plot_composition_grouped,
        plot_confusion_matrix_heatmap,
        plot_metric_fold_ranges,
        plot_spatial_phenotypes,
    )

    counts_a = pd.Series({"A": 2, "B": 1})
    counts_b = pd.Series({"A": 1, "B": 2})
    grouped = plot_composition_grouped(
        counts_a, counts_b, tmp_path / "grouped.png", title="test",
    )
    assert grouped is not None
    assert grouped.is_file()

    cm = plot_confusion_matrix_heatmap(
        pd.Series(["A", "B"]), pd.Series(["A", "B"]),
        tmp_path / "cm.png", title="cm",
    )
    assert cm is not None

    per_fold = tmp_path / "per_fold.csv"
    pd.DataFrame({
        "method": ["m1", "m1"],
        "level": ["level3", "level3"],
        "accuracy": [0.8, 0.9],
        "macro_f1": [0.7, 0.75],
    }).to_csv(per_fold, index=False)
    ranges = plot_metric_fold_ranges(per_fold, tmp_path / "ranges.png", level="level3")
    assert ranges is not None

    spatial = plot_spatial_phenotypes(
        pd.DataFrame({
            "x": [0.0, 1.0], "y": [0.0, 1.0],
            "predicted_phenotype": ["A", "B"],
        }),
        tmp_path / "spatial.png",
        title="spatial",
    )
    assert spatial is not None
