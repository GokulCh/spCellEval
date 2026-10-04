"""Summary tables (CSV / JSON / Markdown). Heavy lifting lives in ``analysis.analyze_results``."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import analyze_results


def to_md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_float_dtype(d[c]):
            d[c] = d[c].map(lambda v: "" if pd.isna(v) else floatfmt.format(v))
    head = "| " + " | ".join(map(str, d.columns)) + " |\n|" + "---|" * len(d.columns) + "\n"
    return head + "".join("| " + " | ".join(r) + " |\n" for r in d.astype(str).to_numpy())


def write_reports(out: str | Path) -> dict[str, pd.DataFrame]:
    """Run the full analysis on a results directory and write summary/{summary.csv,json,md}."""
    out = Path(out)
    tabs = analyze_results(out)
    s = out / "summary"
    s.mkdir(exist_ok=True)
    fr = tabs.get("final_results")
    md = ["# Benchmark summary\n", "Primary split per dataset: " +
          ", ".join(f"{r.dataset} = {r.primary_split}" for r in tabs["primary_split"].itertuples()) +
          " (cv preferred, then holdout). Metrics are mean over folds.\n"]
    if fr is not None and len(fr):
        fr.to_csv(s / "summary.csv", index=False)
        fr.to_json(s / "summary.json", orient="records", indent=2)
        for lv, g in fr.groupby("eval_level"):
            piv = g.pivot_table(index=["category", "method"], columns="dataset", values="overall_performance").reset_index()
            piv.to_csv(s / f"comparison_overall_{lv}.csv", index=False)
            md += [f"\n## Overall performance, {lv}: methods x datasets\n", to_md(piv)]
        cols = ["dataset", "method", "category", "overall_performance", "f1_macro_mean", "f1_weighted_mean", "mcc_mean",
                "sensitivity_macro_mean", "specificity_macro_mean", "rare_accuracy_mean", "run_time", "stability", "scalability"]
        l3 = fr[fr.eval_level == ("level3" if "level3" in set(fr.eval_level) else fr.eval_level.iloc[0])]
        md += ["\n## Detailed results\n", to_md(l3[[c for c in cols if c in l3]].sort_values(["dataset", "overall_performance"], ascending=[True, False]))]
        piv2 = l3.pivot_table(index=["category", "method"], columns="dataset", values="f1_macro_mean").reset_index()
        piv2.to_csv(s / "comparison_f1_macro.csv", index=False)
    if len(tabs.get("unsupervised_qc", [])):
        md += ["\n## Unsupervised QC\n", to_md(tabs["unsupervised_qc"])]
    if len(tabs["failures"]):
        md += ["\n## Runs that did not complete\n", to_md(tabs["failures"])]
    md.append("\nSee `analysis/insights.md` for automatic findings.\n")
    (s / "summary.md").write_text("\n".join(md), encoding="utf-8")
    return tabs
