# Installation

## Base environment (evaluation, QC, preprocessing, classic ML)

```
conda env create -f environment.yml
conda activate spcelleval
```

or `pip install -r requirements.txt` (Python 3.10/3.11).

Submodules (CellSighter, multiplex-image-annotator):

```
git submodule update --init --recursive
```

Napari-based QC notebooks additionally need `napari`, `magicgui`, `qtpy`.

## Per-method packages (use a separate environment per method)

| Method | Extra requirements |
|---|---|
| Stellar | table version needs only torch, torch_geometric, scipy, scikit-learn, pandas (own env if `cta` lacks them); see `src/methods/Stellar/requirements.txt` |
| CellLENS | `celllens`, `torch_geometric` |
| Eva | `Eva` package, `huggingface_hub` |
| KRONOS | `kronos` |
| VirTues | `virtues`, `lightning_fabric` |
| MAPS | clone github.com/mahmoodlab/MAPS, `pip install -e .` (own env: Python 3.9) |
| TRIBUS | clone github.com/farkkilab/tribus, `pip install -e .` (own env) |
| STARLING | `biostarling` (NOT `starling`, an unrelated package; pins numpy<2) |
| scyan | `scyan` |
| astir | `astir` |
| Nimbus | `nimbus_inference`, `alpineer` |
| DeepCellTypes | `deepcell_types` |
| CellSighter | `src/methods/CellSighter_benchmark` submodule |

Follow each method's own upstream installation instructions.

## R

CRAN: `tidyverse`, `data.table`, `argparse`, `logging`, `future`, `future.apply`,
`parallelly`, `Seurat`, `funkyheatmap`, `webr`, `colorspace`, `RColorBrewer`,
`Cairo`, `segmented`, `class`

Bioconductor: `SingleCellExperiment`, `FlowSOM`, `FuseSOM`

GitHub: `Rphenograph`, `TACIT`


## What each method needs

| Method | Python packages | Other |
|---|---|---|
| logistic_regression, random_forest, svm, most_frequent, stratified, singler, spade, spatial_gnn, knn_smooth, marker_score | scikit-learn | |
| xgboost | xgboost, scikit-learn | |
| louvain | networkx | |
| leiden | scanpy, leidenalg, anndata | |
| flowsom | | Rscript, R package `FlowSOM` |
| tacit | | Rscript, R package `TACIT` (the one installed from GitHub) |
| scyan | scyan, anndata | |
| astir | astir, torch | |
| tribus | tribus | |
| starling | starling, anndata, torch | |
| maps | maps | |
| stellar | torch, torch_geometric | |
| scanvi, scarches | scvi-tools | |
| cellsighter, nimbus, deepcelltypes | | image-based: need images + masks, not run from a quantification table |

A method whose package lives in its own conda environment is pointed at it in `configs/script_methods.json`
(`"scyan": {"python": "/path/to/env/bin/python"}`, `"tacit": {"Rscript": "..."}`). `python cli.py methods` shows what is
available in the current environment.
