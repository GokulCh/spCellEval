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
| Stellar | `src/methods/Stellar/requirements.txt` / `environment.yml` |
| CellLENS | `celllens`, `torch_geometric` |
| Eva | `Eva` package, `huggingface_hub` |
| KRONOS | `kronos` |
| VirTues | `virtues`, `lightning_fabric` |
| MAPS | `maps` |
| TRIBUS | `tribus` |
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
