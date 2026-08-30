import argparse
import logging
import os
import sys
import time
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata._core.aligned_df import ImplicitModificationWarning

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_script_dir))
sys.path.insert(0, os.path.join(project_root, "methods", "utils"))

from clustering_benchmark import timed_cluster_run, write_greedy_predictions  # noqa: E402

parser = argparse.ArgumentParser(
    description="Louvain clustering with greedy assignment and optional preprocessing."
)
parser.add_argument("-i", "--input", dest="input", type=str, required=True)
parser.add_argument("-o", "--output", dest="output_path", type=str, required=True)
parser.add_argument("-m", "--markers", dest="markers", nargs="+", required=True)
parser.add_argument("-p", "--PCA", dest="PCA", action="store_true")
parser.add_argument("-a", "--arcsine", dest="arcsine", action="store_true")
parser.add_argument("-n", "--normalization", dest="normalization", action="store_true")
parser.add_argument("-ha", "--harmony", dest="harmony", action="store_true")
parser.add_argument(
    "-l", "--log", dest="log", default="off", choices=["short", "long", "off"]
)
parser.add_argument("-it", "--iterations", dest="iterations", type=int, default=5)
parser.add_argument(
    "-r",
    "--resolutions",
    dest="resolutions",
    nargs="+",
    type=float,
    default=[0.5, 0.8, 1.0, 2.0],
)
args = parser.parse_args()

warnings.filterwarnings("ignore", category=ImplicitModificationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def logging_setup(output_path: str) -> None:
    os.makedirs(f"{output_path}/logs", exist_ok=True)
    logging.basicConfig(
        filename=f'{output_path}/logs/{time.strftime("%Y-%m-%d")}_louvain.log',
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def preprocessing(input_path, markers, pca, normalization, arcsine, harmony):
    df = pd.read_csv(input_path)
    adata = ad.AnnData(X=df[markers], obs=df.drop(columns=markers))
    if arcsine:
        adata.X = np.arcsinh(adata.X)

    if normalization:
        sc.pp.normalize_total(adata, target_sum=1)

    if pca:
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_pca")
    elif harmony:
        sc.pp.pca(adata, n_comps=30)
        sc.external.pp.harmony_integrate(
            adata, "sample_id", basis="X_pca", adjusted_basis="X_harmony"
        )
        sc.pp.neighbors(adata, n_neighbors=10, use_rep="X_harmony")
    else:
        sc.pp.neighbors(adata, n_neighbors=10)
    return adata, df


def _run_louvain(adata, resolution: float, key: str) -> None:
    if hasattr(sc.tl, "louvain"):
        sc.tl.louvain(adata, resolution=resolution, key_added=key)
        return

    import igraph as ig

    g = adata.obsp["connectivities"]
    sources, targets = g.nonzero()
    weights = np.asarray(g[sources, targets]).ravel()
    graph = ig.Graph(n=adata.n_obs, edges=list(zip(sources, targets)), directed=False)
    if weights.size:
        graph.es["weight"] = weights
        membership = graph.community_multilevel(weights="weight").membership
    else:
        membership = graph.community_multilevel().membership
    adata.obs[key] = pd.Categorical([str(m) for m in membership])


def louvain_with_greedy(adata, df, iterations, output_path, log, resolutions):
    for iteration in range(1, iterations + 1):
        for res in resolutions:
            key = f"louvain_res{str(res).replace('.', '_')}"

            def _cluster():
                _run_louvain(adata, res, key)

            elapsed = timed_cluster_run(
                _cluster,
                log=log,
                logger=logging.getLogger(),
                message=f"Louvain resolution={res}",
            )
            output = df.copy()
            output[key] = adata.obs[key].values
            resolution_dir = f"{output_path}/{key}"
            write_greedy_predictions(
                output,
                key,
                resolution_dir,
                iteration,
                log=log,
                logger=logging.getLogger(),
                timing_path=os.path.join(resolution_dir, "fold_times.txt"),
                elapsed_sec=elapsed,
            )


def main():
    if args.log != "off":
        logging_setup(args.output_path)
    adata, df = preprocessing(
        args.input, args.markers, args.PCA, args.normalization, args.arcsine, args.harmony
    )
    louvain_with_greedy(
        adata, df, args.iterations, args.output_path, args.log, args.resolutions
    )


if __name__ == "__main__":
    main()
