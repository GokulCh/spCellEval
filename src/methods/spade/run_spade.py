"""SPADE-inspired clustering: density-dependent downsampling + hierarchical clustering."""

import argparse
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors, KernelDensity

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_script_dir))
sys.path.insert(0, os.path.join(project_root, "methods", "utils"))

from clustering_benchmark import write_greedy_predictions  # noqa: E402

parser = argparse.ArgumentParser(description="SPADE-inspired clustering for protein panels.")
parser.add_argument("-i", "--input", dest="input", type=str, required=True)
parser.add_argument("-o", "--output", dest="output_path", type=str, required=True)
parser.add_argument("-m", "--markers", dest="markers", nargs="+", required=True)
parser.add_argument(
    "-l", "--log", dest="log", default="off", choices=["short", "long", "off"]
)
parser.add_argument("-it", "--iterations", dest="iterations", type=int, default=1)
parser.add_argument("-k", "--n_clusters", dest="n_clusters", type=int, default=20)
parser.add_argument(
    "-d", "--downsample", dest="downsample", type=float, default=0.3,
    help="Target fraction of cells to retain after density-dependent downsampling.",
)
args = parser.parse_args()

warnings.filterwarnings("ignore", category=FutureWarning)
CLUSTER_COL = "spade_cluster"


def logging_setup(output_path: str) -> None:
    os.makedirs(f"{output_path}/logs", exist_ok=True)
    logging.basicConfig(
        filename=f'{output_path}/logs/{time.strftime("%Y-%m-%d")}_spade.log',
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def density_downsample(
    features: np.ndarray,
    target_fraction: float,
    n_clusters: int,
    random_state: int = 42,
) -> np.ndarray:
    """Keep cells inversely proportional to local density (SPADE-style)."""
    n = features.shape[0]
    target_n = max(n_clusters * 3, int(n * target_fraction))
    target_n = min(target_n, n)

    bandwidth = max(1.0, np.sqrt(features.shape[1]))
    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    kde.fit(features)
    log_density = kde.score_samples(features)
    density = np.exp(log_density)
    weights = 1.0 / np.clip(density, 1e-12, None)
    weights /= weights.sum()

    rng = np.random.default_rng(random_state)
    chosen = rng.choice(n, size=target_n, replace=False, p=weights)
    return np.sort(chosen)


def spade_cluster(df: pd.DataFrame, markers, n_clusters: int, downsample: float) -> pd.Series:
    x = df[markers].to_numpy(dtype=float)
    n_components = min(30, x.shape[1], x.shape[0] - 1)
    reduced = PCA(n_components=n_components, random_state=42).fit_transform(x)

    sample_idx = density_downsample(reduced, downsample, n_clusters, random_state=42)
    sampled = reduced[sample_idx]

    model = AgglomerativeClustering(n_clusters=min(n_clusters, len(sampled)))
    sampled_labels = model.fit_predict(sampled)

    centroids = np.vstack([
        sampled[sampled_labels == label].mean(axis=0)
        for label in np.unique(sampled_labels)
    ])
    nn = NearestNeighbors(n_neighbors=1)
    nn.fit(centroids)
    _, indices = nn.kneighbors(reduced)
    label_map = {i: str(np.unique(sampled_labels)[idx]) for i, idx in enumerate(indices.flatten())}
    return pd.Series([label_map[i] for i in range(len(df))], index=df.index, name=CLUSTER_COL)


def spade_with_greedy(df, iterations, output_path, log):
    resolution_dir = f"{output_path}/{CLUSTER_COL}"
    for iteration in range(1, iterations + 1):
        start = time.time()
        clusters = spade_cluster(df, args.markers, args.n_clusters, args.downsample)
        elapsed = time.time() - start
        output = df.copy()
        output[CLUSTER_COL] = clusters.values
        write_greedy_predictions(
            output,
            CLUSTER_COL,
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
    df = pd.read_csv(args.input)
    spade_with_greedy(df, args.iterations, args.output_path, args.log)


if __name__ == "__main__":
    main()
