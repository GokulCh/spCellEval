"""
run_stellar_table.py
====================
STELLAR-style supervised GNN on a quantification TABLE (no images, no segmentation masks, no hard-coded dataset paths).

Same model and training recipe as run_stellar.py (Julia Oesterle): per-image spatial neighbour graph from the cell
centroids, StellarModel = Linear(in->hid) + ReLU + SAGEConv(hid->hid) + Linear(hid->classes), cross-entropy with
inverse-frequency class weights. What differs is only the data handling:
  - expression, x/y, image id and labels come from the pipeline's quantification table (markers + spc_row + x, y, image, label)
  - the folds are the pipeline's fold files (fold_<k>_train.csv / fold_<k>_test.csv, read for their spc_row ids)
  - the graph is built over ALL cells of an image; only the train cells contribute to the loss and only the test cells
    are written out (labels are never used as inputs, so nothing leaks)
  - whole image graphs are the training batches (no NeighborLoader, so no pyg-lib / torch-sparse needed)

Output: <output_dir>/predictions_fold_<k>.csv  (k = 1..5, the fold file numbers)
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from sklearn.metrics import f1_score
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv


class StellarModel(nn.Module):
    def __init__(self, input_dim: int, hid_dim: int, num_classes: int):
        super().__init__()
        self.input_linear = nn.Linear(input_dim, hid_dim)
        self.graph_conv = SAGEConv(hid_dim, hid_dim)
        self.fc_net = nn.Linear(hid_dim, num_classes)

    def forward(self, data):
        feat = F.relu(self.input_linear(data.x))
        return self.fc_net(self.graph_conv(feat, data.edge_index))


def build_graphs(q: pd.DataFrame, a) -> tuple[list[np.ndarray], list[np.ndarray], float]:
    """One graph per image over all its cells. Returns (row positions per image, edge_index per image, threshold used)."""
    groups = [np.asarray(ix) for ix in q.groupby(a.image_col, sort=False).indices.values()]
    pos = q[[a.x_col, a.y_col]].to_numpy(np.float32)
    trees = [cKDTree(pos[ix]) for ix in groups]
    r = a.distance_threshold
    if r <= 0:      # auto: 3x the median nearest-neighbour distance (about 14 um on CODEX at 0.377 um/px, the original 14.28)
        nn_d = np.concatenate([t.query(pos[ix], k=2)[0][:, 1] for t, ix in zip(trees, groups) if len(ix) > 1])
        r = 3.0 * float(np.median(nn_d))
    edges = []
    for t in trees:
        p = t.query_pairs(r, output_type="ndarray")
        edges.append(np.concatenate([p, p[:, ::-1]]).T.astype(np.int64))
    return groups, edges, r


def run_fold(k: int, X: np.ndarray, y: np.ndarray, groups, edges, tr: np.ndarray, te: np.ndarray, classes: np.ndarray, q: pd.DataFrame, a):
    torch.manual_seed(a.seed + k)
    np.random.seed(a.seed + k)
    dev = torch.device(a.device if a.device == "cpu" or torch.cuda.is_available() else "cpu")
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8                      # train statistics only
    Xs = ((X - mu) / sd).astype(np.float32)
    is_tr, is_te = np.zeros(len(X), bool), np.zeros(len(X), bool)
    is_tr[tr], is_te[te] = True, True
    graphs = [Data(x=torch.from_numpy(Xs[ix]), edge_index=torch.from_numpy(e), y=torch.from_numpy(y[ix]),
                   train_mask=torch.from_numpy(is_tr[ix]), test_mask=torch.from_numpy(is_te[ix]), ix=torch.from_numpy(ix))
              for ix, e in zip(groups, edges)]
    train_g = [g for g in graphs if g.train_mask.any()]
    test_g = [g for g in graphs if g.test_mask.any()]

    n_cls = len(classes)
    cnt = np.bincount(y[tr], minlength=n_cls)
    w = torch.tensor(cnt.sum() / (n_cls * np.maximum(cnt, 1)), dtype=torch.float32, device=dev)
    model = StellarModel(X.shape[1], a.hid_dim, n_cls).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    loader = DataLoader(train_g, batch_size=a.image_batch_size, shuffle=True)
    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        tot = 0.0
        for b in loader:
            b = b.to(dev)
            opt.zero_grad()
            loss = F.cross_entropy(model(b)[b.train_mask], b.y[b.train_mask], weight=w)
            loss.backward()
            opt.step()
            tot += loss.item()
        if ep == 0 or (ep + 1) % 10 == 0 or ep == a.epochs - 1:
            print(f"  fold {k} epoch {ep + 1}/{a.epochs} loss={tot / len(loader):.4f} ({time.time() - t0:.0f}s)", flush=True)

    model.eval()
    rows, conf, pred = [], [], []
    with torch.no_grad():
        for g in test_g:
            g = g.to(dev)
            p = F.softmax(model(g)[g.test_mask], dim=1).cpu()
            rows.append(g.ix[g.test_mask].cpu().numpy())
            conf.append(p.max(1).values.numpy())
            pred.append(p.argmax(1).numpy())
    rows, conf, pred = np.concatenate(rows), np.concatenate(conf), np.concatenate(pred)
    out = pd.DataFrame({"spc_row": q["spc_row"].to_numpy()[rows], "image": q[a.image_col].to_numpy()[rows],
                        "x": q[a.x_col].to_numpy()[rows], "y": q[a.y_col].to_numpy()[rows],
                        "true_phenotype": classes[y[rows]], "predicted_phenotype": classes[pred], "confidence": conf})
    out.to_csv(Path(a.output_dir) / f"predictions_fold_{k}.csv", index=False)
    print(f"  fold {k}: macro F1 = {f1_score(y[rows], pred, average='macro', labels=np.arange(n_cls), zero_division=0):.4f} "
          f"({len(rows):,} test cells, {time.time() - t0:.0f}s)", flush=True)


def main():
    p = argparse.ArgumentParser(description="Supervised STELLAR GNN on a quantification table")
    p.add_argument("--quant", required=True, help="quantification csv: markers + spc_row + image + x + y + label column")
    p.add_argument("--folds_dir", required=True, help="folder with fold_<k>_train.csv / fold_<k>_test.csv (read for spc_row)")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--label_col", default="cell_type")
    p.add_argument("--image_col", default="image")
    p.add_argument("--x_col", default="x")
    p.add_argument("--y_col", default="y")
    p.add_argument("--hid_dim", type=int, default=160)
    p.add_argument("--distance_threshold", type=float, default=0.0, help="graph radius in x/y units; 0 = auto from the data")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--weight_decay", type=float, default=0.0001)
    p.add_argument("--image_batch_size", type=int, default=16, help="image graphs per training step")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--markers", nargs="+", required=True, help="marker columns (put this option last)")
    a = p.parse_args()
    Path(a.output_dir).mkdir(parents=True, exist_ok=True)

    q = pd.read_csv(a.quant, usecols=["spc_row", a.image_col, a.x_col, a.y_col, a.label_col, *a.markers])
    X = q[a.markers].to_numpy(np.float32)
    classes = np.array(sorted(q[a.label_col].astype(str).unique()))
    y = pd.Categorical(q[a.label_col].astype(str), categories=classes).codes.astype(np.int64)
    groups, edges, r = build_graphs(q, a)
    print(f"STELLAR table mode: {len(q):,} cells, {len(groups)} images, {len(classes)} classes, "
          f"{sum(e.shape[1] for e in edges):,} directed edges (radius {r:.2f}), device={a.device}", flush=True)
    pos_of = pd.Series(np.arange(len(q)), index=q["spc_row"].to_numpy())

    folds = sorted(Path(a.folds_dir).glob("fold_*_train.csv"), key=lambda f: int(f.stem.split("_")[1]))
    for f in folds:
        k = int(f.stem.split("_")[1])
        te_file = f.with_name(f"fold_{k}_test.csv")
        tr = pos_of.loc[pd.read_csv(f, usecols=["spc_row"]).spc_row.to_numpy()].to_numpy()
        te = pos_of.loc[pd.read_csv(te_file, usecols=["spc_row"]).spc_row.to_numpy()].to_numpy()
        run_fold(k, X, y, groups, edges, tr, te, classes, q, a)


if __name__ == "__main__":
    main()
