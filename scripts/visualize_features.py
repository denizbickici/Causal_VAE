#!/usr/bin/env python3
import argparse
import os
import sys

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def _to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    if isinstance(x, np.ndarray):
        return x
    return np.asarray(x)


def load_features(path, key, targets_key):
    if path.endswith(".npy"):
        feats = np.load(path)
        return feats, None

    obj = torch.load(path, map_location="cpu")
    if torch.is_tensor(obj) or isinstance(obj, np.ndarray):
        return _to_numpy(obj), None

    if not isinstance(obj, dict):
        raise ValueError("Unsupported feature format; expected tensor/ndarray or dict.")

    if key and key in obj:
        feats = obj[key]
    else:
        for fallback in ("feats", "cls_feats"):
            if fallback in obj:
                feats = obj[fallback]
                break
        else:
            raise KeyError(f"No feature key found in {path}.")

    targets = obj.get(targets_key)
    return _to_numpy(feats), _to_numpy(targets) if targets is not None else None


def pool_features(feats, pool):
    if feats.ndim == 1:
        return feats.reshape(1, -1)
    if feats.ndim == 2:
        return feats
    if pool == "none":
        raise ValueError("pool=none requires 2D features; use mean or flatten.")
    if pool == "flatten":
        return feats.reshape(feats.shape[0], -1)
    # default: mean across all dims except batch and last
    if feats.ndim == 3:
        return feats.mean(axis=1)
    axes = tuple(range(1, feats.ndim - 1))
    return feats.mean(axis=axes)


def reduce_dim(feats, method, seed, perplexity, n_iter, pca_dim):
    if method == "pca":
        reducer = PCA(n_components=2)
        return reducer.fit_transform(feats)
    if method == "tsne":
        if pca_dim and feats.shape[1] > pca_dim:
            feats = PCA(n_components=pca_dim).fit_transform(feats)
        reducer = TSNE(
            n_components=2,
            perplexity=perplexity,
            n_iter=n_iter,
            random_state=seed,
        )
        return reducer.fit_transform(feats)
    if method == "umap":
        try:
            import umap  # type: ignore
        except Exception as exc:
            raise ImportError("umap-learn is required for --method umap") from exc
        reducer = umap.UMAP(n_components=2, random_state=seed)
        return reducer.fit_transform(feats)
    raise ValueError(f"Unknown method: {method}")


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize extracted features.")
    parser.add_argument("--input", required=True, help="Path to .pt/.pth/.npy feature file.")
    parser.add_argument("--key", default="feats", help="Feature key for dict .pt files.")
    parser.add_argument("--targets-key", default="targets", help="Targets key for dict .pt files.")
    parser.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca")
    parser.add_argument("--pool", choices=["mean", "flatten", "none"], default="mean")
    parser.add_argument("--max-points", type=int, default=2000, help="0 to disable subsampling.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-standardize", action="store_true")
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--n-iter", type=int, default=1000)
    parser.add_argument("--pca-dim", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--title", default=None)
    parser.add_argument("--no-legend", action="store_true")
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--point-size", type=float, default=10.0)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print("matplotlib is required to render plots.", file=sys.stderr)
        raise

    feats, targets = load_features(args.input, args.key, args.targets_key)
    feats = pool_features(feats, args.pool).astype(np.float32)

    if targets is not None:
        targets = np.asarray(targets).reshape(-1)
        if targets.shape[0] != feats.shape[0]:
            print(
                f"Warning: targets length {targets.shape[0]} != feats {feats.shape[0]}.",
                file=sys.stderr,
            )
            targets = None

    if args.max_points and feats.shape[0] > args.max_points:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(feats.shape[0], size=args.max_points, replace=False)
        feats = feats[idx]
        if targets is not None:
            targets = targets[idx]

    if not args.no_standardize:
        feats = StandardScaler().fit_transform(feats)

    if args.method == "tsne" and args.perplexity >= feats.shape[0]:
        raise ValueError("perplexity must be smaller than the number of points.")

    proj = reduce_dim(
        feats,
        method=args.method,
        seed=args.seed,
        perplexity=args.perplexity,
        n_iter=args.n_iter,
        pca_dim=args.pca_dim,
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    if targets is None:
        ax.scatter(proj[:, 0], proj[:, 1], s=args.point_size, alpha=args.alpha)
    else:
        uniq = np.unique(targets)
        if len(uniq) <= 20:
            cmap = plt.get_cmap("tab20", len(uniq))
        else:
            cmap = plt.get_cmap("hsv", len(uniq))
        colors = np.searchsorted(uniq, targets)
        ax.scatter(
            proj[:, 0],
            proj[:, 1],
            c=colors,
            cmap=cmap,
            s=args.point_size,
            alpha=args.alpha,
            linewidths=0,
        )
        if not args.no_legend and len(uniq) <= 20:
            handles = [
                plt.Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    label=str(u),
                    markerfacecolor=cmap(i),
                    markersize=6,
                )
                for i, u in enumerate(uniq)
            ]
            ax.legend(handles=handles, title="targets", loc="best", frameon=False)

    ax.set_xlabel("dim-1")
    ax.set_ylabel("dim-2")
    if args.title:
        ax.set_title(args.title)
    else:
        ax.set_title(f"{args.method.upper()} projection ({proj.shape[0]} points)")

    out_path = args.output
    if not out_path:
        base = os.path.splitext(os.path.basename(args.input))[0]
        out_path = f"{base}_{args.method}.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
