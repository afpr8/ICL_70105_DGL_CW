import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

from collections import Counter
import community as louvain
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from MatrixVectorizer import MatrixVectorizer

THRESHOLD_PERCENTILE = 80
LOUVAIN_SEED         = 42

mv = MatrixVectorizer()


def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), '../../../data')
    hr = pd.read_csv(os.path.join(data_dir, 'hr_train.csv')).to_numpy()
    lr = pd.read_csv(os.path.join(data_dir, 'lr_train.csv')).to_numpy()
    return lr, hr


def inspect_subjects(vecs, n_nodes, n=3):
    for i in range(n):
        A = mv.anti_vectorize(vecs[i], n_nodes)
        G = nx.from_numpy_array(A)
        print(f"Subject {i}:")
        print(f"  Edge weight range: {A.min():.4f} - {A.max():.4f}")
        print(f"  Mean edge weight:  {A.mean():.4f}")
        print(f"  Density:           {nx.density(G):.4f}")
        print(f"  Non-zero edges:    {np.count_nonzero(A)}")


def compute_k_distribution(vecs, n_nodes, label):
    k_values = []
    q_values = []

    for i in range(len(vecs)):
        A = mv.anti_vectorize(vecs[i], n_nodes)
        threshold = np.percentile(A[A > 0], THRESHOLD_PERCENTILE)
        A_thresh = A.copy()
        A_thresh[A_thresh < threshold] = 0

        G = nx.from_numpy_array(A_thresh)
        partition = louvain.best_partition(G, random_state=LOUVAIN_SEED)
        k_values.append(len(set(partition.values())))
        q_values.append(louvain.modularity(partition, G))

    k_arr = np.array(k_values)
    q_arr = np.array(q_values)
    counts = Counter(k_values)

    print(f"\n{label} Distribution:")
    for k, count in sorted(counts.items()):
        print(f"  K={k:2d}: {count:3d} subjects ({100*count/len(k_arr):.1f}%)")
    print(f"{label} Mode:   {counts.most_common(1)[0][0]}")
    print(f"{label} Mean:   {k_arr.mean():.2f}")
    print(f"{label} Median: {np.median(k_arr):.1f}")
    print(f"{label} Std:    {k_arr.std():.2f}")
    print(f"{label} Mean Q: {q_arr.mean():.3f} ± {q_arr.std():.3f}")

    return k_arr, q_arr


def plot_distributions(lr_k, hr_k):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, k_arr, label, color in zip(
        axes,
        [lr_k, hr_k],
        ["LR (160 nodes)", "HR (268 nodes)"],
        ["steelblue", "coral"]
    ):
        counts = Counter(k_arr)
        ks = sorted(counts.keys())
        ax.bar(ks, [counts[k] for k in ks], color=color, edgecolor='black', alpha=0.8)
        ax.axvline(np.median(k_arr), color='black', linestyle='--', linewidth=1.5,
                   label=f'Median={np.median(k_arr):.0f}')
        ax.axvline(counts.most_common(1)[0][0], color='red', linestyle='-', linewidth=1.5,
                   label=f'Mode={counts.most_common(1)[0][0]}')
        ax.set_xlabel("Number of communities K")
        ax.set_ylabel("Number of subjects")
        ax.set_title(f"K distribution — {label}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(__file__), 'k_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.show()
    print("\nPlot saved to k_distribution.png")


def main():
    lr_vecs, hr_vecs = load_data()

    print("=" * 50)
    print("LR")
    print("=" * 50)
    inspect_subjects(lr_vecs, 160)
    lr_k, _ = compute_k_distribution(lr_vecs, 160, "LR")
    
    print("=" * 50)
    print("HR")
    print("=" * 50)
    inspect_subjects(hr_vecs, 268)   
    hr_k, _ = compute_k_distribution(hr_vecs, 268, "HR")

    plot_distributions(lr_k, hr_k)


if __name__ == "__main__":
    main()
