# GNN evaluation metric functions

# Standard library imports
import warnings

# Third party imports
import networkx as nx
import numpy as np
from scipy.stats import pearsonr
from scipy.sparse.csgraph import shortest_path as sp_shortest_path
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error
import torch
from torch.utils.data import DataLoader

# Local imports
from src.datasets import BrainDataset
from src.matrix_vectorizer import MatrixVectorizer
from src.utils.core_utils import get_device
from src.utils.data_utils import prepare_tensors
from src.utils.model_args import BaseModelArgs

DEVICE, PIN_MEMORY = get_device()

# Filter out the specific deprecation warnings from pyparsing
#   This is because matplotlib is using a deprecated pyparsing feature, it is
#   cluttering test results
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pyparsing")
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")


def get_metrics(
    output_graphs_arr: list[tuple[torch.Tensor, torch.Tensor]],
    gt_centralities_cache: list[dict[str, list]] | None = None,
    final_metrics: bool=False
) -> dict[str, float]:
    """
    Compute evaluation metrics for predicted vs ground truth graphs.
    Metrics include:
        - MAE (flattened adjacency matrices)
        - Pearson correlation coefficient (PCC)
        - Jensen-Shannon distance
        - (Optional) centrality MAEs & other chosen MAEs

    Params:
        output_graphs_arr: List of (prediction_batch, ground_truth_batch) pairs
            Each tensor should have shape (B, N, N)
        gt_centralities_cache:
            Precomputed centrality measures for ground truth graphs
            Required if final_metrics=True and centrality comparison is desired
        final_metrics: Whether to compute centrality-based metrics in addition
            to adjacency-based metrics
    Returns:
        metrics: Dictionary mapping metric names to scalar values
    """
    if final_metrics and gt_centralities_cache is None:
        raise ValueError(
            "gt_centralities_cache must be provided when final_metrics=True "
            "to avoid expensive recomputation of ground-truth centralities."
        )

    # Initialize lists to store MAEs for each centrality measure
    centrality_errors = {
        k: [] for k in ["bc", "ec", "pc", "glob_eff", "modularity"]
    }

    pred_vectors = []
    gt_vectors = []

    sample_idx = 0  # tracks position into gt_centralities_cache

    # Iterate over each test sample
    for pred_batch, gt_batch in output_graphs_arr:
        pred_batch = pred_batch.detach().cpu().numpy()
        gt_batch = gt_batch.detach().cpu().numpy()

        for b in range(pred_batch.shape[0]):
            pred_mat = pred_batch[b]
            gt_mat = gt_batch[b]

            pred_vectors.append(MatrixVectorizer.vectorize(pred_mat))
            gt_vectors.append(MatrixVectorizer.vectorize(gt_mat))

            if final_metrics:
                gt_cent = gt_centralities_cache[sample_idx]

                bc_err, ec_err, pc_err, ge_e, mod_e = (
                    _compute_centrality_errors(
                        pred_mat,
                        gt_mat,
                        gt_cent,
                    )
                )

                centrality_errors["bc"].append(bc_err)
                centrality_errors["ec"].append(ec_err)
                centrality_errors["pc"].append(pc_err)
                centrality_errors["glob_eff"].append(ge_e)
                centrality_errors["modularity"].append(mod_e)

            sample_idx += 1

    pred_1d = np.concatenate(pred_vectors)
    gt_1d = np.concatenate(gt_vectors)

    metrics = _compute_adjacency_metrics(pred_1d, gt_1d)

    if final_metrics:
        metrics.update({
            "mae_bc": np.mean(centrality_errors["bc"]),
            "mae_ec": np.mean(centrality_errors["ec"]),
            "mae_pc": np.mean(centrality_errors["pc"]),
            "mae_glob_eff": np.mean(centrality_errors["glob_eff"]),
            "mae_modularity": np.mean(centrality_errors["modularity"]),
        })

    return metrics


def precompute_gt_centralities(
    hr_data: np.ndarray
) -> list[dict[str, list]]:
    """
    Precompute centrality measures for all ground-truth graphs

    Params:
        hr_data: Array of shape (N, H, W) containing adjacency matrices
    Returns:
        gt_cache: List where each element contains:
            {
                "bc": betweenness centrality values,
                "ec": eigenvector centrality values,
                "pc": PageRank values
                "glob_eff": Measures network integration capacity
                "modularity": Measures Q modularity
            }
    """
    print("Computing cache...")
    gt_cache = []
    for mat in hr_data:
        mat_sparse = sparsify_adj(mat, threshold_pct=80.0)
        g = nx.from_numpy_array(mat_sparse, edge_attr="weight")
        gt_cache.append({
            "bc": list(nx.betweenness_centrality(g, weight="weight").values()),
            "ec": list(nx.eigenvector_centrality(
                g,
                weight="weight",
                max_iter=1000
            ).values()),
            "pc": list(nx.pagerank(g, weight="weight").values()),
            "glob_eff": compute_global_efficiency(g),
            "modularity": compute_modularity(g)
        })
    return gt_cache


def _compute_centrality_errors(
    pred_mat: np.ndarray,
    gt_mat: np.ndarray,
    gt_cent: dict[str, list[float]] | None = None,
) -> tuple[float, float, float]:
    """
    Compute centrality-based mean absolute errors (MAEs) between a
    predicted graph and its corresponding ground-truth graph.
    The following node-level centrality measures are evaluated:
        - Betweenness centrality (BC)
        - Eigenvector centrality (EC)
        - PageRank centrality (PC)
        - Global Efficiency: Measures network integration capacity
        - Average Clustering Coefficient: Measures local network modularity
    If precomputed ground-truth centralities are provided via `gt_cent`,
    they are used directly. Otherwise, centralities for the ground-truth
    graph are computed on the fly.

    Params:
        pred_mat: Predicted adjacency matrix of shape (N, N)
            Must represent a weighted, undirected graph
        gt_mat: Ground-truth adjacency matrix of shape (N, N)
            Must represent a weighted, undirected graph
        gt_cent: Optional dictionary containing precomputed ground-truth
            centralities with keys:
                - "bc": List[float] (length N)
                - "ec": List[float] (length N)
                - "pc": List[float] (length N)
                - "glob_eff": List[float] (length N)
                - "modularity": List[float] (length N)
            If None, centralities will be computed from `gt_mat`
    Returns:
        Tuple containing:
            - mae_bc: Mean absolute error of betweenness centrality
            - mae_ec: Mean absolute error of eigenvector centrality
            - mae_pc: Mean absolute error of PageRank centrality
            - err_glob_eff: Absolute error of global efficiency
            - err_modularity: Absolute error of Q modularity
    Raises:
        ValueError: If `pred_mat` and `gt_mat` have mismatched shapes
    """
    pred_mat_sparse = sparsify_adj(pred_mat, threshold_pct=80.0)
    gt_mat_sparse = sparsify_adj(gt_mat, threshold_pct=80.0)

    pred_graph = nx.from_numpy_array(pred_mat_sparse, edge_attr="weight")

    pred_bc = list(
        nx.betweenness_centrality(pred_graph, weight="weight").values()
    )
    pred_ec = list(
        nx.eigenvector_centrality(
            pred_graph,
            weight="weight",
            max_iter=1000
        ).values()
    )
    pred_pc = list(
        nx.pagerank(pred_graph, weight="weight").values()
    )
    pred_glob_eff = compute_global_efficiency(pred_graph)
    pred_mod = compute_modularity(pred_graph)

    if gt_cent is not None:
        gt_bc = gt_cent["bc"]
        gt_ec = gt_cent["ec"]
        gt_pc = gt_cent["pc"]
        gt_glob_eff = gt_cent["glob_eff"]
        gt_mod = gt_cent["modularity"]
    else:
        gt_graph = nx.from_numpy_array(gt_mat_sparse, edge_attr="weight")
        gt_bc = list(
            nx.betweenness_centrality(gt_graph, weight="weight").values()
        )
        gt_ec = list(
            nx.eigenvector_centrality(
                gt_graph,
                weight="weight",
                max_iter=1000
            ).values()
        )
        gt_pc = list(
            nx.pagerank(gt_graph, weight="weight").values()
        )
        gt_glob_eff = compute_global_efficiency(gt_graph)
        gt_mod = compute_modularity(gt_graph)

    return (
        mean_absolute_error(pred_bc, gt_bc),
        mean_absolute_error(pred_ec, gt_ec),
        mean_absolute_error(pred_pc, gt_pc),
        abs(pred_glob_eff - gt_glob_eff),
        abs(pred_mod - gt_mod)
    )


def _compute_adjacency_metrics(
    pred_1d: np.ndarray,
    gt_1d: np.ndarray
) -> dict[str, float]:
    # Normalize - Jensen-Shannon assumes probabbility distributions
    eps = 1e-12 # Avoid div by 0
    pred_1d_norm = pred_1d / (pred_1d.sum() + eps)
    gt_1d_norm = gt_1d / (gt_1d.sum() + eps)

    # Pearson-R will throw an error in the rare case of 0 variance
    try:
        pcc = pearsonr(pred_1d, gt_1d)[0]
    except Exception:
        pcc = 0.0

    return {
        "mae": mean_absolute_error(pred_1d, gt_1d),
        "pcc": pcc,
        "js_distance": jensenshannon(pred_1d_norm, gt_1d_norm),
    }


def compute_modularity(g: nx.Graph | nx.DiGraph) -> float:
    """
    Computes the modularity (Q) of a graph represented by an adjacency matrix
    Modularity measures the strength of division of a network into modules 
        (communities). Networks with high modularity have dense connections 
        between the nodes within modules but sparse connections between nodes in
        different modules
    
    Params:
        g: A NetworkX Graph or DiGraph object
           The function internally simplifies the graph to an undirected,
           simple graph for community detection
    Returns:
        q: The modularity score. Ranges roughly from -0.5 to 1
           Returns 0.0 if the graph has no edges or is empty
    Note:
        This implementation uses the Label Propagation algorithm to detect 
        communities, which is computationally efficient for large graphs.
        Note that this algorithm is stochastic and may produce slightly
        different partitions across multiple runs.
    """
    # Simplify graph for community detection
    g_simple = nx.Graph(g)
    g_simple.remove_edges_from(nx.selfloop_edges(g_simple))
    
    if g_simple.number_of_edges() == 0:
        q = 0.0
    else:
        # Use your chosen community detection algorithm
        communities = nx.community.label_propagation_communities(g_simple)
        q = nx.community.modularity(g_simple, communities)

    return q


def compute_global_efficiency(mat: np.ndarray) -> float:
    """
    Computes global efficiency using SciPy's optimized shortest path
    Assumes an unweighted graph structure for path calculation
    """
    if isinstance(mat, (nx.Graph, nx.DiGraph)):
        # If it's a graph, convert it to a numpy adjacency matrix
        A = nx.to_numpy_array(mat)
    else:
        # If it's already a numpy array, use it directly
        A = mat

    # Match the colleague's inversion: 1.0/weight
    with np.errstate(divide='ignore', invalid='ignore'):
        dist_matrix = np.where(A > 0, 1.0 / A, 0.0)
    
    # Use unweighted=False because we are passing a distance matrix
    dist = sp_shortest_path(dist_matrix, directed=False, unweighted=False)
    
    # Global efficiency calculation
    # 1.0 / distance, treating infinity (unreachable) as 0.0
    with np.errstate(divide='ignore'):
        inv_dist = 1.0 / dist
    inv_dist[~np.isfinite(inv_dist)] = 0.0
    np.fill_diagonal(inv_dist, 0.0)
    
    n = A.shape[0]

    return inv_dist.sum() / (n * (n - 1))

def sparsify_adj(A: np.ndarray, threshold_pct: float = 80.0) -> np.ndarray:
    """
    Matches the discriminator's internal sparsification
    """
    # Ensure it's square
    assert A.shape[0] == A.shape[1]
    mask = ~np.eye(A.shape[0], dtype=bool)
    threshold = np.percentile(A[mask], threshold_pct)

    return np.where(A >= threshold, A, 0.0)


def compute_metrics(
        model: torch.nn.Module,
        dataset: BrainDataset,
        args: BaseModelArgs
    ) -> dict[str, float]:
    """
    Compute evaluation metrics for a dataset using the generator model

    Params:
        model: Trained AGSRNet generator
        dataset: BrainDataset containing LR-HR pairs
        args: AGSRArgs with padding, lr_dim, hr_dim
    Returns:
        dict[str, float]: Computed metrics (e.g., MSE, PSNR) for the dataset
    """
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=PIN_MEMORY
    )
    model.eval()
    mats_arr = []
    with torch.no_grad():
        for lr_np, hr_np in loader:
            lr_t, padded_hr = prepare_tensors(
                lr_np.squeeze(0).numpy(),
                hr_np.squeeze(0).numpy(),
                args
            )
            raw_output = model(lr_t)
            preds = (
                raw_output[0]
                if isinstance(raw_output, (tuple, list)) else raw_output
            )
            mats_arr.append((preds.unsqueeze(0), padded_hr.unsqueeze(0)))
    return get_metrics(mats_arr, final_metrics=False)
