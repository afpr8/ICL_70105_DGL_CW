# NeuroSRGAN — Neurodynamics-Informed Spectral Super-Resolution GAN

NeuroSRGAN is a graph neural network for brain connectome super-resolution:
given a low-resolution (LR) brain connectivity graph (160 nodes), it predicts
the corresponding high-resolution (HR) graph (268 nodes). It extends the
ArgsNet architecture with two novel neurodynamically motivated contributions:
(1) a community-aware spectral super-resolution layer that refines the global
spectral resolution jump with learned per-community residual corrections, and
(2) a topology-aware discriminator that evaluates the structural realism of
generated graphs using a compact neurodynamic topological fingerprint.

---

## High-Level Data Flow

```
LR adjacency (160×160)
        │
        ├─────────────────────────────────┐
        │                                 ▼
        │                       Louvain Clustering
        │                       (top-80th-percentile edges)
        │                       {C_1, ..., C_7}
        │                                 │
        │                                 ▼
        │                       HR community masks (268→320 padded)
        │                       mask_k = v_k · v_k^T ∈ {0,1}^{320×320}
        │                       (HR node i → LR node round(i×159/267))
        ▼                                 │
  normalize_adj_torch                     │
        │                                 │
        ▼                                 │
  GraphUnet                               │    ← hierarchical multi-scale
        │  net_outs (160×320)             │       feature extraction
        ▼                                 │
  CommunityAwareSRLayer                   │    ← spectral SR 160→320
        │  (wraps GSRLayer)               │
        └──────────────┬──────────────────┘
                       ▼
        Z_k = Z_global ⊙ mask_k          ← extract community blocks
                       │
                       ▼
        W_k = U_k · V_k^T                ← low-rank correction (r=16)
        correction_k = (Z_k @ W_k) ⊙ mask_k
                       │
                       ▼
        α = softmax([α_1, ..., α_7])     ← learned community weights
                       │
                       ▼
        Z_HR = Z_global                  ← residual addition    [NOVEL 1]
             + Σ_k α_k · correction_k
             (shape: 320×320)
                       │
                       ▼
  GraphConvolution × 2                   ← global HR refinement
        │
        ▼
  symmetrise + fill_diagonal(1)          ← enforce valid adjacency
        │
        ▼
HR adjacency prediction (320×320, padded)
        │
        ├── strip padding → (268×268) for output
        │
        ▼
  TopologyAwareDiscriminator             ← [NOVEL 2]
  input: concat([flatten(A_320), SWI, GE, Q])
  output: real/fake probability ∈ (0,1)
```

---

## Novel Contributions

### Contribution 1 — Community-Aware Spectral SR via Residual Community Modulation

ArgsNet's GSRLayer performs the LR → HR resolution jump via a single global
eigendecomposition, treating the brain graph as a homogeneous structure and
ignoring its well-established functional community organisation. We augment
the global spectral SR with learned per-community residual corrections:

```
Z_HR = Z_global + Σ_k α_k · (Z_global ⊙ mask_k · U_k V_k^T) ⊙ mask_k

where:
  Z_global        = GSRLayer(A_LR)         global spectral SR (unchanged)
  mask_k          = v_k · v_k^T            community k binary mask 320×320
  U_k, V_k        ∈ R^{320×16}            low-rank correction factors (r=16)
  α_k             = softmax(α)[k]          learned per-community weight
  K               = 7                      number of communities
```

The global GSRLayer captures cross-community connectivity. The per-community
corrections then independently refine intra-community edge weights, allowing
the model to learn community-specific LR → HR spectral mappings that reflect
the heterogeneous connectivity profiles of distinct functional brain networks.

The residual structure ensures corrections start near zero at initialisation
(V initialised to zeros), meaning training begins from the GSRNet solution and
learns corrections on top — making optimisation stable. Low-rank factorisation
(r=16) reduces parameters from 102,400 to 10,240 per community.

**Community detection:** Louvain clustering (`python-louvain`) is applied to
each LR subject graph after thresholding to the top-80th-percentile edges.
K=7 communities are used. HR node i is mapped to the same community as
LR node `round(i × 159/267)`. Masks are zero-padded from 268×268 to 320×320
to match the padded HR adjacency. All masks are precomputed once before
training and cached per-subject.

### Contribution 2 — Topology-Aware Adversarial Training

ArgsNet's discriminator evaluates realism purely from edge weight values and
can be fooled by graphs with plausible weight distributions but unrealistic
topology. We augment the discriminator with a compact topological fingerprint
computed from the predicted HR adjacency:

```
τ(A) = [SWI(A), GE(A), Q(A)]

D(A) = Sigmoid(MLP(concat([flatten(A_320), τ(A)])))

where:
  SWI = (C / C_rand) / (L / L_rand)           small-world index
  GE  = 1/N(N-1) · Σ_{i≠j} 1/d(i,j)          global efficiency
  Q   = modularity via label propagation        community segregation
```

The three features are grounded in the segregation-integration framework of
network neuroscience: SWI captures local small-world structure, GE captures
global integration efficiency, and Q captures functional community segregation.

**Implementation details:**
- Topology is computed on the core 268×268 submatrix (padding stripped via
  `to_networkx`, which slices `A[26:294, 26:294]`)
- Shortest paths use scipy's C implementation (`scipy.sparse.csgraph.shortest_path`)
  for speed; NetworkX `average_clustering` is used for C
- Modularity Q uses `nx.community.label_propagation_communities` (faster than Louvain)
- GT topology features are precomputed and cached before training to avoid
  recomputation each epoch
- For predicted outputs fed to the discriminator, topology is computed fresh
  (inside `torch.no_grad`)

---

## How the Two Contributions Relate

The two contributions together address a single coherent limitation of existing
brain graph SR methods — topology is ignored at two distinct points:

```
Point 1 — During generation:
        GSRLayer treats all edges equally regardless of
        community membership
        → Fixed by Contribution 1: residual community corrections

Point 2 — During training:
        Discriminator evaluates realism from edge weights only
        → Fixed by Contribution 2: topology-aware discriminator
```

---

## Four Variants (Ablation Study)

```
Variant "full" — NeuroSRGAN (final model):
        CommunityAwareSRLayer + TopologyAwareDiscriminator
        Both contributions combined

Variant "community_only" — NeuroSRGAN-SR:
        CommunityAwareSRLayer + StandardDiscriminator
        Isolates Contribution 1

Variant "topology_only" — NeuroSRGAN-D:
        GSRLayer + TopologyAwareDiscriminator
        Isolates Contribution 2

Variant "baseline" — ≈ ArgsNet:
        GSRLayer + StandardDiscriminator
        No novel contributions
```

Selected via `ChrisNetArgs(variant=...)` in `config.py`.

---

## Components

### 1. `normalize_adj_torch` — Symmetric Adjacency Normalisation
```
Â = D^{-1/2} A D^{-1/2}
```
Unchanged from ArgsNet. Bounds spectral radius to [-1,1] for stable gradient
flow during propagation.

---

### 2. `GraphUnet` — Hierarchical Encoder-Decoder
Unchanged from ArgsNet. Extracts multi-scale LR features via alternating
GraphPool (learnable top-k coarsening) and GraphUnpool (index-based restoration)
with skip connections. Pool ratios: [0.9, 0.7, 0.6, 0.5].
Output `net_outs` has shape 160×320 — each of the 160 LR nodes carries a
320-dimensional feature vector for use by the SR layer.

---

### 3. `CommunityAwareSRLayer` — Residual Community Modulation [NOVEL 1]

Wraps and extends the original GSRLayer.

```python
class CommunityAwareSRLayer(torch.nn.Module):
    def __init__(self, hr_dim: int, k: int, rank: int = 16) -> None:
        super().__init__()
        self.gsr = GSRLayer(hr_dim)           # unchanged from AGSRNet
        self.Us = torch.nn.ParameterList([
            torch.nn.Parameter(torch.randn(hr_dim, rank) * 0.001) for _ in range(k)
        ])
        self.Vs = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros(hr_dim, rank)) for _ in range(k)
        ])
        self.alphas = torch.nn.Parameter(torch.ones(k) / k)

    def forward(
            self, A: torch.Tensor, X: torch.Tensor, masks: list[torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
        adj, _ = self.gsr(A, X)               # global spectral SR

        out_adj = adj.clone()
        alpha = torch.softmax(self.alphas, dim=0)
        for i in range(self.k):
            W_k = self.Us[i] @ self.Vs[i].T   # (hr_dim × hr_dim) low-rank
            Z_k = adj * masks[i]               # community-masked adjacency
            correction_k = (Z_k @ W_k) * masks[i]
            out_adj = out_adj + alpha[i] * correction_k

        X_out = out_adj @ out_adj.T
        X_out = symmetrize_with_identity(X_out)
        return out_adj, torch.abs(X_out)
```

Note: V initialised to zeros ensures corrections are zero at init. U initialised
with small random noise (×0.001) for symmetry breaking.

---

### 4. `GraphConvolution` × 2 — HR Graph Refinement
Unchanged from ArgsNet. Two stacked Kipf & Welling GCN layers smooth local
inconsistencies in Z_HR by propagating features across the estimated HR graph:

```
H^(1) = ReLU( out_adj · Z · W_1 )
H^(2) = ReLU( out_adj · H^(1) · W_2 )
```

---

### 5. Post-processing
```
z = (z + z.T) / 2          ← symmetrise
z.fill_diagonal_(1)         ← self-connections set to 1
output = |z|                ← absolute value (non-negative)
```
Note: no hard clipping. The 320×320 padded output has the 26-pixel border
stripped to yield the final 268×268 HR prediction at test time.

---

### 6. `TopologyAwareDiscriminator` — Neurodynamic Critic [NOVEL 2]

```python
class TopologyAwareDiscriminator(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # input: flattened 320×320 matrix + 3 topology scalars = 102403
        self.dense_1 = Dense(args.hr_dim * args.hr_dim + 3, args.hr_dim, args)
        self.relu_1  = torch.nn.ReLU(inplace=False)
        self.dense_2 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_2  = torch.nn.ReLU(inplace=False)
        self.dense_3 = Dense(args.hr_dim, 1, args)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(
            self, inputs: torch.Tensor, topo: torch.Tensor | None = None
        ) -> torch.Tensor:
        if topo is None:
            with torch.no_grad():
                G = to_networkx(inputs)        # strips padding: A[26:294, 26:294]
                swi, ge, q = compute_topo_features(G)
                topo = torch.tensor([swi, ge, q], dtype=torch.float32)
        x = torch.cat(
            [inputs.flatten(), topo.to(inputs.device)]
        ).unsqueeze(0)                         # (1, hr_dim² + 3)
        dc_den1 = self.relu_1(self.dense_1(x))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))
        return torch.abs(self.sigmoid(self.dense_3(dc_den2)))  # (1, 1)
```

During training:
- **`d_real`** (model output): topo computed fresh from predicted adjacency
- **`d_fake`** (noisy GT): topo from precomputed GT cache (`gt_topo`)

---

### 7. `StandardDiscriminator` — ArgsNet-Style Critic
Used in `community_only` and `baseline` variants. Operates row-by-row on the
320×320 matrix (same as ArgsNet's original Discriminator). Output shape: (320, 1).

---

### 8. `gaussian_noise_layer` — Stochastic Regularisation
Applied to ground-truth HR adjacency before discriminator input. Symmetrises
and fills diagonal with 1.

**Current config:** `std_gaussian=0.0` — noise is disabled (effectively
passes GT through unchanged with symmetrisation only).

---

## Training Objective

```
Generator loss:
L_G = lmbda · MSE(net_outs, start_gcn_outs)   ← self-reconstruction
    + MSE(gsr_weights, U_hr)                   ← spectral alignment
    + MSE(model_outputs, padded_hr)             ← HR reconstruction
    + BCE(D(noisy_GT), 1)                       ← adversarial (fool D)

Discriminator loss:
L_D = BCE(D(model_output), 1)                  ← real = model prediction
    + BCE(D(noisy_GT), 0)                       ← fake = noisy GT

Training loop (per subject, shuffled each epoch):
  1. Forward pass: Z_HR = G(A_LR, masks)
  2. Compute L_D, backward, update D
  3. Compute L_G, backward, update G
```

Note: `d_real` and `d_fake` labels are swapped relative to standard GAN
convention (matching original ArgsNet behaviour): the model output is labelled
"real" and the (noisy) ground truth is labelled "fake".

---

## Hyperparameters

```python
lr            = 0.0001      # Adam learning rate (G and D)
epochs        = 200         # max epochs (early stopping may trigger earlier)
lmbda         = 16          # self-reconstruction loss weight
lr_dim        = 160         # LR node count
hr_dim        = 320         # padded HR dim (268 + 2×26)
padding       = 26          # zero-padding on each side
hidden_dim    = 320         # GCN hidden dimension
K_communities = 7           # Louvain communities
rank          = 16          # low-rank correction dimension
threshold_pct = 80          # percentile threshold for Louvain input graph
mean_dense    = 0.0
std_dense     = 0.01        # Dense layer weight init std
mean_gaussian = 0.0
std_gaussian  = 0.0         # noise disabled
```

---

## Training Details

- **Optimiser:** Adam (no weight decay) for both G and D
- **Data order:** index list shuffled each epoch via `np.random.shuffle` (masks stay in sync)
- **Mask caching:** community masks and GT topology features precomputed once per run before epoch 1
- **Seed:** 42 (covers Python, NumPy, PyTorch CPU/GPU, cuDNN)
- **Cross-validation:** 3-fold, `KFold(shuffle=True, random_state=42)`
- **CUDA determinism:** set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launching (required when `torch.use_deterministic_algorithms(True)` is active)

---

## Additional Evaluation Measures

Beyond the six required measures (MAE, PCC, JSD, MAE-PC, MAE-EC, MAE-BC),
NeuroSRGAN is additionally evaluated on:

**Global Efficiency (GE):** Average inverse shortest path length. Captures
network integration capacity. Computed via scipy + numpy.

**Modularity Q:** Strength of community structure in the predicted HR graph.
Captures functional segregation. Computed via label propagation.

Both reported as MAE between predicted and ground-truth values across subjects.

---

## Ablation Results

| Model | MAE | PCC | JSD | MAE-PC | MAE-EC | MAE-BC | MAE-GE | MAE-Q |
|---|---|---|---|---|---|---|---|---|
| ArgsNet (baseline) | - | - | - | - | - | - | - | - |
| community_only | - | - | - | - | - | - | - | - |
| topology_only | - | - | - | - | - | - | - | - |
| NeuroSRGAN (full) | - | - | - | - | - | - | - | - |

*To be completed with 3-fold CV results.*

---

## Summary Table

| Component | Role | vs ArgsNet |
|---|---|---|
| `normalize_adj_torch` | Symmetric normalisation | Unchanged |
| `GraphUnet` | Multi-scale LR feature extraction | Unchanged |
| `GSRLayer` | Global spectral SR 160→320 | Unchanged |
| `CommunityAwareSRLayer` | Residual community modulation | **Novel — wraps GSRLayer** |
| `GraphConvolution` × 2 | HR graph refinement | Unchanged |
| `StandardDiscriminator` | ArgsNet-style critic (ablation use) | Unchanged |
| `TopologyAwareDiscriminator` | Neurodynamic realism critic | **Novel — extends Discriminator** |
| `gaussian_noise_layer` | GAN training stabilisation (std=0 currently) | Unchanged |

---

## Requirements

```bash
pip install -r gpu_requirements.txt   # GPU (CUDA 12.1)
pip install -r cpu_requirements.txt   # CPU-only
```

Key new dependency vs AGSRNet: `python-louvain==0.16` (imported as `community`)

---

## References

- Kipf & Welling, "Semi-Supervised Classification with GCNs", ICLR 2017
- Watts & Strogatz, "Collective dynamics of small-world networks", Nature 1998
- Bullmore & Sporns, "Complex brain networks", Nature Reviews Neuroscience 2009
- Newman, "Modularity and community structure in networks", PNAS 2006
- He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
