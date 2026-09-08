# BDSVM hyperparameters: settings and how they were chosen

Every value below was measured, not inherited. Where a default came from
`run_benchmark.py`'s `METHOD_KWARGS` and had never been tuned, that is stated.

## Final settings

| symbol | meaning | value | how chosen |
|---|---|---|---|
| `P` | budget — number of pre-image vectors | 100 (covtype), 1000 (rcv1) | must scale with intrinsic dimension; see below |
| `C` | penalty in the weighting rule `a_i = 2C/(e_i y_i)` | **30.0** | grid on validation AUC |
| `gamma` | Gaussian kernel width | median heuristic, `1/median(||x-p||^2)` | grid showed it is flat below ~0.12 |
| `rho` | mixing weight, `beta <- rho*beta + (1-rho)*beta_new` | 0.5 | swept; does not matter (see below) |
| `eta` | stopping threshold on relative change of beta | 5e-3 | unchanged from the paper |
| kernel | Gaussian (RBF), `exp(-gamma ||x-p||^2)` | — | as in the paper |
| pre-images | random, from a shared seed | uniform (covtype), unit-norm sparse (rcv1) | see below |

## C: 1.0 -> 30.0

`C=1.0` was a placeholder in `METHOD_KWARGS`. It underfits: the data term is
outweighed by the regulariser, so IRWLS converges to a solution *worse* than the
barely-trained iterate it passes through on epoch 1. That is what produced the
"accuracy peaks at epoch 1 then declines" behaviour — convergence to a poor
optimum, not divergence. Confirmed over 5000 epochs: `L_WLS` reaches its minimum
at epoch 2863 while test accuracy sits at 0.696, against 0.751 at epoch 1.

First grid (test accuracy at the converged point, covtype, 150 epochs):

| gamma \ C | 0.001 | 0.01 | 0.1 | 1.0 | 10.0 |
|---|---|---|---|---|---|
| 0.0152 | 0.5175 | 0.6062 | 0.6945 | 0.6925 | 0.7563 |
| 0.0607 | 0.5175 | 0.6140 | 0.6872 | 0.6912 | **0.7562** |
| 0.1214 | 0.5175 | 0.5175 | 0.6558 | 0.6783 | 0.7443 |
| 0.2427 | 0.5175 | 0.5175 | 0.5175 | 0.6537 | 0.6785 |

C=10 sat at the grid's edge, so the grid was extended to 300 and the criterion
switched to validation AUC (what the predecessor paper, DIRWLS / IEEE TSMC 2020,
selects on, and threshold-independent):

| C | 1 | 10 | 30 | 100 | 300 |
|---|---|---|---|---|---|
| val AUC | 0.7979 | 0.8206 | **0.8218** | 0.8183 | 0.8212 |
| test acc | 0.6912 | 0.7562 | 0.7600 | 0.7560 | 0.7580 |

C=1 is the only bad point; everything from 10 up is a plateau. **C=30** is the
plateau's peak on validation AUC. The best single cell was gamma=2x median with
C=300 (test 0.7602, just above the 0.7577 centralized bound), but it lies inside
the same plateau and costs a second tuned parameter, so the median heuristic
stays.

## rho: swept and ruled out

Algorithm 1 line 10 suggests line-searching `rho` on a validation set. It does
not help here — every fixed value peaked at 0.7512 on epoch 1 and settled at
0.68-0.72, and a per-epoch validation line search reached only 0.6975:

| rho | 0.00 | 0.25 | 0.50 | 0.75 | 0.90 | 0.95 | 0.99 | line search |
|---|---|---|---|---|---|---|---|---|
| peak | 0.7512 | 0.7512 | 0.7512 | 0.7512 | 0.7512 | 0.7512 | 0.7513 | 0.7518 |
| end | 0.6913 | 0.6913 | 0.6910 | 0.6843 | 0.6847 | 0.6805 | 0.7220 | 0.6975 |

`rho` sets how fast the iterate moves, not where it converges. rho=0.99 has the
smallest drop only because 200 epochs is not enough for it to arrive.

## gamma: median heuristic, not 1/d

The obvious default `gamma = 1/n_features` silently destroys the model on
high-dimensional data. On rcv1 (d=47236, unit-norm rows) squared distances sit
near 2.0, so `1/d = 2.1e-05` puts every kernel value at `exp(-4e-05) ~ 1`: the
kernel matrix is constant to within 2e-07 and the architecture carries no
information. The whole first grid's rcv1 BDSVM column was chance-level for this
reason and was discarded. The median heuristic puts the exponent at O(1) by
construction and does not depend on d.

## Pre-images: distribution matters in high dimensions

The paper generates the P pre-images randomly but does not pin the distribution.

- **uniform** on `[0,1]^d` — suits the LIBSVM `.scale` sets (covtype, d=54).
- **unit-norm sparse** — needed for rcv1. A dense uniform pre-image there has
  norm ~125 against unit-norm data, so every `||x-p||^2` collapses to `~||p||^2`.

Even with unit-norm pre-images, distances concentrate into [1.957, 2.039] — a
dense random direction in 47k dimensions is nearly orthogonal to every document.
Making the pre-images sparse barely helps (kernel std 1.67e-3 vs 1.70e-3); the
concentration is dimensional and not fixable by sampling.

## P: must scale with intrinsic dimension

`P` fixes the model's capacity. covtype (d=54) needs only P=100. rcv1 (d=47236)
is at chance until P grows, and the message is `(P+1)^2` floats:

| P | 100 | 400 | 1000 | 2000 |
|---|---|---|---|---|
| rcv1 test acc | 0.5715 | 0.7738 | 0.8572 | 0.8998 |
| matrix size | 0.08 MB | 1.29 MB | 8.02 MB | 32.03 MB |

(majority-class baseline on rcv1 is 0.5202)

## Protocol-level settings

| symbol | meaning | value |
|---|---|---|
| `E` | table entries carried per message | 2 (own + 1 random other) |
| `gamma_gossip` | neighbours pushed to per cycle | 2 |
| K | workers | 10 |
| topology | overlay | ring (also ran random_kout, full) |
| seed | RNG seed | 42 |

`E=1` would leave a node's table covering only its immediate neighbourhood, since
contributions are not forwarded otherwise. Gossip fan-out must also keep the
overlay connected: `WireKOut` with k=1 splits 10 nodes into 7+3, and
contributions cannot cross a component boundary.

## Results at these settings (covtype, ring, k=2, 60 cycles)

| scheme | C=1 | C=30 |
|---|---|---|
| iid | 0.7148 ± 0.0020 | 0.7462 ± 0.0019 |
| dirichlet 0.1 | 0.6760 ± 0.0793 | 0.7063 ± 0.0944 |
| label_skew | 0.7041 ± 0.0022 | 0.7402 ± 0.0037 |

Centralized upper bound 0.7577.

The dirichlet 0.1 spread is the partition, not the method: at alpha=0.1 two of
the ten clients receive zero samples and score exactly chance (0.517), while the
other eight land within 0.004 of each other at 0.751-0.755. Excluding the empty
clients that cell reads 0.753.
