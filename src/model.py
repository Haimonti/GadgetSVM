import time
import numpy as np
import torch
import torch.nn as nn
import lightning as pl

from p2pfl.management.logger import logger




class _Shared:
    """Wrapper that survives deepcopy by returning the same object.

    p2pfl calls copy.deepcopy(lm) after each aggregation round.
    SDCA state (alpha, averages, metrics) must persist across those copies,
    so we hold all mutable state here; __deepcopy__ always returns self.
    """
    __slots__ = ("alpha", "w_avg", "avg_cnt", "step", "metrics", "start", "comm_bytes")

    def __init__(self, n, d):
        self.alpha      = torch.zeros(n, dtype=torch.float32)
        self.w_avg      = torch.zeros(d, dtype=torch.float32)
        self.avg_cnt    = 0
        self.step       = 0
        self.metrics: list = []
        self.start      = time.time()
        self.comm_bytes = 0

    def __deepcopy__(self, memo):
        return self  # intentionally shared — all copies see the same state

class LinearSVM(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.weight = nn.Parameter(
            torch.zeros(n_features, dtype=torch.float32),
            requires_grad=False,
        )

    def forward(self, x):
        return x @ self.weight


class SVMSDCALightning(pl.LightningModule):
    def __init__(self, X_csr, y_np, lambda_reg, t0_fraction=0.5, _shared=None):
        super().__init__()
        self.automatic_optimization = False
        n, d = X_csr.shape
        self.n = n
        self.d = d
        self.lambda_reg = lambda_reg
        self._t0    = max(1, int(t0_fraction * n))
        self.model  = LinearSVM(d)
        self._X_csr = X_csr
        self._y_np  = y_np
        self._s     = _shared if _shared is not None else _Shared(n, d)

    @property
    def _metrics(self):
        return self._s.metrics

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)

    def on_fit_start(self):
        self.cpu()

    def on_test_start(self):
        self.cpu()

    def training_step(self, batch, batch_idx):
        s        = self._s
        idxs     = batch["idx"].view(-1)
        ys       = batch["y"].view(-1).float()
        w        = self.model.weight.data.cpu()
        loss_sum = 0.0
        t_start  = time.time()

        for k in range(len(idxs)):
            local_i    = int(idxs[k].item())
            yi         = float(ys[k].item())
            xi_np      = self._X_csr[local_i].toarray().ravel().astype(np.float32)
            xi         = torch.from_numpy(xi_np)
            xi_norm_sq = float(xi.dot(xi).item())
            if xi_norm_sq < 1e-12:
                continue
            alpha_i      = float(s.alpha[local_i].item())
            score        = float(xi.dot(w).item()) * yi
            denom        = xi_norm_sq / (self.lambda_reg * self.n)
            new_alpha_yi = max(0.0, min(1.0, (1.0 - score) / denom + alpha_i * yi))
            delta        = yi * new_alpha_yi - alpha_i
            s.alpha[local_i] += delta
            w.add_(xi, alpha=delta / (self.lambda_reg * self.n))
            s.step += 1
            if s.step >= self._t0:
                s.w_avg.add_(w)
                s.avg_cnt += 1
            loss_sum += max(0.0, 1.0 - yi * float(xi.dot(w).item()))

        self.model.weight.data.copy_(w)
        avg_loss  = loss_sum / max(1, len(idxs))
        self.log("train_loss", avg_loss, prog_bar=True, on_step=False, on_epoch=True)
        return torch.tensor(avg_loss).detach()

    def validation_step(self, batch, batch_idx):
        return torch.tensor(0.0)

    def test_step(self, batch, batch_idx):
        idxs    = batch["idx"].view(-1).cpu().numpy()
        ys      = batch["y"].view(-1).float().cpu().numpy()
        w_np    = self.model.weight.data.cpu().numpy()
        X_batch = self._X_csr[idxs].toarray().astype(np.float32)
        scores  = X_batch @ w_np
        hinge   = float(np.mean(np.maximum(0.0, 1.0 - ys * scores)))
        self.log("test_hinge_loss", hinge, on_step=False, on_epoch=True)
        return torch.tensor(hinge).detach()

    def on_train_epoch_end(self):
        self._compute_metrics()

    def _compute_metrics(self):
        s = self._s
        if s.avg_cnt > 0:
            w_np = (s.w_avg / s.avg_cnt).cpu().numpy()
        else:
            w_np = self.model.weight.data.cpu().numpy()

        alpha_np  = s.alpha.cpu().numpy()
        scores    = self._X_csr.dot(w_np)
        margins   = 1.0 - self._y_np * scores
        hinge     = float(np.mean(np.maximum(0.0, margins)))
        reg       = float((self.lambda_reg / 2.0) * np.dot(w_np, w_np))
        primal    = hinge + reg
        w_alpha   = self._X_csr.T.dot(alpha_np) / (self.lambda_reg * self.n)
        dual_data = float(np.mean(alpha_np * self._y_np))
        dual_reg  = float((self.lambda_reg / 2.0) * np.dot(w_alpha, w_alpha))
        dual      = dual_data - dual_reg
        gap       = primal - dual
        wall      = time.time() - s.start
        rnd       = len(s.metrics) + 1

        # cumulative bytes: one weight vector (float32) sent per round
        s.comm_bytes += self.d * 4

        logger.info(
            "SDCA",
            f"round={rnd}  gap={gap:.6f}  primal={primal:.6f}  "
            f"dual={dual:.6f}  hinge={hinge:.4f}  weight_norm={float(np.linalg.norm(w_np)):.4f}  "
            f"comm_bytes={s.comm_bytes}  t={wall:.1f}s"
        )

        for name, val in [("primal", primal), ("dual", dual),
                           ("duality_gap", gap), ("hinge_loss", hinge), ("wall_time", wall)]:
            self.log(name, float(val), on_step=False, on_epoch=True)

        s.metrics.append({
            "round":       rnd,
            "primal":      primal,
            "dual":        dual,
            "duality_gap": gap,
            "hinge_loss":  hinge,
            "wall_time":   wall,
            "comm_bytes":  s.comm_bytes,
        })
