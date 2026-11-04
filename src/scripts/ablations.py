"""Linear-setup ablation specs (10/26 meeting list) — data only, no running.

Each ablation is a list of *settings*. A setting has a `label`, `common`
CONFIG overrides applied to every model, and optional `per_model` entries that
either add model-specific overrides (a dict) or skip the model (`Skip(reason)`,
recorded in comparison.json). `base` overrides apply to every setting.

    ABLATIONS["topology"]["settings"][0] ->
        {"label": "ring", "common": {"TOPOLOGY": "ring"}, "per_model": {}}

`ablation_runner.py` executes them; edit values here, not in the runner.
"""

LINEAR_MODELS = ["central_sdca", "sdca", "gadget", "cola", "choco"]


class Skip:
    def __init__(self, reason):
        self.reason = reason


def _s(label, common=None, **per_model):
    return {"label": label, "common": common or {}, "per_model": per_model}


_NO_GAP = Skip("no duality gap: primal-only method; its native criterion is shown in 'native'")
_SDCA_ONLY = Skip("stopping rule is specific to our SDCA's per-node latch")

# CHOCO's consensus step must be tuned per compressor; these are the values the
# official experiments use on epsilon (experiment_epsilon_final.py) — qsgd-8 0.34,
# top-k 0.04, full 1.0. qsgd 4-bit is not in their runs; 0.34 is a starting point.
_CHOCO_GAMMA = {"full": 1.0, "lossless_sparse": 1.0, "qsgd:127": 0.34, "qsgd:7": 0.34,
                "top_k:0.1": 0.04}


def _quant(spec):
    return _s(spec.replace(":", "_"), {"COMPRESSOR": spec},
              choco={"CHOCO_COMPRESSOR": spec, "CHOCO_CONSENSUS_LR": _CHOCO_GAMMA[spec]},
              central_sdca=Skip("no communication"))


ABLATIONS = {
    # duality-gap / threshold criterion, and local vs global stopping rule
    "stopping": {
        "description": "Gap threshold and SDCA's per-node local vs global stopping rule.",
        "base": {"STOP_ON_THRESHOLD": True},
        "settings": [
            _s("native"),
            *[_s(f"thr_{t:g}", {"GAP_THRESHOLD": t}, gadget=_NO_GAP, choco=_NO_GAP,
                 central_sdca=Skip("reference only (runs to CENTRAL_GAP_TOL)"))
              for t in (1e-2, 1e-3, 1e-4, 1e-5)],
            *[_s(f"rule_{r}", {"STOP_RULE": r, "GAP_THRESHOLD": 1e-4}, gadget=_SDCA_ONLY,
                 choco=_SDCA_ONLY, cola=_SDCA_ONLY,
                 central_sdca=Skip("reference only (runs to CENTRAL_GAP_TOL)"))
              for r in ("node_global_gap", "node_local_gap", "mean_gap")],
        ],
    },
    # step size: lambda (all models) and each model's own step parameter
    "stepsize": {
        "description": "Regularisation lambda and each model's native step size vs iterations.",
        "base": {},
        "settings": [
            *[_s(f"lambda_{lam:g}", {"LAMBDA": lam}) for lam in (1e-3, 1e-4, 1e-5)],
            *[_s(f"sdca_scale_{lbl}", {"SDCA_STEP_SCALE": v},
                 **{m: Skip("SDCA-only step parameter") for m in ("gadget", "cola", "choco", "central_sdca")})
              for lbl, v in (("1", 1.0), ("0.5", 0.5), ("1overK", None))],
            *[_s(f"cola_gamma_{g:g}", {"COLA_GAMMA": g},
                 **{m: Skip("COLA-only step parameter") for m in ("sdca", "gadget", "choco", "central_sdca")})
              for g in (1.0, 0.5)],
            *[_s(f"choco_lr_{lr:g}", {"CHOCO_INITIAL_LR": lr},
                 **{m: Skip("CHOCO-only step parameter") for m in ("sdca", "gadget", "cola", "central_sdca")})
              for lr in (0.01, 0.1, 1.0)],
        ],
    },
    # coordinate ascent vs SGD, as two matched pairs:
    #   SDCA (ours) vs GADGET  — same random-k gossip, same coordinate/SGD steps per cycle
    #   COLA vs CHOCO-SGD      — same synchronous W-mixing; dual CD vs SGD
    # across the amount of local work per communication round
    "solver": {
        "description": "Coordinate ascent vs SGD: (SDCA vs GADGET) and (COLA vs CHOCO) "
                       "at low / mid / high local work per round.",
        "pairs": [["sdca", "gadget"], ["cola", "choco"]],
        "base": {},
        "settings": [
            _s(lbl, {"SDCA_LOCAL_STEPS": steps, "COLA_LOCAL_ITERS": passes},
               choco={"CHOCO_BATCH": batch})
            for lbl, steps, passes, batch in (("work_low", 1024, 0.25, 1),
                                               ("work_mid", 4096, 1.0, 8),
                                               ("work_high", 16384, 4.0, 32))
        ],
    },
    "topology": {
        "description": "Graph topology at K = 16 nodes (perfect square for the grid, as in COLA).",
        "base": {"NUM_WORKERS": 16},
        "settings": [
            _s("ring", {"TOPOLOGY": "ring"}),
            _s("grid", {"TOPOLOGY": "grid"}),
            _s("cycle_c2", {"TOPOLOGY": "connected_cycle", "CYCLE_CONNECTIVITY": 2}),
            _s("cycle_c3", {"TOPOLOGY": "connected_cycle", "CYCLE_CONNECTIVITY": 3}),
            _s("complete", {"TOPOLOGY": "complete"}),
            _s("random_kout", {"TOPOLOGY": "random_kout"}),
        ],
    },
    "nodes": {
        "description": "Number of nodes 10 -> 100 on the random k-out graph.",
        "base": {"TOPOLOGY": "random_kout"},
        "settings": [_s(f"K{k}", {"NUM_WORKERS": k}) for k in (10, 25, 50, 100)],
    },
    "noniid": {
        "description": "Data distribution across nodes: IID vs label-skewed.",
        "base": {},
        "settings": [
            _s("iid", {"SHARDING": "iid"}),
            _s("dirichlet_1", {"SHARDING": "dirichlet", "DIRICHLET_ALPHA": 1.0}),
            _s("dirichlet_0.1", {"SHARDING": "dirichlet", "DIRICHLET_ALPHA": 0.1}),
            _s("label_sorted", {"SHARDING": "label_sorted"}),
        ],
    },
    "quantization": {
        "description": "Message compression: lossless (full, sparse) vs lossy (QSGD 8/4-bit, top-10%).",
        "base": {},
        "settings": [_quant(spec) for spec in _CHOCO_GAMMA],
    },
}
