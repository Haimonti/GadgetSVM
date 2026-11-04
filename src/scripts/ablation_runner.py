"""Run the linear ablations and write one self-contained folder per ablation run.

    python src/scripts/ablation_runner.py -a topology -d adult --seeds 0 1 2 3 4
    python src/scripts/ablation_runner.py -a all -d adult                 # every ablation
    python src/scripts/ablation_runner.py -a quantization -m sdca choco -d adult --set ROUNDS=2000
    python src/scripts/ablation_runner.py --replot results/ablations/topology_run1_a9a_10-30-2026

For every (setting x model x seed) of an ablation in `ablations.py` it calls the
same `run_model()` as `main.py`, with the setting's CONFIG overrides, and writes

    results/ablations/<ablation>_run<N>_<dataset>_<mm-dd-yyyy>/
        runs/<model>__<setting>__seed<k>.json   # every single run (full metrics + config)
        comparison.json                          # per model x setting: mean ± std across seeds
        plots/curves_<metric>.png                # one panel per model, one curve per setting
        plots/bars_<stat>.png                    # grouped bars (settings x models), std bars

Centralized SDCA is the reference: it runs once per distinct LAMBDA (topology,
node count and sharding do not change the global problem), and its best final
primal is P* for that setting's suboptimality P(w) - P* and "iterations to eps".
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import CONFIG, CODE_DIR
from src.peersim_python.logger import logger
from src.scripts.ablations import ABLATIONS, LINEAR_MODELS, Skip
from src.scripts.model_picker import run_model, parse_overrides
from src.scripts.plot_analyzer import ALGO_STYLE, _ordered, _seed_band, _final, log_safe_band

ABL_ROOT = CODE_DIR / "results" / "ablations"


def _run_dir(ablation, ds):
    ABL_ROOT.mkdir(parents=True, exist_ok=True)
    nums = [int(m.group(1)) for p in ABL_ROOT.iterdir() if p.is_dir()
            if (m := re.match(rf"{ablation}_run(\d+)_", p.name))]
    return ABL_ROOT / f"{ablation}_run{max(nums, default=0) + 1}_{ds}_{datetime.now():%m-%d-%Y}"


def _overrides(spec, setting, model, extra):
    pm = setting["per_model"].get(model, {})
    if isinstance(pm, Skip):
        return pm
    return {**spec.get("base", {}), **setting["common"], **pm, **extra}


def _lam_key(ov):
    return f"lam{ov.get('LAMBDA', CONFIG['LAMBDA']):g}"


# --------------------------------------------------------------------------- #
# running
# --------------------------------------------------------------------------- #
def run_ablation(name, dataset, models, seeds, extra):
    spec = ABLATIONS[name]
    out = _run_dir(name, dataset)
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"ablation": name, "description": spec["description"], "dataset": dataset,
                "models": models, "seeds": seeds, "extra_overrides": extra,
                "settings": [], "skipped": [], "failed": [], "central": {}}
    for setting in spec["settings"]:
        manifest["settings"].append(setting["label"])
        for model in models:
            ov = _overrides(spec, setting, model, extra)
            if isinstance(ov, Skip):
                manifest["skipped"].append({"setting": setting["label"], "model": model,
                                            "reason": ov.reason})
                continue
            if model == "central_sdca":
                key = _lam_key(ov)
                manifest["central"][setting["label"]] = key
                if any(runs_dir.glob(f"central_sdca__{key}__seed*.json")):
                    continue                     # already run for this lambda
                tag = key
            else:
                tag = setting["label"]
            for seed in seeds:
                path = runs_dir / f"{model}__{tag}__seed{seed}.json"
                try:
                    run_model(model, dataset, overrides={**ov, "SEED": seed,
                                                         "RUN_TAG": f"{name}/{setting['label']}"},
                              out_path=path)
                except Exception as e:           # noqa: BLE001 — one leg must not kill the sweep
                    logger.info("ablation", f"!!! {model} {setting['label']} seed {seed} FAILED: {e}")
                    manifest["failed"].append({"setting": setting["label"], "model": model,
                                               "seed": seed, "error": str(e)})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    summarize(out)
    return out


# --------------------------------------------------------------------------- #
# comparison.json
# --------------------------------------------------------------------------- #
def _load_runs(runs_dir):
    groups = {}
    for p in sorted(runs_dir.glob("*.json")):
        model, tag, _ = p.stem.split("__")
        groups.setdefault((model, tag), []).append(json.loads(p.read_text()))
    return groups


def _stats(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n_seeds": len(vals)}


def _node_mean_curve(run, key):
    per = {}
    for m in run["metrics"]:
        for r in m:
            if r.get(key) is not None:
                per.setdefault(r["round"], []).append(r[key])
    return sorted((rd, float(np.mean(v))) for rd, v in per.items())


def _to_eps(run, p_star, eps):
    """(first round, node-mean bytes at it) where node-mean P(w) - P* <= eps."""
    key = "objective" if run["algorithm"] == "gadget" else "primal"
    bytes_at = dict(_node_mean_curve(run, "comm_bytes"))
    for rd, val in _node_mean_curve(run, key):
        if val - p_star <= eps:
            return rd, bytes_at.get(rd)
    return None, None


def summarize(out):
    manifest = json.loads((out / "manifest.json").read_text())
    groups = _load_runs(out / "runs")
    eps = manifest["extra_overrides"].get("ABLATION_EPS", CONFIG["ABLATION_EPS"])
    p_star = {tag: min([r["p_star"] for r in runs if r.get("p_star") is not None]
                       + [m[-1]["primal"] for r in runs for m in r["metrics"] if m])
              for (model, tag), runs in groups.items() if model == "central_sdca"}
    comp = {"ablation": manifest["ablation"], "dataset": manifest["dataset"], "eps": eps,
            "p_star": p_star, "skipped": manifest["skipped"], "failed": manifest["failed"],
            "pairs": ABLATIONS[manifest["ablation"]].get("pairs"), "results": {}}
    for (model, tag), runs in groups.items():
        if model == "central_sdca":
            continue
        ps = p_star.get(manifest["central"].get(tag))
        entry = {
            "final_accuracy": _stats([_final(r["metrics"], "accuracy") for r in runs]),
            "final_hinge_loss": _stats([_final(r["metrics"], "hinge_loss") for r in runs]),
            "final_primal": _stats([_final(r["metrics"], "objective" if model == "gadget" else "primal")
                                    for r in runs]),
            "final_duality_gap": _stats([_final(r["metrics"], "duality_gap") for r in runs]),
            "total_comm_bytes": _stats([r.get("total_comm_bytes") for r in runs]),
            "rounds": _stats([max((m[-1]["round"] for m in r["metrics"] if m), default=None)
                              for r in runs]),
            "wall_time": _stats([_final(r["metrics"], "wall_time", np.max) for r in runs]),
            "stop_reasons": {k: sum(r.get("stop_reason") == k for r in runs)
                             for k in ("converged", "plateau", "budget")},
        }
        if ps is not None:
            entry["p_star"] = ps
            entry["final_suboptimality"] = _stats([
                (_final(r["metrics"], "objective" if model == "gadget" else "primal") or np.nan) - ps
                for r in runs])
            hits = [_to_eps(r, ps, eps) for r in runs]
            entry["iterations_to_eps"] = _stats([h[0] for h in hits])
            entry["bytes_to_eps"] = _stats([h[1] for h in hits])
            entry["reached_eps"] = f"{sum(h[0] is not None for h in hits)}/{len(hits)}"
        comp["results"].setdefault(model, {})[tag] = entry
    (out / "comparison.json").write_text(json.dumps(comp, indent=2))
    _plots(out, manifest, groups, comp)
    logger.info("ablation", f"comparison -> {out / 'comparison.json'}")


# --------------------------------------------------------------------------- #
# plots
# --------------------------------------------------------------------------- #
CURVES = [("suboptimality", "P(w) - P*", True), ("accuracy", "Test accuracy", False),
          ("comm_bytes", "Cumulative bytes / node", False), ("duality_gap", "Duality gap", True)]
BARS = ["final_accuracy", "final_suboptimality", "iterations_to_eps", "bytes_to_eps",
        "total_comm_bytes", "rounds"]


def _plots(out, manifest, groups, comp):
    plots = out / "plots"
    plots.mkdir(exist_ok=True)
    settings = manifest["settings"]
    models = _ordered({m for (m, _) in groups if m != "central_sdca"})
    if not models:
        return
    cmap = plt.get_cmap("viridis", max(len(settings), 2))
    for key, ylabel, log_y in CURVES:
        fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.2), squeeze=False)
        for ax, model in zip(axes[0], models):
            for i, tag in enumerate(settings):
                runs = groups.get((model, tag))
                if not runs:
                    continue
                ps = comp["results"][model][tag].get("p_star")
                if key == "suboptimality":
                    if ps is None:
                        continue
                    y_key = "objective" if model == "gadget" else "primal"
                    xs, mu, sd = _seed_band({"runs": runs}, y_key, "round",
                                            transform=lambda v, p=ps: np.maximum(v - p, 1e-16))
                else:
                    xs, mu, sd = _seed_band({"runs": runs}, key, "round")
                if len(xs):
                    mu, lo, hi = log_safe_band(mu, sd) if log_y else (mu, mu - sd, mu + sd)
                    ax.plot(xs, mu, color=cmap(i), label=tag)
                    ax.fill_between(xs, lo, hi, color=cmap(i), alpha=0.15, linewidth=0)
            ax.set(title=ALGO_STYLE[model][0], xlabel="Iteration (cycle, log scale)", ylabel=ylabel)
            ax.set_xscale("log")
            ax.grid(True, which="both", alpha=0.25)
            if log_y:
                ax.set_yscale("log")
            ax.legend(fontsize=7)
        fig.suptitle(f"{manifest['ablation']} ablation — {ylabel} (mean ± std over seeds) "
                     f"[{manifest['dataset']}]", fontsize=12)
        plt.tight_layout(rect=(0, 0, 1, 0.93))
        plt.savefig(plots / f"curves_{key}.png", dpi=150, bbox_inches="tight")
        plt.close()

    width = 0.8 / len(models)
    for stat in BARS:
        fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(settings)), 4.5))
        drawn = False
        for j, model in enumerate(models):
            mu, sd, xs = [], [], []
            for i, tag in enumerate(settings):
                st = comp["results"].get(model, {}).get(tag, {}).get(stat)
                if st:
                    xs.append(i + j * width)
                    mu.append(st["mean"])
                    sd.append(st["std"])
            if xs:
                drawn = True
                ax.bar(xs, mu, width, yerr=sd, capsize=3, color=ALGO_STYLE[model][2],
                       label=ALGO_STYLE[model][0])
        if not drawn:
            plt.close()
            continue
        ax.set_xticks([i + 0.4 - width / 2 for i in range(len(settings))])
        ax.set_xticklabels(settings, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{manifest['ablation']}: {stat} (mean ± std over seeds)", fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        if stat in ("total_comm_bytes", "bytes_to_eps", "final_suboptimality"):
            ax.set_yscale("log")
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(plots / f"bars_{stat}.png", dpi=150, bbox_inches="tight")
        plt.close()


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Run linear ablations (see src/scripts/ablations.py).")
    ap.add_argument("-a", "--ablations", nargs="+", default=None,
                    help=f"{' '.join(ABLATIONS)} or 'all'")
    ap.add_argument("-m", "--models", nargs="+", default=LINEAR_MODELS,
                    help=f"subset of {' '.join(LINEAR_MODELS)} (default: all linear)")
    ap.add_argument("-d", "--datasets", nargs="+", default=["adult"], help="dataset keywords (default: adult = a9a)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--set", nargs="+", default=[], dest="overrides", metavar="KEY=VAL",
                    help="extra CONFIG overrides applied to every run (e.g. ROUNDS=2000)")
    ap.add_argument("--replot", default=None, metavar="RUN_DIR",
                    help="rebuild comparison.json + plots of an existing ablation folder")
    args = ap.parse_args()
    if args.replot:
        summarize(Path(args.replot))
        return
    if not args.ablations:
        raise SystemExit("choose -a <ablation ...> or --replot <dir>")
    names = list(ABLATIONS) if "all" in args.ablations else args.ablations
    bad = [a for a in names if a not in ABLATIONS] + [m for m in args.models if m not in LINEAR_MODELS]
    if bad:
        raise SystemExit(f"unknown ablation/model: {bad}")
    extra = parse_overrides(args.overrides)
    for ds in args.datasets:
        for name in names:
            out = run_ablation(name, ds, args.models, args.seeds, extra)
            print(f"{name} [{ds}] -> {out}")


if __name__ == "__main__":
    main()
