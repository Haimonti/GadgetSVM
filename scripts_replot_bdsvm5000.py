"""Re-render from the saved trace. The bound now rides in the legend instead of
as floating text — at C=30 the curve sits on the bound, so there was nowhere to
put a label that did not land on either the line or the peak box."""
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

z = np.load("results/plots/bdsvm_5000_raw.npz")
runs = {1.0: z["c1"], 30.0: z["c30"]}

fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8), sharey=True)
for ax, C in zip(axes, (1.0, 30.0)):
    T = runs[C]; ep, wls, tr, te = T[:,0], T[:,1], T[:,2], T[:,3]
    ax2 = ax.twinx()
    ax2.plot(ep, wls, color="#B4762A", lw=1.2, alpha=.8, zorder=1,
             label="$L_{WLS}$ (right axis)")
    ax2.set_ylabel("$L_{WLS}$  (IRWLS objective)", color="#B4762A", fontsize=9)
    ax2.tick_params(axis="y", labelcolor="#B4762A", labelsize=8)
    ax.plot(ep, tr, color="#8A929C", lw=1.3, label="train accuracy", zorder=3)
    ax.plot(ep, te, color="#17595E", lw=2.1, label="test accuracy", zorder=4)
    ax.axhline(0.7577, color="#2E6B46", ls=":", lw=1.4, zorder=2,
               label="centralized bound 0.7577")
    i = int(np.argmax(te))
    ax.plot(ep[i], te[i], "o", ms=5.5, color="#A33A2C", zorder=6)
    ax.text(1.35, 0.600,
            f"peak  {te[i]:.4f}  @ ep {int(ep[i])}\nend    {te[-1]:.4f}"
            f"\ndrop   {te[i]-te[-1]:+.4f}",
            fontsize=9, color="#A33A2C", va="top", linespacing=1.55,
            bbox=dict(fc="white", ec="#A33A2C", lw=.8, alpha=.95, pad=5))
    ax.set_zorder(ax2.get_zorder()+1); ax.patch.set_visible(False)
    ax.set_xscale("log"); ax.set_xlim(1, 5000)
    ax.set_xlabel("Epoch (log scale)")
    ax.set_title(f"C = {C:g}" + ("   — previous default, underfits"
                 if C == 1 else "   — tuned on validation AUC"), fontsize=11.5)
    ax.grid(True, alpha=.22, zorder=0)
axes[0].set_ylabel("Accuracy"); axes[0].set_ylim(0.55, 0.79)
# Legend from the left panel's own artists — calling twinx() again here would
# have created a third axis, which is what put a stray 0-1 scale on panel one.
h1, l1 = axes[0].get_legend_handles_labels()
axes[1].legend(h1, l1, fontsize=8.5, loc="lower right", framealpha=.95)
fig.suptitle("BDSVM IRWLS, 5000 epochs — the decline was underfitting, not divergence",
             fontsize=13)
fig.tight_layout(rect=[0,0,1,.955])
fig.savefig("results/plots/bdsvm_5000_epochs.png", dpi=150)
print("  ok")
