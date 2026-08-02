"""Week 3: two-panel hero figure — theta trajectories, baseline vs. HOCBF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = "results/trajectories.npz"
OUT = "docs/hero.png"

def panel(ax, trajs, theta_max, title):
    for traj in trajs:
        t = np.arange(len(traj))
        ax.plot(t, traj, color="#4C72B0", alpha=0.5, linewidth=1.0)
        unsafe = np.abs(traj) > theta_max
        ax.plot(t[unsafe], traj[unsafe], "o", color="#C44E52",
                markersize=2.5, zorder=3)
    ax.axhspan(-theta_max, theta_max, color="#55A868", alpha=0.12, zorder=0)
    ax.axhline(theta_max, color="#55A868", linestyle="--", linewidth=1.2)
    ax.axhline(-theta_max, color="#55A868", linestyle="--", linewidth=1.2)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("timestep")
    ax.set_ylim(-0.30, 0.30)

def main():
    d = np.load(DATA)
    theta_max = float(d["theta_max"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panel(axes[0], d["baseline"], theta_max, "PPO (unfiltered)")
    panel(axes[1], d["hocbf"], theta_max, "PPO + HOCBF safety filter")
    axes[0].set_ylabel(r"pole angle $\theta$ (rad)")
    fig.suptitle("Angle constraint violations: 117 → 10 (91.5% reduction), "
                 "return 108.9 → 106.8", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()