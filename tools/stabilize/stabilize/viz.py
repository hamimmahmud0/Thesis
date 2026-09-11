"""Trajectory and camera-motion visualisation.

Generates a multi-panel diagnostic plot from motion estimation results.
Can also work directly from a ``motion.csv`` file for backward
compatibility.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from .utils import runs_of


def _make_plots(df, out_png: str | Path, title: str) -> None:
    """Render a 3x3 diagnostic figure from a DataFrame.

    Expected columns: frame, dx, dy, d_yaw_deg, scale, inliers,
    inlier_ratio, usable, reliable, cum_x, cum_y, cum_yaw_deg,
    cum_log_scale.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = df["frame"].to_numpy()
    unreliable = ~df["reliable"].to_numpy()

    fig = plt.figure(figsize=(20, 10.5))
    gs = fig.add_gridspec(3, 3, width_ratios=[1.25, 1, 1])
    fig.suptitle(title, fontsize=13)

    def shade(ax):
        for a, b in runs_of(unreliable):
            ax.axvspan(t[a], t[min(b, len(t) - 1)], color="red", alpha=0.18)

    # 1. Trajectory X-Y coloured by time
    ax1 = fig.add_subplot(gs[:2, 0])
    sc = ax1.scatter(df["cum_x"], df["cum_y"], c=t, cmap="viridis", s=4)
    fig.colorbar(sc, ax=ax1, label="frame")
    ax1.plot(
        df["cum_x"].iloc[0], df["cum_y"].iloc[0],
        marker="*", ms=18, color="lime", ls="none", label="start",
    )
    ax1.plot(
        df["cum_x"].iloc[-1], df["cum_y"].iloc[-1],
        marker="X", ms=14, color="red", ls="none", label="end",
    )
    ax1.set_aspect("equal", adjustable="datalim")
    ax1.set_xlabel("cumulative X (px)")
    ax1.set_ylabel("cumulative Y (px)")
    ax1.set_title("Camera trajectory (frame-0 reference)")
    ax1.legend(loc="best")
    ax1.grid(alpha=0.3)

    # 2. Per-frame translation
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t, df["dx"], lw=0.7, label="dx")
    ax2.plot(t, df["dy"], lw=0.7, label="dy")
    shade(ax2)
    ax2.set_xlabel("frame")
    ax2.set_ylabel("px/frame")
    ax2.set_title("Per-frame translation (camera)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    # 3. Yaw
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.plot(t, df["d_yaw_deg"], lw=0.7, color="tab:blue", label="yaw rate")
    shade(ax3)
    ax3.set_xlabel("frame")
    ax3.set_ylabel("deg/frame")
    ax3b = ax3.twinx()
    ax3b.plot(t, df["cum_yaw_deg"], lw=1.2, color="tab:orange", label="cumulative")
    ax3b.set_ylabel("deg (cum)")
    l1, la1 = ax3.get_legend_handles_labels()
    l2, la2 = ax3b.get_legend_handles_labels()
    ax3.legend(l1 + l2, la1 + la2, fontsize=8, loc="upper left")
    ax3.set_title("Yaw")
    ax3.grid(alpha=0.3)

    # 4. Scale (log y)
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.semilogy(t, df["scale"].clip(lower=1e-6), lw=0.7)
    ax4.axhline(1.0, color="k", lw=0.8, ls="--")
    shade(ax4)
    ax4.set_xlabel("frame")
    ax4.set_ylabel("image scale (>1 descending)")
    ax4.set_title("Zoom scale per frame")
    ax4.grid(alpha=0.3, which="both")

    # 5. Diagnostics
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.plot(t, df["usable"], lw=0.7, color="tab:green", label="usable pts")
    shade(ax5)
    ax5.set_xlabel("frame")
    ax5.set_ylabel("# usable")
    ax5b = ax5.twinx()
    ax5b.plot(t, df["inlier_ratio"], lw=0.7, color="tab:red", label="inlier ratio")
    ax5b.set_ylim(-0.02, 1.05)
    ax5b.set_ylabel("inlier ratio")
    l1, la1 = ax5.get_legend_handles_labels()
    l2, la2 = ax5b.get_legend_handles_labels()
    ax5.legend(l1 + l2, la1 + la2, fontsize=8, loc="lower left")
    ax5.set_title("Diagnostics")
    ax5.grid(alpha=0.3)

    # 6. Summary strip
    ax6 = fig.add_subplot(gs[2, :])
    ax6.axis("off")
    m = df[df["reliable"]]
    step = np.hypot(m["dx"], m["dy"])
    txt = (
        f"path length: {step.sum():,.0f} px   |   "
        f"mean speed: {step.mean():.3f} px/f   median: {step.median():.3f} px/f   |   "
        f"total yaw: {df['cum_yaw_deg'].iloc[-1]:+.2f} deg   |   "
        f"net scale: x{np.exp(df['cum_log_scale'].iloc[-1]):.4f}   |   "
        f"unreliable: {(~df['reliable']).mean() * 100:.2f}%"
    )
    ax6.text(
        0.5, 0.5, txt, ha="center", va="center", fontsize=12,
        family="monospace",
        bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#999999"),
    )

    fig.subplots_adjust(hspace=0.45, wspace=0.35)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def viz_from_npz(
    motion_npz: str | Path,
    out_png: str | Path = "trajectory.png",
    video_name: str = "",
) -> None:
    """Generate trajectory visualisation from a motion.npz file."""
    import pandas as pd

    from .motion import load_motion

    m = load_motion(motion_npz)

    # Reconstruct cumulative decomposition for the plot.
    # cum_x/cum_y use the CAMERA convention (negated cumulative IMAGE
    # translation), consistent with the per-frame dx/dy camera columns.
    T = len(m["reliable"])
    cum_x = -m["cumulative"][:, 0, 2]
    cum_y = -m["cumulative"][:, 1, 2]
    cum_yaw = np.zeros(T)
    cum_log_scale = np.zeros(T)
    cyaw = 0.0
    clogs = 0.0
    for t in range(T):
        if m["reliable"][t]:
            cyaw += m["d_yaw_deg"][t]
            clogs += float(np.log(max(m["scale"][t], 1e-9)))
        cum_yaw[t] = cyaw
        cum_log_scale[t] = clogs

    df = pd.DataFrame(dict(
        frame=np.arange(T),
        dx=m["dx"],
        dy=m["dy"],
        d_yaw_deg=m["d_yaw_deg"],
        scale=m["scale"],
        inliers=m["inliers"],
        inlier_ratio=m["inlier_ratio"],
        usable=m["usable"],
        reliable=m["reliable"],
        cum_x=cum_x,
        cum_y=cum_y,
        cum_yaw_deg=cum_yaw,
        cum_log_scale=cum_log_scale,
    ))

    name = video_name or Path(motion_npz).stem
    title = f"Camera motion ({name})"
    _make_plots(df, out_png, title)

    # Print summary.
    rel = np.where(m["reliable"])[0]
    if len(rel) >= 2:
        step = np.hypot(m["dx"][rel], m["dy"][rel])
        print(f"Path length : {step.sum():,.0f} px")
        print(f"Mean speed  : {step.mean():.3f} px/frame")
        print(f"Total yaw   : {cum_yaw[-1]:+.2f} deg")
        print(f"Net scale   : x{np.exp(cum_log_scale[-1]):.4f}")
    print(f"Wrote: {out_png}")


def viz_from_csv(
    motion_csv: str | Path,
    out_png: str | Path = "trajectory.png",
    video_name: str = "",
    fps: float | None = None,
    gsd: float | None = None,
) -> None:
    """Generate trajectory visualisation from a motion.csv (legacy)."""
    import pandas as pd

    df = pd.read_csv(motion_csv)

    # Reconstruct missing columns.
    if "cum_log_scale" not in df.columns:
        reliable = df["reliable"].astype(bool).to_numpy()
        logs = np.log(df["scale"].clip(lower=1e-9)).to_numpy()
        logs[~reliable] = 0.0
        df["cum_log_scale"] = np.cumsum(logs)
    if "usable" not in df.columns:
        ratio = df["inlier_ratio"].to_numpy()
        df["usable"] = np.where(
            ratio > 0, (df["inliers"] / np.maximum(ratio, 1e-9)), 0
        )
    if "cum_yaw_deg" not in df.columns:
        df["cum_yaw_deg"] = np.cumsum(np.where(
            df["reliable"].astype(bool), df["d_yaw_deg"], 0.0
        ))

    name = video_name or Path(motion_csv).stem
    fps_str = f" @ {fps:.3f} fps" if fps else ""
    title = f"Camera motion ({name}){fps_str}"
    _make_plots(df, out_png, title)

    m = df[df["reliable"]]
    step = np.hypot(m["dx"], m["dy"])
    print(f"Path length : {step.sum():,.0f} px", end="")
    if gsd:
        print(f"  = {step.sum() * gsd:,.1f} m", end="")
    print()
    if fps:
        print(f"Mean speed  : {step.mean():.3f} px/frame", end="")
        if gsd:
            print(f"  = {step.mean() * gsd * fps:.3f} m/s", end="")
        print()
    print(f"Total yaw   : {df['cum_yaw_deg'].iloc[-1]:+.2f} deg")
    print(f"Net scale   : x{np.exp(df['cum_log_scale'].iloc[-1]):.4f}")
    print(f"Unreliable  : {(~df['reliable']).mean() * 100:.2f}%")
    print(f"Wrote: {out_png}")
