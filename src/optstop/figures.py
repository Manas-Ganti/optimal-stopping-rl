"""Figures. Every one is drawn from `results/metrics.json`, never from in-memory or placeholder
data. Colours: categorical slots 1-3 of the validated reference palette, fixed per entity
(learned agent / LSM / hold-to-expiry), neutral ink for exact references."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
COLORS = {"fitted_q": "#eb6834", "lsm": "#2a78d6", "hold": "#1baf7a", "exact": "#0b0b0b"}
LABELS = {"fitted_q": "Fitted Q-iteration (learned)", "lsm": "Longstaff-Schwartz",
          "hold": "Hold to expiry (European)"}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def _fig(w=7.5, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=SURFACE)
    style(ax)
    return fig, ax


def _save(fig, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_boundaries(metrics_path: Path, out: Path) -> Path:
    """Headline figure: learned vs LSM vs exact lattice boundary, S/K against time to expiry."""
    m = json.loads(Path(metrics_path).read_text())
    bd = m["boundary"]
    c = bd["contract"]
    fig, ax = _fig()
    spec = [
        ("lattice", dict(color=COLORS["exact"], lw=1.2, ls="-", label="Exact Bermudan (CRR lattice)")),
        ("lsm", dict(color=COLORS["lsm"], lw=0, marker="o", ms=4.5, label=LABELS["lsm"])),
        ("fitted_q", dict(color=COLORS["fitted_q"], lw=2, ls="-", label=LABELS["fitted_q"])),
    ]
    for name, st in spec:
        tau = np.array(bd["curves"][name]["tau"], dtype=float)
        b = np.array([np.nan if v is None else v for v in bd["curves"][name]["b"]])
        keep = tau > 1e-12  # at expiry the boundary is x = 1 by definition
        ax.plot(tau[keep], b[keep], **st)
    if bd.get("perpetual_b") is not None:
        ax.axhline(bd["perpetual_b"], color=INK_2, lw=1, ls=":")
        ax.text(c["expiry"], bd["perpetual_b"] + 0.004, f"perpetual-put limit {bd['perpetual_b']:.3f}",
                fontsize=7.5, color=INK_2, ha="left", va="bottom")
    agree = bd["agreement"]["fitted_q_vs_lsm"]
    ax.set_title(f"American put exercise boundary — K={c['strike']:g}, r={c['rate']:g}, "
                 f"σ={bd['vol']:g}, {c['steps']} exercise dates", fontsize=10, color=INK, loc="left")
    ax.text(0.01, 0.97, f"|learned − LSM| over {agree['n_dates']} dates: mean {agree['mean_abs']:.4f}, "
            f"max {agree['max_abs']:.4f} (S/K)   seed {m['boundary']['seed']}",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=INK_2)
    ax.set_xlabel("Time to expiry τ (years)", color=INK_2, fontsize=9)
    ax.set_ylabel("Boundary S*/K  (exercise below)", color=INK_2, fontsize=9)
    ax.invert_xaxis()
    ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.1), frameon=False, labelcolor=INK)
    return _save(fig, out)


def plot_degradation(metrics_path: Path, out: Path) -> Path:
    """Regret vs a model-aware reference (LSM refitted on the true dynamics), per regime."""
    m = json.loads(Path(metrics_path).read_text())
    regimes = m["evaluate"]["misspecification"]["regimes"]
    names = list(regimes)
    policies = ["fitted_q", "lsm", "hold"]
    fig, ax = _fig(7.5, 4.2)
    width = 0.26
    x = np.arange(len(names))
    for j, p in enumerate(policies):
        est = [regimes[r]["regret_vs_refit"][p] for r in names]
        mean = np.array([e["mean"] for e in est])
        lo = mean - np.array([e["ci_low"] for e in est])
        hi = np.array([e["ci_high"] for e in est]) - mean
        ax.bar(x + (j - 1) * width, mean, width * 0.92, color=COLORS[p], label=LABELS[p],
               yerr=[lo, hi], error_kw=dict(ecolor=INK_2, lw=1, capsize=2))
    ax.axhline(0, color=INK_2, lw=0.8)
    ax.set_xticks(x, [regimes[r]["label"] for r in names], fontsize=8, color=INK)
    ax.set_ylabel("Value lost vs refitted LSM ($, K=40)", color=INK_2, fontsize=9)
    ax.set_title("Policies trained on GBM, evaluated off-model (95% CI, paired paths)",
                 fontsize=10, color=INK, loc="left")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    return _save(fig, out)


def plot_real(metrics_path: Path, out: Path) -> Path:
    """Per-ticker mean paired difference (learned − LSM) on real windows."""
    m = json.loads(Path(metrics_path).read_text())
    real = m["evaluate"]["real"]
    rows = sorted(real["per_ticker"], key=lambda r: r["mean_diff_per_100_strike"])
    fig, ax = _fig(7.5, 0.28 * len(rows) + 1.6)
    y = np.arange(len(rows))
    ax.scatter([r["mean_diff_per_100_strike"] for r in rows], y, s=36, color=COLORS["fitted_q"],
               zorder=3)
    ax.axvline(0, color=INK_2, lw=0.8)
    pooled = real["paired_diff_fqi_minus_lsm_per_100_strike"]
    ax.axvspan(pooled["ci_low"], pooled["ci_high"], color=GRID, alpha=0.8, zorder=1)
    ax.set_yticks(y, [f"{r['ticker']} (n={r['n']})" for r in rows], fontsize=7.5, color=INK)
    ax.set_xlabel("Mean realised value, learned − LSM  (per 100 of strike)", color=INK_2, fontsize=9)
    ax.set_title(f"Real paths: {real['n_windows']} one-year ATM puts, {real['n_tickers']} tickers. "
                 f"Band = pooled 95% CI (year-cluster bootstrap)", fontsize=9, color=INK, loc="left")
    return _save(fig, out)
