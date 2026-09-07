"""
Simulations and figures for the multi-armed bandit article.

Pure simulation -> PNG. Every number in the article's charts comes from here.
    python bandit_sim.py
Writes images/*.png and images/data.json

Testbed: 10 Bernoulli arms. Machine 6 is the best at 60%, but nobody
running the experiment knows that -- that's the whole point.
"""
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- testbed ---

ARMS = [0.20, 0.32, 0.40, 0.12, 0.25, 0.60, 0.05, 0.45, 0.18, 0.28]
K = len(ARMS)
BEST = max(range(K), key=lambda i: ARMS[i])
P_STAR = ARMS[BEST]
HORIZON = 1000
OUT = "images"

# ------------------------------------------------------------- strategies ---
# Each returns (choices, counts). Regret is computed from the true rates,
# so it is *pseudo*-regret: the expected loss of the arms we picked, which
# is far less noisy than the realised coin flips.


def _init(rng):
    return [0] * K, [0.0] * K  # counts, sum of rewards


def run_greedy(rng, horizon=HORIZON):
    counts, wins = _init(rng)
    choices = []
    for t in range(horizon):
        if t < K:                       # one pull each to get started
            a = t
        else:
            a = max(range(K), key=lambda i: wins[i] / counts[i])
        r = 1.0 if rng.random() < ARMS[a] else 0.0
        counts[a] += 1
        wins[a] += r
        choices.append(a)
    return choices, counts


def run_eps_greedy(rng, horizon=HORIZON, eps=0.10):
    counts, wins = _init(rng)
    choices = []
    for t in range(horizon):
        if t < K:
            a = t
        elif rng.random() < eps:
            a = rng.randrange(K)        # explore: uniformly at random
        else:
            a = max(range(K), key=lambda i: wins[i] / counts[i])
        r = 1.0 if rng.random() < ARMS[a] else 0.0
        counts[a] += 1
        wins[a] += r
        choices.append(a)
    return choices, counts


def run_ucb1(rng, horizon=HORIZON):
    counts, wins = _init(rng)
    choices = []
    for t in range(horizon):
        if t < K:
            a = t
        else:
            ln = math.log(t + 1)
            a = max(range(K), key=lambda i: wins[i] / counts[i]
                    + math.sqrt(2.0 * ln / counts[i]))   # mean + optimism
        r = 1.0 if rng.random() < ARMS[a] else 0.0
        counts[a] += 1
        wins[a] += r
        choices.append(a)
    return choices, counts


def run_thompson(rng, horizon=HORIZON):
    counts, wins = _init(rng)
    choices = []
    for _ in range(horizon):
        # one draw from each arm's Beta posterior; play the highest draw
        a = max(range(K), key=lambda i: rng.betavariate(1 + wins[i],
                                                        1 + counts[i] - wins[i]))
        r = 1.0 if rng.random() < ARMS[a] else 0.0
        counts[a] += 1
        wins[a] += r
        choices.append(a)
    return choices, counts


STRATEGIES = [
    ("Greedy", run_greedy),
    ("ε-greedy (ε=0.1)", run_eps_greedy),
    ("UCB1", run_ucb1),
    ("Thompson sampling", run_thompson),
]

# ------------------------------------------------------------------ style ---

SURFACE, GRID = "#fcfcfb", "#e8e7e3"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [S1, S2, S3, S4]
MUTED = "#c9c8c2"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "font.size": 10.5,
})


def frame(w=9.0, h=5.0, title="", subtitle=""):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=0.085, right=0.965, top=0.80, bottom=0.13)
    if title:
        fig.text(0.085, 0.945, title, ha="left", va="top",
                 fontsize=16.5, color=INK, weight=600)
    if subtitle:
        fig.text(0.085, 0.875, subtitle, ha="left", va="top",
                 fontsize=11.5, color=INK2)
    return fig, ax


def style(ax, ygrid=True, xgrid=False):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.9)
    ax.tick_params(colors=INK2, labelsize=9.5, length=0, pad=6)
    ax.set_axisbelow(True)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.9, linestyle="-")
    if xgrid:
        ax.grid(axis="x", color=GRID, linewidth=0.9, linestyle="-")


def legend(ax, labels, colors, loc="upper left", marker="line"):
    """Legend is always present for >=2 series. Text stays in ink tokens."""
    handles = [Line2D([], [], color=c, lw=2.4, marker="o", markersize=0,
                      solid_capstyle="round") if marker == "line" else
               Line2D([], [], color=c, lw=0, marker="s", markersize=8)
               for c in colors]
    lg = ax.legend(handles, labels, loc=loc, frameon=False, fontsize=10,
                   labelcolor=INK2, handlelength=1.6, handletextpad=0.7,
                   borderaxespad=0.2, labelspacing=0.55)
    return lg


def declutter(ys, min_gap):
    """Push overlapping direct labels apart, keeping their order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    out = list(ys)
    for n, i in enumerate(order):
        if n and out[i] - out[order[n - 1]] < min_gap:
            out[i] = out[order[n - 1]] + min_gap
    return out


def save(fig, name, note=None):
    if note:
        fig.text(0.085, 0.022, note, ha="left", va="bottom",
                 fontsize=9, color=INK3)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("wrote", path)


def end_dot(ax, x, y, color):
    """>=8px marker with a 2px surface ring so it stays legible on crossings."""
    ax.plot([x], [y], marker="o", markersize=8, color=color,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=6,
            clip_on=False, linestyle="none")


data = {}

# ------------------------------------------------- fig 1: the hidden rates ---

def fig_hidden_rates():
    fig, ax = frame(9.0, 4.6,
                    "The information you do not have",
                    "True payout rate of each machine. You never see this chart — you only\n"
                    "see wins and losses, one pull at a time.")
    xs = np.arange(1, K + 1)
    colors = [S1 if i == BEST else MUTED for i in range(K)]
    ax.bar(xs, ARMS, width=0.52, color=colors, zorder=3)
    for i, p in enumerate(ARMS):
        ax.text(i + 1, p + 0.014, f"{p:.0%}", ha="center", va="bottom",
                fontsize=9.5, color=INK if i == BEST else INK2,
                weight=600 if i == BEST else "normal")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"M{i}" for i in xs])
    ax.set_xlim(0.4, K + 0.6)
    ax.set_ylim(0, 0.70)
    ax.set_yticks([0, 0.15, 0.30, 0.45, 0.60])
    ax.set_yticklabels(["0%", "15%", "30%", "45%", "60%"])
    ax.set_xlabel("Machine", color=INK2, fontsize=10, labelpad=8)
    style(ax)
    legend(ax, ["Best machine (M6)", "Every other machine"],
           [S1, MUTED], loc="upper right", marker="patch")
    save(fig, "01-the-hidden-rates.png",
         "10 Bernoulli arms · the testbed used in every figure below")
    data["arms"] = ARMS

# --------------------------------------- fig 2: how much to explore (U-curve) ---

def fig_epsilon_sweep(runs=250):
    eps_grid = [round(0.05 * i, 2) for i in range(21)]
    means = []
    for eps in eps_grid:
        tot = 0.0
        for r in range(runs):
            rng = random.Random(9000 + r)
            ch, _ = run_eps_greedy(rng, HORIZON, eps)
            tot += sum(ARMS[a] for a in ch)     # expected winnings
        means.append(tot / runs)
    best_i = max(range(len(eps_grid)), key=lambda i: means[i])

    fig, ax = frame(9.0, 5.0,
                    "There is a sweet spot, and it is not zero",
                    "Expected winnings over 1,000 pulls as ε-greedy is dialled from pure exploitation\n"
                    "to pure randomness. Average of 250 simulated runs per setting.")
    ax.plot(eps_grid, means, color=S1, lw=2.4, solid_capstyle="round", zorder=4)
    end_dot(ax, eps_grid[best_i], means[best_i], S1)
    ax.annotate(f"best here: ε ≈ {eps_grid[best_i]:.2f}\n{means[best_i]:.0f} wins",
                xy=(eps_grid[best_i], means[best_i]),
                xytext=(eps_grid[best_i] + 0.075, means[best_i] + 12),
                fontsize=10, color=INK, weight=600, va="center",
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.9,
                                shrinkA=2, shrinkB=6))
    ax.annotate("ε = 0\nnever explore\n(gets stuck on a lucky loser)",
                xy=(0.0, means[0]), xytext=(0.055, means[0] - 20),
                fontsize=9.5, color=INK2, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.9,
                                shrinkA=2, shrinkB=4))
    ax.annotate("ε = 1\npure random\n(never cashes in)",
                xy=(1.0, means[-1]), xytext=(0.955, means[-1] + 45),
                fontsize=9.5, color=INK2, ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.9,
                                shrinkA=2, shrinkB=4))
    ax.axhline(P_STAR * HORIZON, color=INK3, lw=0.9, linestyle=(0, (4, 3)), zorder=2)
    ax.text(1.0, P_STAR * HORIZON + 8, "perfect play — 600 wins",
            ha="right", va="bottom", fontsize=9.5, color=INK2)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("ε  (share of pulls spent exploring at random)",
                  color=INK2, fontsize=10, labelpad=8)
    ax.set_ylabel("Expected winnings", color=INK2, fontsize=10, labelpad=8)
    style(ax)
    save(fig, "03-explore-exploit-curve.png")
    data["epsilon_sweep"] = {"eps": eps_grid, "winnings": [round(m, 1) for m in means]}

# ------------------------------------------- fig 3: small samples lie ---------

def fig_small_samples(trials=8000):
    p = 0.60
    ns = [1, 2, 3, 5, 8, 13, 20, 32, 50, 80, 130, 200, 320, 500, 800, 1000]
    rng = np.random.default_rng(7)
    lo, hi, med = [], [], []
    for n in ns:
        obs = rng.binomial(n, p, trials) / n
        lo.append(np.percentile(obs, 5))
        hi.append(np.percentile(obs, 95))
        med.append(np.percentile(obs, 50))

    fig, ax = frame(9.0, 5.0,
                    "Small samples lie",
                    "What a 60% machine can look like after n pulls. Shaded band holds 90% of\n"
                    "outcomes across 8,000 simulations.")
    ax.fill_between(ns, lo, hi, color=S1, alpha=0.12, linewidth=0, zorder=2)
    ax.plot(ns, hi, color=S1, lw=2.0, solid_capstyle="round", zorder=4)
    ax.plot(ns, lo, color=S1, lw=2.0, solid_capstyle="round", zorder=4)
    ax.axhline(p, color=INK3, lw=0.9, linestyle=(0, (4, 3)), zorder=3)
    ax.text(1.15, p + 0.02, "the truth — 60%", ha="left", va="bottom",
            fontsize=9.5, color=INK2)
    ax.annotate("after 2 pulls, a 60% machine\ncan look like a 0% dud\nor a 100% jackpot",
                xy=(2, 0.97), xytext=(3.6, 0.86), fontsize=9.5, color=INK2,
                va="top", arrowprops=dict(arrowstyle="-", color=INK3, lw=0.9,
                                          shrinkA=2, shrinkB=4))
    ax.annotate("after 200 pulls it is\npinned to 55–65%",
                xy=(200, 0.655), xytext=(70, 0.32), fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.9,
                                shrinkA=2, shrinkB=4))
    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_xticklabels(["1", "10", "100", "1,000"])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Pulls of this one machine (log scale)", color=INK2,
                  fontsize=10, labelpad=8)
    ax.set_ylabel("Observed win rate", color=INK2, fontsize=10, labelpad=8)
    style(ax)
    save(fig, "02-small-samples-lie.png",
         "8,000 simulations per sample size · band spans the 5th to 95th percentile")
    data["sampling_band"] = {"n": ns, "p05": [round(v, 3) for v in lo],
                             "p95": [round(v, 3) for v in hi]}

# ------------------------------------------------- fig 4 + 5: the showdown ---

def simulate_all(runs=400, horizon=HORIZON):
    regret = {}
    alloc = {name: np.zeros(K) for name, _ in STRATEGIES}
    for name, fn in STRATEGIES:
        acc = np.zeros(horizon)
        for r in range(runs):
            rng = random.Random(1234 + r)
            ch, counts = fn(rng, horizon)
            step = np.fromiter((P_STAR - ARMS[a] for a in ch),
                               dtype=float, count=horizon)
            acc += np.cumsum(step)
            alloc[name] += np.asarray(counts, dtype=float)
        regret[name] = acc / runs
        alloc[name] = alloc[name] / runs
    return regret, alloc


def fig_regret(regret):
    fig, ax = frame(9.0, 5.2,
                    "Four strategies, 1,000 pulls, same machines",
                    "Cumulative regret = winnings lost versus knowing the best machine from pull 1.\n"
                    "Lower is better; a flattening curve means the strategy has stopped making mistakes.")
    xs = np.arange(1, HORIZON + 1)
    names = [n for n, _ in STRATEGIES]
    finals = []
    for (name, _), c in zip(STRATEGIES, SERIES):
        ax.plot(xs, regret[name], color=c, lw=2.4, solid_capstyle="round", zorder=4)
        end_dot(ax, HORIZON, regret[name][-1], c)
        finals.append(regret[name][-1])

    ylo, yhi = 0, max(finals) * 1.10
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0, HORIZON)
    gap = (yhi - ylo) * 0.075
    placed = declutter(finals, gap)
    for name, c, y, ly in zip(names, SERIES, finals, placed):
        ax.annotate(f"{name} — {y:.0f}", xy=(HORIZON, y),
                    xytext=(HORIZON + 28, ly), fontsize=10, color=INK2,
                    va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.9,
                                    shrinkA=1, shrinkB=1))
    ax.set_xticks([0, 250, 500, 750, 1000])
    ax.set_xticklabels(["0", "250", "500", "750", "1,000"])
    ax.set_xlabel("Pulls", color=INK2, fontsize=10, labelpad=8)
    ax.set_ylabel("Cumulative regret (wins forgone)", color=INK2,
                  fontsize=10, labelpad=8)
    style(ax)
    legend(ax, names, SERIES, loc="upper left")
    fig.subplots_adjust(right=0.775)
    save(fig, "04-regret-curves.png",
         "Average of 400 simulated runs per strategy · pseudo-regret against the true rates")
    data["final_regret"] = {n: round(float(regret[n][-1]), 1) for n in names}
    data["regret_curves"] = {n: [round(float(regret[n][i]), 2)
                                 for i in (49, 99, 249, 499, 999)] for n in names}


def fig_long_horizon(runs=100, horizon=20000):
    """Does UCB1's bad showing at 1,000 pulls survive a longer game? No."""
    regret, _ = simulate_all(runs=runs, horizon=horizon)
    fig, ax = frame(9.0, 5.2,
                    "Give it 20,000 pulls and the ranking flips",
                    "The same four strategies over a longer game. What matters is not where a curve\n"
                    "sits at pull 1,000 — it is whether the curve ever bends.")
    xs = np.arange(1, horizon + 1)
    names = [n for n, _ in STRATEGIES]
    finals = [regret[n][-1] for n in names]
    for name, c in zip(names, SERIES):
        ax.plot(xs, regret[name], color=c, lw=2.4, solid_capstyle="round", zorder=4)
        end_dot(ax, horizon, regret[name][-1], c)

    yhi = max(finals) * 1.10
    ax.set_ylim(0, yhi)
    ax.set_xlim(0, horizon)
    placed = declutter(finals, yhi * 0.075)
    for name, y, ly in zip(names, finals, placed):
        ax.annotate(f"{name} — {y:,.0f}", xy=(horizon, y),
                    xytext=(horizon + 560, ly), fontsize=10, color=INK2,
                    va="center", ha="left", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.9,
                                    shrinkA=1, shrinkB=1))
    ax.annotate("straight line = still\nlosing at the same rate,\nforever",
                xy=(12000, regret["Greedy"][11999]),
                xytext=(9000, yhi * 0.86), fontsize=9.5, color=INK2, va="top",
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.9,
                                shrinkA=2, shrinkB=4))
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.set_xticklabels(["0", "5,000", "10,000", "15,000", "20,000"])
    ax.set_xlabel("Pulls", color=INK2, fontsize=10, labelpad=8)
    ax.set_ylabel("Cumulative regret (wins forgone)", color=INK2,
                  fontsize=10, labelpad=8)
    style(ax)
    legend(ax, names, SERIES, loc="upper left")
    fig.subplots_adjust(right=0.775, bottom=0.16)
    save(fig, "05-long-horizon.png",
         f"Average of {runs} simulated runs per strategy")
    data["final_regret_20k"] = {n: round(float(regret[n][-1]), 1) for n in names}
    data["regret_20k_at"] = {n: {str(t): round(float(regret[n][t - 1]), 1)
                                 for t in (1000, 5000, 10000, 20000)}
                             for n in names}


def fig_allocation(alloc):
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.0))
    fig.subplots_adjust(left=0.075, right=0.97, top=0.755, bottom=0.105,
                        hspace=0.52, wspace=0.16)
    fig.text(0.075, 0.962, "Where the 1,000 pulls actually went",
             ha="left", va="top", fontsize=16.5, color=INK, weight=600)
    fig.text(0.075, 0.903,
             "Average pulls per machine. A good strategy spends almost everything on M6 —\n"
             "and spends its leftovers on checking, not on machines it already ruled out.",
             ha="left", va="top", fontsize=11.5, color=INK2)
    xs = np.arange(1, K + 1)
    for ax, (name, _), c in zip(axes.ravel(), STRATEGIES, SERIES):
        counts = alloc[name]
        colors = [c if i == BEST else MUTED for i in range(K)]
        ax.bar(xs, counts, width=0.56, color=colors, zorder=3)
        ax.text(BEST + 1, counts[BEST] + 28, f"{counts[BEST]:.0f}",
                ha="center", va="bottom", fontsize=9.5, color=INK, weight=600)
        share = counts[BEST] / HORIZON
        ax.set_title(f"{name}   ·   {share:.0%} of pulls on M6",
                     loc="left", fontsize=11, color=INK, weight=600, pad=10)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"M{i}" for i in xs], fontsize=8.5)
        ax.set_ylim(0, 1080)
        ax.set_yticks([0, 250, 500, 750, 1000])
        ax.set_yticklabels(["0", "250", "500", "750", "1,000"])
        style(ax)
    fig.text(0.075, 0.018,
             "Average of 400 simulated runs · M6 (the 60% machine) highlighted in colour",
             ha="left", va="bottom", fontsize=9, color=INK3)
    fig.savefig(os.path.join(OUT, "06-arm-allocation.png"), dpi=200)
    plt.close(fig)
    print("wrote", os.path.join(OUT, "06-arm-allocation.png"))
    data["allocation_best_arm"] = {n: round(float(alloc[n][BEST]), 1)
                                   for n, _ in STRATEGIES}

# ------------------------------------------- fig 6: bandit vs an A/B test ----

def fig_ab_vs_bandit(days=14, per_day=1500, runs=80):
    pa, pb = 0.050, 0.065          # B is the real winner, by a realistic margin
    share = np.zeros(days)
    conv_bandit = np.zeros(days)
    for r in range(runs):
        rng = random.Random(4242 + r)
        counts = [0, 0]
        wins = [0.0, 0.0]
        for d in range(days):
            picks_b = 0
            for _ in range(per_day):
                a = 0 if rng.betavariate(1 + wins[0], 1 + counts[0] - wins[0]) > \
                          rng.betavariate(1 + wins[1], 1 + counts[1] - wins[1]) else 1
                p = pa if a == 0 else pb
                r_ = 1.0 if rng.random() < p else 0.0
                counts[a] += 1
                wins[a] += r_
                picks_b += a
                conv_bandit[d] += r_
            share[d] += picks_b / per_day
    share /= runs
    conv_bandit /= runs

    fig, ax = frame(9.0, 5.0,
                    "The A/B test keeps paying for an answer it already has",
                    "Share of traffic sent to the variant that actually wins (5.0% vs 6.5% conversion),\n"
                    "over a two-week test at 1,500 visitors a day.")
    xs = np.arange(1, days + 1)
    ab = np.full(days, 0.5)
    ax.plot(xs, share, color=S1, lw=2.4, solid_capstyle="round", zorder=5)
    ax.plot(xs, ab, color=S2, lw=2.4, solid_capstyle="round", zorder=4)
    end_dot(ax, days, share[-1], S1)
    end_dot(ax, days, ab[-1], S2)
    for y, label in ((share[-1], f"Bandit — {share[-1]:.0%}"),
                     (0.5, "A/B test — 50%")):
        ax.annotate(label, xy=(days, y), xytext=(days + 0.25, y),
                    fontsize=10, color=INK2, va="center", ha="left",
                    annotation_clip=False)
    ax.fill_between(xs, share, ab, where=(share > ab), color=S1,
                    alpha=0.08, linewidth=0, zorder=2)
    ax.annotate("every visitor in here is one the A/B test\nsent to the losing variant on purpose",
                xy=(9, 0.5 + (share[8] - 0.5) / 2), xytext=(7.6, 0.235),
                fontsize=9.5, color=INK2, ha="left", va="top",
                arrowprops=dict(arrowstyle="-", color=INK3, lw=0.9,
                                shrinkA=2, shrinkB=4))
    ax.set_xlim(1, days)
    ax.set_xticks([1, 4, 7, 10, 14])
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("Day of test", color=INK2, fontsize=10, labelpad=8)
    ax.set_ylabel("Traffic to the winning variant", color=INK2,
                  fontsize=10, labelpad=8)
    style(ax)
    legend(ax, ["Bandit (Thompson)", "Fixed 50/50 A/B test"], [S1, S2],
           loc="lower left")
    fig.subplots_adjust(right=0.815, bottom=0.175)
    total_ab = days * per_day * (pa + pb) / 2
    total_bandit = float(conv_bandit.sum())
    save(fig, "07-bandit-vs-abtest.png",
         f"Average of {runs} simulated runs · extra conversions from the bandit: "
         f"{total_bandit - total_ab:,.0f} over the fortnight")
    data["ab_vs_bandit"] = {
        "share_to_winner_by_day": [round(float(v), 3) for v in share],
        "conversions_ab": round(total_ab, 1),
        "conversions_bandit": round(total_bandit, 1),
    }

# ------------------------------------------------------------------- main ----

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_hidden_rates()
    fig_small_samples()
    fig_epsilon_sweep()
    regret, alloc = simulate_all()
    fig_regret(regret)
    fig_long_horizon()
    fig_allocation(alloc)
    fig_ab_vs_bandit()
    with open(os.path.join(OUT, "data.json"), "w") as f:
        json.dump(data, f, indent=2)
    print("wrote", os.path.join(OUT, "data.json"))
