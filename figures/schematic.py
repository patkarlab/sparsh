"""
Schematic of the training and the network of a model, for the manuscript.

Every number in the figure is read from the run's config.json, which scripts/train.py writes with the settings
of the model, the training and the data: the arrays, classes and CpGs; the dilution with normal marrow and the
coverage of the simulated nanopore reads; the input encoding; the layers, their dropout and activation; the
loss, the sampler and the optimiser; the folds and the inner validation split; and the threshold of the calls.
The small heatmaps that illustrate the steps are drawn from random numbers and stand for no real array.
"""

import json
import textwrap
from pathlib import Path

import numpy as np

ACTIVATIONS = {"gelu": "GELU", "relu": "ReLU", "leaky_relu": "leaky ReLU"}
SIM_TEXT = {"binary": "one call (0 or 1) per covered CpG", "reads": "the methylated share of its reads per CpG",
            "oneread": "the call of one read per covered CpG", "mask": "array values at the covered CpGs"}


def read_settings(model_dir: Path) -> dict:
    config = json.loads((Path(model_dir) / "config.json").read_text())
    missing = [k for k in ("model", "training", "data", "inference") if k not in config]
    if missing:
        raise ValueError(f"{model_dir}/config.json lacks {missing}; is it a run of scripts/train.py?")
    m, t, d, i = config["model"], config["training"], config["data"], config["inference"]
    return {
        "arrays": int(d["n_samples"]), "classes": int(m["n_classes"]), "cpgs": int(m["input_dim"]),
        "hidden": [int(h) for h in m["hidden_dims"]], "dropout": float(m["dropout"]),
        "activation": ACTIVATIONS.get(m.get("activation", "gelu"), m.get("activation", "")),
        "encoding": m.get("input_encoding", "midpoint"),
        "sim": t["train_sim"], "cov_min": float(t["cov_min"]), "cov_max": float(t["cov_max"]),
        "dilution": float(t.get("dilution_prob", 0.0)), "blast_min": float(t.get("blast_min", 1.0)),
        "gamma": float(t["focal_gamma"]), "smoothing": float(t["label_smoothing"]),
        "per_class": int(t["samples_per_class_per_batch"]), "lr": float(t["learning_rate"]),
        "weight_decay": float(t["weight_decay"]), "epochs": int(t["epochs"]),
        "patience": int(t["early_stopping_patience"]), "folds": int(t["n_folds"]),
        "inner": float(t["inner_val_frac"]), "calibrate": bool(t.get("calibrate", True)),
        "threshold": float(i["threshold"]), "fold_models": len(i.get("fold_models") or []),
    }


def _number(x: float) -> str:
    """1e-4 as 10⁻⁴, other numbers as they are."""
    exponent = int(np.floor(np.log10(abs(x)))) if x else 0
    if x and abs(x) < 0.01 and np.isclose(x, 10.0 ** exponent):
        sup = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
        return "10" + str(exponent).translate(sup)
    return f"{x:g}"


def plot_schematic(s: dict, path_stem) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patheffects
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

    beta_map = LinearSegmentedColormap.from_list("beta", ["#2166ac", "#f7f7f7", "#b2182b"])
    beta_map.set_bad("#d9d9d9")
    rng = np.random.default_rng(7)
    ink, soft = "#252525", "#737373"
    halo = [patheffects.withStroke(linewidth=1.6, foreground="white")]
    fig = plt.figure(figsize=(7.1, 4.6))

    def text(x, y, s_, size=5.5, **kw):
        fig.text(x, y, s_, fontsize=size, color=kw.pop("color", ink), **kw)

    def arrow(x0, y0, x1, y1):
        fig.patches.append(FancyArrowPatch((x0, y0), (x1, y1), transform=fig.transFigure, arrowstyle="-|>",
                                           mutation_scale=6, lw=0.7, color=soft))

    def panel(letter, x, y, title):
        text(x, y, letter, size=8, fontweight="bold", va="top")
        text(x + 0.025, y - 0.004, title, size=6.5, va="top")

    def frame(a):
        a.set_xticks([])
        a.set_yticks([])
        for side in a.spines.values():
            side.set_linewidth(0.4)

    # ------------------------------------------------------------------ a: input
    panel("a", 0.01, 0.985, "Training arrays and simulated nanopore input")
    top, height, width = 0.70, 0.205, 0.165
    xs = [0.03, 0.28, 0.53, 0.78]
    caption_y = top - 0.035
    profile = np.where(rng.random(48) < 0.5, rng.uniform(0.02, 0.25, 48), rng.uniform(0.75, 0.98, 48))

    ax = fig.add_axes([xs[0], top, width, height])
    arrays = np.clip(profile[None, :] + rng.normal(0, 0.12, (14, 48)), 0, 1)
    arrays[7:] = np.clip(1 - profile[None, :] * 0.8 + rng.normal(0, 0.12, (7, 48)), 0, 1)
    ax.imshow(arrays, cmap=beta_map, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    frame(ax)
    ax.set_ylabel("arrays", fontsize=5, labelpad=1)
    ax.set_xlabel("CpGs", fontsize=5, labelpad=1)
    text(xs[0] + width / 2, caption_y - 0.02, f"{s['arrays']:,} arrays in {s['classes']} classes,\n"
         f"{s['cpgs']:,} CpGs with beta values", ha="center", va="top", linespacing=1.4)

    ax = fig.add_axes([xs[1], top, width, height])
    normal = np.clip(0.55 + 0.35 * np.sin(np.arange(48) / 3.0), 0, 1)
    b = 0.6
    ax.imshow(np.vstack([profile, normal, b * profile + (1 - b) * normal]), cmap=beta_map, vmin=0, vmax=1,
              aspect="auto", interpolation="nearest")
    frame(ax)
    for row, label in enumerate(("leukemia", "normal marrow", f"mixed at {b:.0%} blasts")):
        ax.text(0.5, row, label, fontsize=5, ha="left", va="center", color=ink, path_effects=halo)
    for y in (0.5, 1.5):
        ax.axhline(y, color="white", lw=1.5)
    mix_text = (f"Leukemia arrays mixed with normal marrow\nin {s['dilution']:.0%} of draws, at "
                f"{s['blast_min']:.0%} to 100% blasts" if s["dilution"] > 0 else "No dilution")
    text(xs[1] + width / 2, caption_y - 0.02, mix_text, ha="center", va="top", linespacing=1.4)

    ax = fig.add_axes([xs[2], top, width, height])
    calls = np.full((14, 48), np.nan)
    for r in range(14):
        cover = rng.uniform(max(s["cov_min"], 0.15), s["cov_max"])
        seen = rng.random(48) < cover
        calls[r, seen] = (rng.random(int(seen.sum())) < arrays[r, seen]).astype(float)
    ax.imshow(np.ma.masked_invalid(calls), cmap=beta_map, vmin=0, vmax=1, aspect="auto", interpolation="nearest")
    frame(ax)
    ax.set_xlabel("CpGs; grey without reads", fontsize=5, labelpad=1)
    text(xs[2] + width / 2, caption_y - 0.02,
         f"Reads cover {s['cov_min']:.0%} to {s['cov_max']:.0%} of the CpGs, drawn\nfor each array; "
         f"{SIM_TEXT.get(s['sim'], s['sim'])}", ha="center", va="top", linespacing=1.4)

    ax = fig.add_axes([xs[3], top, width, height])
    seen = ~np.isnan(calls)
    share = np.maximum(seen.mean(axis=1, keepdims=True), 1e-3)
    if s["encoding"] == "scaled":
        encoded = np.where(seen, (np.nan_to_num(calls) - 0.5) / share, 0.0)
        enc_text = "Calls centered at 0.5 and divided\nby the covered share; no read: 0"
    else:
        encoded = np.where(seen, np.nan_to_num(calls) - 0.5, 0.0)
        enc_text = "Calls as they are;\nno read: 0.5"
    lim = 0.5 / float(np.quantile(share, 0.25))
    ax.imshow(encoded, cmap=beta_map, vmin=-lim, vmax=lim, aspect="auto", interpolation="nearest")
    frame(ax)
    ax.set_xlabel("network input", fontsize=5, labelpad=1)
    text(xs[3] + width / 2, caption_y - 0.02, enc_text, ha="center", va="top", linespacing=1.4)
    for x0 in xs[:3]:
        arrow(x0 + width + 0.015, top + height / 2, x0 + 0.25 - 0.015, top + height / 2)

    # ------------------------------------------------------------------ b: network
    panel("b", 0.01, 0.50, "Network")
    ax = fig.add_axes([0.03, 0.02, 0.52, 0.43])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    sizes = [s["cpgs"]] + s["hidden"] + [s["classes"]]
    heights = np.interp(np.log10(sizes), [np.log10(min(sizes)), np.log10(max(sizes))], [1.1, 2.9])
    centres = np.linspace(0.7, 7.3, len(sizes))
    mid = 3.95
    for k, (xc, hgt) in enumerate(zip(centres, heights)):
        hidden = 0 < k < len(sizes) - 1
        face = "#d9d9d9" if k == 0 else ("#bdd7e7" if hidden else "#fdd0a2")
        ax.add_patch(FancyBboxPatch((xc - 0.17, mid - hgt / 2), 0.34, hgt,
                                    boxstyle="round,pad=0.02,rounding_size=0.1", facecolor=face, edgecolor=soft,
                                    lw=0.5))
        if k + 1 < len(sizes):
            nxt, h2 = centres[k + 1], heights[k + 1]
            for f0 in np.linspace(-0.42, 0.42, 5):
                for f1 in np.linspace(-0.42, 0.42, 4):
                    ax.plot([xc + 0.17, nxt - 0.17], [mid + f0 * hgt, mid + f1 * h2], color="#bdbdbd", lw=0.25,
                            zorder=0)
        label = (f"input\n{sizes[k]:,} CpGs" if k == 0 else
                 f"output\n{sizes[k]} classes" if k == len(sizes) - 1 else f"{sizes[k]:,}")
        ax.text(xc, mid - heights[0] / 2 - 0.15, label, ha="center", va="top", fontsize=5, color=ink,
                linespacing=1.25)
    hid = centres[1:-1]
    if len(hid):
        y = mid + heights[0] / 2 + 0.1
        ax.plot([hid[0] - 0.2, hid[-1] + 0.2], [y, y], color=soft, lw=0.5)
        ax.text(hid.mean(), y + 0.12, f"each hidden layer: linear, batch normalization, {s['activation']}, "
                f"dropout {s['dropout']:g}", ha="center", va="bottom", fontsize=5, color=ink)
    ax.text(centres[-1] + 0.4, mid, "softmax with a\nfitted temperature" if s["calibrate"] else "softmax",
            ha="left", va="center", fontsize=5, color=ink, linespacing=1.3)
    training = (f"Focal loss (gamma {s['gamma']:g}) with label smoothing {s['smoothing']:g}; batches with "
                f"{s['per_class']} arrays of every class; AdamW with learning rate {_number(s['lr'])} and weight "
                f"decay {_number(s['weight_decay'])}; up to {s['epochs']} epochs, stopping after "
                f"{s['patience']} without improvement on the inner validation split.")
    ax.text(0.1, 0.0, textwrap.fill(training, 96), ha="left", va="bottom", fontsize=5, color=ink,
            linespacing=1.35)

    # ------------------------------------------------------------------ c: folds and the released model
    panel("c", 0.585, 0.50, "Cross-validation and the released model")
    ax = fig.add_axes([0.60, 0.02, 0.39, 0.43])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    n = s["folds"]
    row_h, gap, seg_w, x0 = 0.34, 0.09, 1.55, 1.1
    top_row = 5.5
    for r in range(n):
        y = top_row - r * (row_h + gap)
        ax.text(x0 - 0.15, y + row_h / 2, f"fold {r + 1}", ha="right", va="center", fontsize=5, color=ink)
        for c in range(n):
            x = x0 + c * seg_w
            held = c == r
            ax.add_patch(Rectangle((x, y), seg_w - 0.06, row_h, facecolor="#525252" if held else "#d9d9d9",
                                   edgecolor="none"))
            if not held:                                     # the inner split is drawn from every training part
                part = (seg_w - 0.06) * s["inner"]
                ax.add_patch(Rectangle((x + seg_w - 0.06 - part, y), part, row_h, facecolor="#6baed6",
                                       edgecolor="none"))
    y_key = top_row - n * (row_h + gap) - 0.12
    for k, (face, label) in enumerate((("#d9d9d9", "training"),
                                       ("#6baed6", f"inner validation, {s['inner']:.0%} of the training part"),
                                       ("#525252", "held out"))):
        yy = y_key - k * 0.3
        ax.add_patch(Rectangle((x0 - 1.0, yy), 0.25, 0.2, facecolor=face, edgecolor="none"))
        ax.text(x0 - 0.62, yy + 0.1, label, ha="left", va="center", fontsize=5, color=ink)
    note = ("The inner validation chooses the epoch and fits the temperature of each network; every array is "
            "scored once, by the network that did not see it.")
    ax.text(x0 - 1.0, y_key - 0.75, textwrap.fill(note, 80), ha="left", va="top", fontsize=5, color=ink,
            linespacing=1.35)
    released = (f"Released model: the {s['fold_models'] or n} fold networks, each with its temperature, with "
                f"their probabilities averaged. The most probable class is reported when it reaches "
                f"{s['threshold']:.2f}; otherwise the most probable family of related classes, then the most "
                "probable lineage, when its summed probability reaches it.")
    ax.add_patch(FancyBboxPatch((x0 - 1.05, 0.05), 9.9, 1.3, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor="#fff7ec", edgecolor="#fdae6b", lw=0.5))
    ax.text(x0 - 0.9, 0.7, textwrap.fill(released, 78), ha="left", va="center", fontsize=5, color=ink,
            linespacing=1.35)

    for ext, extra in (("png", {"dpi": 600}), ("pdf", {}), ("svg", {})):
        fig.savefig(f"{path_stem}.{ext}", **extra)
    plt.close(fig)
