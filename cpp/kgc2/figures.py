#!/usr/bin/env python3
"""Figures for the kgc demonstration.

  figures.py <img.bin> <selected.bin> <outdir> [nodescores.csv] [params.txt]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


def read_hdr(p):
    txt = open(p).read()
    out, i = {}, txt.find("\n") + 1
    while i < len(txt):
        e = txt.find("=", i)
        if e < 0:
            break
        k = txt[i:e].strip().lower()
        j = e + 1
        while j < len(txt) and txt[j] in " \t":
            j += 1
        if j < len(txt) and txt[j] == "{":
            c = txt.find("}", j)
            v = txt[j + 1:c]
            i = txt.find("\n", c)
            i = len(txt) if i < 0 else i + 1
        else:
            c = txt.find("\n", j)
            c = len(txt) if c < 0 else c
            v = txt[j:c]
            i = c + 1
        out[k] = v.strip()
    return out


def read_envi(p):
    h = read_hdr(os.path.splitext(p)[0] + ".hdr")
    ns, nl, nb = int(h["samples"]), int(h["lines"]), int(h["bands"])
    a = np.fromfile(p, dtype="<f4")
    il = h.get("interleave", "bsq").lower()
    if il == "bsq":
        a = a.reshape(nb, nl, ns)
    elif il == "bip":
        a = a.reshape(nl, ns, nb).transpose(2, 0, 1)
    else:
        a = a.reshape(nl, nb, ns).transpose(1, 0, 2)
    return a, h


def stretch(b, lo=2, hi=98):
    b = np.nan_to_num(b, nan=np.nanmedian(b), posinf=0.0, neginf=0.0)
    a, z = np.nanpercentile(b, [lo, hi])
    return np.clip((b - a) / max(z - a, 1e-9), 0, 1)


def composite(img):
    """RGB composite for any band count: 3+ bands -> first three as R,G,B;
    2 bands -> band1/band2/mean; 1 band -> greyscale."""
    nb = img.shape[0]
    if nb >= 3:
        return np.dstack([stretch(img[0]), stretch(img[1]), stretch(img[2])])
    if nb == 2:
        return np.dstack([stretch(img[0]), stretch(img[1]),
                          stretch(0.5 * (img[0] + img[1]))])
    g = stretch(img[0])
    return np.dstack([g, g, g])


def main():
    imgp, selp, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    scores = sys.argv[4] if len(sys.argv) > 4 else None
    params = sys.argv[5] if len(sys.argv) > 5 else None
    os.makedirs(outdir, exist_ok=True)

    img, ih = read_envi(imgp)
    sel, sh = read_envi(selp)
    mask, hint, pure, llr, cw = sel[0], sel[1], sel[2], sel[3], sel[4]
    rgb = composite(img)

    # ---- three-pane comparison ------------------------------------------
    # Stacked vertically: each panel is then reproduced far larger on the page
    # than three side by side in a two-column layout allow.
    ar = rgb.shape[0] / float(rgb.shape[1])
    fig, ax = plt.subplots(3, 1, figsize=(6.0, 3 * (6.0 * ar + 0.55)))
    bin_cmap = ListedColormap(["#12161c", "#ffd166"])
    ax[0].imshow(hint, cmap=bin_cmap, vmin=0, vmax=1, interpolation="nearest")
    ax[0].set_title("Hint mask (input)\n%d px, %.1f%% of scene"
                    % (hint.sum(), 100 * hint.mean()), fontsize=10)
    ax[1].imshow(rgb, interpolation="nearest")
    ax[1].set_title("Input imagery\n%d bands, composite of first three"
                    % img.shape[0], fontsize=10)
    sel_cmap = ListedColormap(["#12161c", "#4cc9f0"])
    ax[2].imshow(mask, cmap=sel_cmap, vmin=0, vmax=1, interpolation="nearest")
    inter = float(((mask > .5) & (hint > .5)).sum())
    ax[2].set_title("Selected class (output)\n%d px, precision %.3f, recall %.3f"
                    % (mask.sum(), inter / max(mask.sum(), 1),
                       inter / max(hint.sum(), 1)), fontsize=10)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "three_pane.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ---- agreement map ---------------------------------------------------
    agree = np.zeros(mask.shape)
    agree[(mask > .5) & (hint > .5)] = 1      # true positive
    agree[(mask > .5) & (hint <= .5)] = 2     # outside the hint
    agree[(mask <= .5) & (hint > .5)] = 3     # hint not captured
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    cm = ListedColormap(["#12161c", "#4cc9f0", "#ef476f", "#8d99ae"])
    ax[0].imshow(agree, cmap=cm, vmin=0, vmax=3, interpolation="nearest")
    ax[0].set_title("Agreement: cyan both, red class only, grey hint only",
                    fontsize=9)
    im = ax[1].imshow(llr, cmap="RdBu_r",
                      vmin=-np.percentile(np.abs(llr), 99),
                      vmax=np.percentile(np.abs(llr), 99),
                      interpolation="nearest")
    ax[1].set_title("Codeword log-likelihood ratio log(p_hint/p_bg), nats",
                    fontsize=9)
    plt.colorbar(im, ax=ax[1], fraction=0.035)
    for a in ax:
        a.set_xticks([])
        a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "agreement.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)

    # ---- codebook / mode map --------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    ax[0].imshow(cw, cmap="tab20", interpolation="nearest")
    ax[0].set_title("Density modes used as the evidence codebook (%d)"
                    % int(cw.max() + 1), fontsize=9)
    ax[0].set_xticks([]); ax[0].set_yticks([])
    if scores and os.path.exists(scores):
        d = np.genfromtxt(scores, delimiter=",", names=True)
        npix = np.atleast_1d(d["n_pixels"])
        mi = np.atleast_1d(d["mi_pure"])
        lift = np.atleast_1d(d["lift"])
        ok = lift > 1
        ax[1].scatter(npix[~ok], mi[~ok], s=3, c="#8d99ae", alpha=.35,
                      label="lift $\\leq$ 1 (rejected)")
        ax[1].scatter(npix[ok], mi[ok], s=4, c="#4cc9f0", alpha=.6,
                      label="lift > 1 (eligible)")
        if ok.any():
            j = np.argmax(np.where(ok, mi, -1))
            ax[1].scatter([npix[j]], [mi[j]], s=90, facecolors="none",
                          edgecolors="#ef476f", linewidths=2, label="selected")
        ax[1].set_xscale("log")
        ax[1].set_xlabel("class size (pixels)")
        ax[1].set_ylabel("mutual information with purified hint (nats)")
        ax[1].legend(fontsize=7, loc="upper left")
        ax[1].set_title("Every dendrogram node scored", fontsize=9)
        ax[1].grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "selection.png"), dpi=140,
                bbox_inches="tight")
    plt.close(fig)
    print("figures written to", outdir)


if __name__ == "__main__":
    if sys.argv[1] != "--extras":
        main()


def extras():
    """Optional appendix figures: navigation ladder and hierarchy profile.

        figures.py --extras <prefix> <outdir>
    """
    pre, outdir = sys.argv[2], sys.argv[3]
    os.makedirs(outdir, exist_ok=True)
    hint, _ = read_envi(pre + "_state_hint.bin")
    hint = hint[0]

    panels = []
    for tag, lab in [("_navdn1", "one level finer"),
                     ("_selected", "automatic selection"),
                     ("_navup1", "one level coarser")]:
        p = pre + tag + ".bin"
        if os.path.exists(p):
            a, _ = read_envi(p)
            panels.append((a[0], lab))
    if panels:
        ar = panels[0][0].shape[0] / float(panels[0][0].shape[1])
        fig, ax = plt.subplots(len(panels), 1,
                               figsize=(6.0, len(panels) * (6.0 * ar + 0.55)))
        ax = np.atleast_1d(ax)
        cm = ListedColormap(["#12161c", "#4cc9f0"])
        for k, (m, lab) in enumerate(panels):
            ax[k].imshow(m, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            i = float(((m > .5) & (hint > .5)).sum())
            ax[k].set_title("%s\n%d px, precision %.3f, recall %.3f"
                            % (lab, m.sum(), i / max(m.sum(), 1),
                               i / max(hint.sum(), 1)), fontsize=10)
            ax[k].set_xticks([]); ax[k].set_yticks([])
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "navigation.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)

    lv = pre + "_levels.csv"
    if os.path.exists(lv):
        d = np.genfromtxt(lv, delimiter=",", names=True)
        fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.8))
        ax[0].plot(d["n_classes"], d["epsilon_k"], lw=.8, c="#4cc9f0")
        ax[0].set_xscale("log")
        ax[0].set_xlabel("classes remaining")
        ax[0].set_ylabel("saddle log-density at merge")
        ax[0].set_title("Merge height against hierarchy depth", fontsize=9)
        ax[0].grid(alpha=.2)
        sc = pre + "_nodescores.csv"
        if os.path.exists(sc):
            e = np.genfromtxt(sc, delimiter=",", names=True)
            pr = np.atleast_1d(e["precision"])
            rc = np.atleast_1d(e["recall"])
            mi = np.atleast_1d(e["mi_pure"])
            ax[1].scatter(rc, pr, s=3, c=mi, cmap="viridis", alpha=.6)
            j = int(np.argmax(mi))
            ax[1].scatter([rc[j]], [pr[j]], s=90, facecolors="none",
                          edgecolors="#ef476f", linewidths=2)
            ax[1].set_xlabel("recall against the hint")
            ax[1].set_ylabel("precision against the hint")
            ax[1].set_title("Precision-recall of every class; colour is MI",
                            fontsize=9)
            ax[1].grid(alpha=.2)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "hierarchy.png"), dpi=140,
                    bbox_inches="tight")
        plt.close(fig)
    print("extra figures written to", outdir)


if len(sys.argv) > 1 and sys.argv[1] == "--extras":
    extras()
    sys.exit(0)
