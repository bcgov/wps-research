#!/usr/bin/env python3
"""Figure plates for the kgc demonstration.

  plates.py <image.bin> <prefix> <outdir> <tag> [title]

Produces, per case:
  plate_case.png   six panels -- hint, imagery and the evidence field on the
                   left; the best class at K-1, at the chosen K, and at K+1 on
                   the right
  plate_agreement.png  agreement map and the evidence field side by side
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from figures import read_envi, composite

BIN = ListedColormap(["#12161c", "#ffd166"])
SEL = ListedColormap(["#12161c", "#4cc9f0"])
AGR = ListedColormap(["#12161c", "#4cc9f0", "#ef476f", "#8d99ae"])


def params(prefix):
    out = {}
    p = prefix + "_params.txt"
    if os.path.exists(p):
        for line in open(p):
            if "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.split("#")[0].strip()
    return out


def pr(m, h):
    i = float(((m > .5) & (h > .5)).sum())
    return i / max(float(m.sum()), 1), i / max(float(h.sum()), 1)


def bare(a):
    a.set_xticks([]); a.set_yticks([])


def main():
    imgp, prefix, outdir = sys.argv[1], sys.argv[2], sys.argv[3]
    tag = sys.argv[4] if len(sys.argv) > 4 else "case"
    title = sys.argv[5] if len(sys.argv) > 5 else tag
    os.makedirs(outdir, exist_ok=True)
    img, _ = read_envi(imgp)
    sel, _ = read_envi(prefix + "_selected.bin")
    best, hint, llr, _cmap, kdn, kup = sel
    P = params(prefix)
    K = int(P.get("best_k", 0))
    # The three K values come from dedicated header fields.  Parsing them out
    # of the band names would be fragile: ENVI separates band names with
    # commas, so any name containing one shifts every later field.
    import re as _re
    hdr = open(os.path.splitext(prefix + "_selected.bin")[0] + ".hdr").read()
    def field(name, default):
        m = _re.search(r"^%s\s*=\s*(\d+)" % _re.escape(name), hdr, _re.M)
        return int(m.group(1)) if m else default
    K = field("kgc best k", K)
    step = field("kgc k step", 1)
    Kdn = field("kgc k below", max(step, K - step))
    Kup = field("kgc k above", K + step)
    rgb = composite(img)
    ar = rgb.shape[0] / float(rgb.shape[1])

    fig, ax = plt.subplots(3, 2, figsize=(11.0, 3 * (5.5 * ar + 0.5)))
    lo = [(hint, BIN, "hint mask as supplied\n%d px, %.1f%% of scene"
           % (hint.sum(), 100 * hint.mean())),
          (None, None, "input imagery\n%d bands, composite of the first three"
           % img.shape[0]),
          ("llr", None, "codeword log-likelihood ratio\nlog(p_hint / p_background), nats")]
    ri = []
    for m, lab in [(kdn, "K = %d" % Kdn), (best, "K = %d  (chosen)" % K),
                   (kup, "K = %d" % Kup)]:
        p, r = pr(m, hint)
        ri.append((m, SEL, "best class at %s\n%d px, precision %.3f, recall %.3f"
                   % (lab, m.sum(), p, r)))
    for row in range(3):
        for col, series in enumerate((lo, ri)):
            a = ax[row][col]
            m, cm, title = series[row]
            if isinstance(m, str) and m == "llr":
                v = np.percentile(np.abs(llr), 99)
                im = a.imshow(llr, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
                plt.colorbar(im, ax=a, fraction=0.035)
            elif m is None:
                a.imshow(rgb, interpolation="nearest")
            else:
                a.imshow(m, cmap=cm, vmin=0, vmax=1, interpolation="nearest")
            a.set_title(title, fontsize=9)
            bare(a)
    ax[0][0].set_ylabel("inputs and evidence", fontsize=10)
    ax[0][1].set_ylabel("neighbourhood size", fontsize=10)
    fig.suptitle(title, fontsize=12, y=0.999)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig.savefig(os.path.join(outdir, "plate_%s.png" % tag), dpi=130, bbox_inches="tight")
    plt.close(fig)

    g = np.zeros(best.shape)
    g[(best > .5) & (hint > .5)] = 1
    g[(best > .5) & (hint <= .5)] = 2
    g[(best <= .5) & (hint > .5)] = 3
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 5.5 * ar + 0.6))
    ax[0].imshow(g, cmap=AGR, vmin=0, vmax=3, interpolation="nearest")
    ax[0].set_title("agreement: cyan both, red class only, grey hint only", fontsize=9)
    v = np.percentile(np.abs(llr), 99)
    im = ax[1].imshow(llr, cmap="RdBu_r", vmin=-v, vmax=v, interpolation="nearest")
    ax[1].set_title("codeword log-likelihood ratio, nats", fontsize=9)
    plt.colorbar(im, ax=ax[1], fraction=0.035)
    for a in ax:
        bare(a)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "agreement_%s.png" % tag), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("plates written: %s/plate_%s.png and %s/agreement_%s.png"
          % (outdir, tag, outdir, tag))


if __name__ == "__main__":
    main()
