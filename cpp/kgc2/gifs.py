#!/usr/bin/env python3
"""Blink-comparator GIFs: input imagery alternating with the selected class.

Two frames per case: the input imagery, then the selected class as a plain
black-and-white mask. Flipping between them makes coverage and over-reach
obvious in a way two side-by-side panels do not, and keeping the second frame
free of imagery avoids any doubt about which pixels are in the class.

    gifs.py <outdir>
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

from figures import read_envi, composite

# driven from the command line: gifs.py <image.bin> <prefix> <outdir> <tag> [title]


SCALE_TO = 900    # target width in pixels
MS = 850          # milliseconds per frame


def label(img, text, colour):
    """Burn a caption strip along the bottom of a PIL image."""
    d = ImageDraw.Draw(img)
    w, h = img.size
    d.rectangle([0, h - 22, w, h], fill=(10, 12, 16))
    d.text((8, h - 16), text, fill=colour)
    return img


def to_img(arr, scale):
    im = Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))
    if scale != 1:
        im = im.resize((im.width * scale, im.height * scale), Image.NEAREST)
    return im


def main(argv):
    imgp, pre, outdir = argv[1], argv[2], argv[3]
    tag = argv[4] if len(argv) > 4 else "case"
    title = argv[5] if len(argv) > 5 else tag
    os.makedirs(outdir, exist_ok=True)
    for _once in (0,):
        img, _ = read_envi(imgp)
        sel, _ = read_envi(pre + "_selected.bin")
        mask, hint = sel[0], sel[1]
        rgb = composite(img)

        scale = max(1, int(round(SCALE_TO / rgb.shape[1])))
        m = (mask > 0.5)[:, :, None]
        inter = float(((mask > .5) & (hint > .5)).sum())
        prec = inter / max(float(mask.sum()), 1)
        rec = inter / max(float(hint.sum()), 1)

        f1 = label(to_img(rgb, scale), "%s  -  input imagery" % title,
                   (170, 190, 210))
        bw = np.repeat(m.astype(np.float32), 3, axis=2)
        f2 = label(to_img(bw, scale),
                   "selected class  %d px   precision %.3f   recall %.3f"
                   % (mask.sum(), prec, rec),
                   (120, 205, 240))

        path = os.path.join(outdir, "blink_%s.gif" % tag)
        f1.save(path, save_all=True, append_images=[f2], duration=MS, loop=0,
                optimize=True)
        print("%-44s %d x %d" % (path, f1.width, f1.height))


if __name__ == "__main__":
    main(sys.argv)
