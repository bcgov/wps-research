# kgc — kNN density clustering with hint-guided choice of K

    ./run_all.sh

builds the program and runs all three example scenes against their BCWS
fire-perimeter hints, producing a six-panel plate, an agreement figure and a
blink GIF for each. Nothing needs to be supplied. To run one:

    ./run_all.sh c10467

POSIX shell — bash, dash or zsh, Linux or macOS. Override the toolchain with
`CXX=g++-13 ./run_all.sh` or `PYTHON=python3.11 ./run_all.sh` if needed.

Knobs, all optional:

| variable | effect |
|---|---|
| `NSKIP=8` | fix the decimation stride outright |
| `BUDGET=60000` | ask for that many retained points; the stride follows |
| `KMAX=5000` | neighbours held per point — the memory knob |
| `KSTEP=10` | the sweep visits K = kstep, 2·kstep, … |
| `THREADS=64` | pin the worker count |

`n_skip` is automatic by default: `ceil(n_dedup / BUDGET)` with `BUDGET` at
20000. More retained points means finer classes and a larger neighbour table,
which grows as `n_points × k_max`.

| path | what it is |
|---|---|
| `kgc.h`, `kgc.cpp` | the implementation — C++11 and threads, nothing else |
| `run_all.sh` | builds and runs all three examples |
| `plates.py`, `gifs.py`, `figures.py` | figures (need python3, numpy, matplotlib, pillow) |
| `IDEAS.md` | removed features and untested ideas, with the evidence |
| `data/` | three scenes, each with a perimeter hint and a spectral-index hint |
| `latex/` | the report for the first scene; `make -C latex` rebuilds the PDF |

`run_all.sh` writes the figures to `plates/` and `gifs/` at the top level, and
writes `latex/results_<tag>.tex` so the report shows the numbers from your run
rather than numbers fixed when it was written. The report finds the figures via
`\graphicspath{{../plates/}{plates/}}`, so build it with `make -C latex` after
a run.

## Method

Each band is scaled to [0,1]; the whole image is deduplicated on exact equality
of the band vectors; every `n_skip`-th surviving record is retained. For each swept
neighbourhood size K the density is `rho_K(i) = -S_K(i)`, the sum of distances
to the K nearest other retained points, and each point ascends to the densest
point of its K-neighbourhood until it reaches a fixed point. Every class of
every K is scored against the hint by mutual information; the best is returned
with the corresponding classes at K-1 and K+1.

## Threading

Every stage runs one POSIX worker per CPU. Work is handed out as integer job
indices from a single mutex-guarded counter: one job per row of the neighbour
table, per block of pixels, and per value of K in the sweep. Workers claim jobs
in order but finish out of order, and none waits for another.

Read-only data shared by the workers — the neighbour table, the per-point pixel
and hint counts, the sizes — lives at file scope, so every worker reads one
definite object. The mutex guards only what the workers write: the results
vector, the running best, and the stop flag. Each worker clusters on its own
scratch buffers and touches shared state once, at the end of a level.

Only a handful of scalars are kept per level, so the sweep's memory is the same
whichever K a worker is on. Once the best K is known the class maps for it and
its two neighbouring levels are recomputed from the neighbour table, which is
still in memory.

**Early stopping is off by default.** `--patience <n>` raises a stop flag once
that many completed levels above the current best have failed to improve on it,
and every worker drains after its current job. It is off because the score
against the hint is not unimodal in K: measured on C50929, a patience of 40
stopped at K=535 with mutual information 0.247 and recall 0.567, missing K=875
at 0.451 and recall 0.844. See `IDEAS.md`.

The answer does not depend on the worker count: ties in the score are broken in
favour of the smaller K, so the same input gives the same output at any degree
of parallelism. Verified identical at 1, 4, 16 and 64 workers.

## Cost and caching

The neighbour table is `n_points x k_max` entries of 12 bytes; at the default
`k_max` of 10000 that is a little over 2 GB of RAM and a cache file of the same
size per scene, so `KMAX=5000 ./run_all.sh` is worth knowing. The sweep visits K = `k_step`,
2·`k_step`, … (`--kstep`, default 5, which is also the first K) and stops early,
so it rarely reaches `k_max`. The deduplication and the neighbour table are memoised beside each image,
keyed on a checksum of the binary, so a second run of the same scene skips both.
`rm data/*.kgc_*` reclaims the cache.
`--no-cache` forces recomputation. `./kgc --help` lists every option.

## Only three of six cases are run

Each scene ships with two hints. `run_all.sh` uses the fire-perimeter hint for
each. The spectral-index hint is included and can be substituted:

    ./kgc data/C50929_mrap_prepost.bin \
          --hint data/C50929_hint_redwins_post.bin \
          --out out/c50929_rw
