#!/usr/bin/env python3
"""
hier_tree.py - hierarchical, radix-style grouping of filenames.

Level 1  : file extension (equivalence class by extension).
Level 2+ : "common-prefix-until-the-next-difference" classes. Two names stay
           in the same class while their characters agree; the moment they
           differ, the class branches. This is a radix / PATRICIA tree: each
           branch point is one hierarchy level.

Each printed line shows the *cumulative* stem so far (the real filename prefix
up to that branch point), so the full pattern is visible at a glance.

Only files in the given directory (default: CWD) are considered - no recursion.
Exemplars (sample filenames) are reported only at the bottom level of the
(possibly truncated) hierarchy; a trailing "..." marks that more files share
that class (the count is already shown on the class line above).

Recall / archive: when a directory holds more than --cache-threshold files
(default 1500), the computed hierarchy is written to ".hier_tree.hls" in that
directory. A later run reuses it (same files + same --levels) instead of
recomputing. Use --refresh to recompute, --no-cache to never write.

Usage:
    python hier_tree.py [DIR] [--levels N] [--exemplars K]
                        [--refresh] [--no-cache] [--cache-threshold N]
"""

import os
import json
import hashlib
import argparse
from collections import OrderedDict

CACHE_NAME = '.hier_tree.hls'
CACHE_FORMAT = 'hls/1'


def divergence_index(strings, start):
    """First index >= start where the strings stop all agreeing.

    Agreement breaks when two characters differ or when one string ends while
    others continue. If every string is identical from `start` onward, returns
    the (shortest) length, i.e. "no divergence".
    """
    i = start
    while True:
        ref = None
        for s in strings:
            if i >= len(s):
                return i              # a string ends here -> divergence
            if ref is None:
                ref = s[i]
            elif s[i] != ref:
                return i              # characters differ -> divergence
        i += 1


def build_subtree(strings, start, level, max_level):
    """Split `strings` (all sharing strings[0][:start]) into the next level of
    classes. Returns a list of child-node dicts. Each `label` is the run of
    common text consumed to reach that child (siblings' shared run + the one
    distinguishing character)."""
    div = divergence_index(strings, start)

    groups = OrderedDict()
    for s in strings:
        key = s[div] if div < len(s) else ''     # discriminating character
        groups.setdefault(key, []).append(s)

    nodes = []
    for key in sorted(groups):
        grp = groups[key]
        end = div + 1 if key != '' else div
        node = {
            'label': grp[0][start:end],
            'count': len(grp),
            'members': sorted(grp),
            'children': [],
            'is_leaf': True,
        }
        if len(grp) > 1 and (level + 1) < max_level:
            sub = build_subtree(grp, end, level + 1, max_level)
            if len(sub) > 1:                     # only branch if it really splits
                node['children'] = sub
                node['is_leaf'] = False
        nodes.append(node)
    return nodes


INDENT = '  '  # single systematic indent applied to everything under an extension


def print_tree(nodes, acc, exemplars, stem_to_name):
    """Print nodes flush at one indent level (no per-depth indentation, no
    bullets). `acc` is the cumulative stem inherited from ancestors, so each
    line prints acc+label and the full prefix pattern lines up legibly."""
    for node in nodes:
        cum = acc + node['label']
        shown_label = cum if cum != '' else '(empty stem)'
        print(f"{INDENT}{shown_label}  \u00d7{node['count']}")

        if node['is_leaf']:
            shown = node['members'][:exemplars]
            more = node['count'] > len(shown)
            for k, stem in enumerate(shown):
                trail = ' \u2026' if (more and k == len(shown) - 1) else ''
                print(f"{INDENT}{stem_to_name[stem]}{trail}")
        else:
            print_tree(node['children'], cum, exemplars, stem_to_name)


# ----------------------------- caching ----------------------------------- #

def _prune_for_cache(nodes):
    """Return a copy of nodes with `members` kept only on leaves (smaller file)."""
    out = []
    for n in nodes:
        c = {'label': n['label'], 'count': n['count'], 'is_leaf': n['is_leaf']}
        if n['is_leaf']:
            c['members'] = n['members']
            c['children'] = []
        else:
            c['children'] = _prune_for_cache(n['children'])
        out.append(c)
    return out


def fingerprint(filenames):
    h = hashlib.sha256()
    h.update('\n'.join(sorted(filenames)).encode('utf-8', 'surrogatepass'))
    return h.hexdigest()


def load_cache(path, fp, levels):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if (data.get('format') == CACHE_FORMAT
            and data.get('fingerprint') == fp
            and data.get('levels') == levels):
        return data
    return None


def write_cache(path, fp, levels, abspath, file_count, extensions):
    payload = {
        'format': CACHE_FORMAT,
        'directory': abspath,
        'levels': levels,
        'file_count': file_count,
        'fingerprint': fp,
        'extensions': [
            {'ext': e['ext'], 'count': e['count'],
             'stem_to_name': e['stem_to_name'],
             'children': _prune_for_cache(e['children'])}
            for e in extensions
        ],
    }
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, separators=(',', ':'))
        return True
    except OSError:
        return False


# ------------------------------- main ------------------------------------ #

def main():
    ap = argparse.ArgumentParser(
        description="Hierarchical radix grouping of filenames in one directory.")
    ap.add_argument('directory', nargs='?', default='.',
                    help='directory to scan (default: current working directory)')
    ap.add_argument('--levels', type=int, default=5,
                    help='max hierarchy depth, including the extension level (default 5)')
    ap.add_argument('--exemplars', type=int, default=1,
                    help='sample filenames to show per bottom class (default 1)')
    ap.add_argument('--refresh', action='store_true',
                    help='ignore any existing cache and recompute')
    ap.add_argument('--no-cache', action='store_true',
                    help='never read or write the .hier_tree.hls cache')
    ap.add_argument('--cache-threshold', type=int, default=1500,
                    help='write a cache only when file count exceeds this (default 1500)')
    args = ap.parse_args()

    if args.levels < 1:
        ap.error('--levels must be >= 1')
    if args.exemplars < 1:
        ap.error('--exemplars must be >= 1')

    abspath = os.path.abspath(args.directory)
    cache_path = os.path.join(args.directory, CACHE_NAME)

    try:
        entries = [e.name for e in os.scandir(args.directory)
                   if e.is_file() and e.name != CACHE_NAME]
    except FileNotFoundError:
        ap.error(f"directory not found: {args.directory}")

    if not entries:
        print(f"(no files in {abspath})")
        return

    fp = fingerprint(entries)
    use_cache = not args.no_cache

    # ---- try recall ----
    cached = None
    if use_cache and not args.refresh and os.path.exists(cache_path):
        cached = load_cache(cache_path, fp, args.levels)

    if cached is not None:
        print(f"{abspath}  [{cached['file_count']} files, "
              f"{len(cached['extensions'])} extensions, levels={args.levels}] "
              f"(recalled from {CACHE_NAME})")
        print()
        for e in cached['extensions']:
            ext_label = e['ext'] if e['ext'] else '(no extension)'
            print(f"{ext_label}  \u00d7{e['count']}")
            if e['children']:
                print_tree(e['children'], '', args.exemplars, e['stem_to_name'])
            else:  # levels==1 case: list exemplars under the extension
                stems = sorted(e['stem_to_name'])
                shown = stems[:args.exemplars]
                more = len(stems) > len(shown)
                for k, stem in enumerate(shown):
                    trail = ' \u2026' if (more and k == len(shown) - 1) else ''
                    print(f"{INDENT}{e['stem_to_name'][stem]}{trail}")
            print()
        return

    # ---- compute fresh ----
    ext_groups = OrderedDict()
    for name in entries:
        stem, ext = os.path.splitext(name)
        ext_groups.setdefault(ext, {})[stem] = name

    extensions = []  # ordered records for printing + caching
    for ext in sorted(ext_groups):
        stem_to_name = ext_groups[ext]
        stems = list(stem_to_name)
        if args.levels < 2:
            children = []          # extension itself is the bottom level
        else:
            children = build_subtree(stems, 0, 1, args.levels)
        extensions.append({'ext': ext, 'count': len(stems),
                           'stem_to_name': stem_to_name, 'children': children})

    print(f"{abspath}  [{len(entries)} files, {len(extensions)} extensions, "
          f"levels={args.levels}]")
    print()
    for e in extensions:
        ext_label = e['ext'] if e['ext'] else '(no extension)'
        print(f"{ext_label}  \u00d7{e['count']}")
        if e['children']:
            print_tree(e['children'], '', args.exemplars, e['stem_to_name'])
        else:
            stems = sorted(e['stem_to_name'])
            shown = stems[:args.exemplars]
            more = len(stems) > len(shown)
            for k, stem in enumerate(shown):
                trail = ' \u2026' if (more and k == len(shown) - 1) else ''
                print(f"{INDENT}{e['stem_to_name'][stem]}{trail}")
        print()

    # ---- archive if large ----
    if use_cache and len(entries) > args.cache_threshold:
        ok = write_cache(cache_path, fp, args.levels, abspath, len(entries), extensions)
        if ok:
            print(f"[archived {len(entries)} files to {cache_path}]")
        else:
            print(f"[could not write cache to {cache_path} - directory not writable?]")


if __name__ == '__main__':
    main()
