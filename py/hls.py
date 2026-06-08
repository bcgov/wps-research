#!/usr/bin/env python3
"""
hls.py - hierarchical, radix-style grouping of filenames.

Level 1  : file extension (equivalence class by extension).
Level 2+ : "common-prefix-until-the-next-difference" classes. Two names stay
           in the same class while their characters agree; the moment they
           differ, the class branches. This is exactly a radix / PATRICIA
           tree: each branch point in the tree is one hierarchy level.

Only files in the given directory (default: CWD) are considered - no recursion.
Exemplars (sample filenames + counts) are reported ONLY at the bottom level of
the (possibly truncated) hierarchy.

Usage:
    python hier_tree.py [DIR] [--levels N] [--exemplars K]

    --levels     max hierarchy depth, *including* the extension level (default 5)
    --exemplars  number of sample filenames to print per bottom class (default 1)
"""

import os
import argparse
from collections import OrderedDict


def divergence_index(strings, start):
    """First index >= start where the strings stop all agreeing.

    Agreement breaks either when two characters differ or when one string
    ends while others continue. If every string is identical from `start`
    onward, this returns the (shortest) length, i.e. "no divergence".
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
    """Split `strings` (all of which share strings[0][:start]) into the next
    level of classes. Returns a list of child-node dicts.

    Each child's `label` is the run of common text consumed to reach it,
    i.e. (text shared with siblings up to the split point) + (the one
    character that distinguishes this branch from its siblings).
    """
    div = divergence_index(strings, start)

    groups = OrderedDict()
    for s in strings:
        key = s[div] if div < len(s) else ''     # the discriminating character
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
        # Recurse only if the group can still split AND we have levels left.
        if len(grp) > 1 and (level + 1) < max_level:
            sub = build_subtree(grp, end, level + 1, max_level)
            if len(sub) > 1:                     # only branch if it really splits
                node['children'] = sub
                node['is_leaf'] = False
        nodes.append(node)
    return nodes


def print_tree(nodes, prefix, exemplars, stem_to_name):
    for i, node in enumerate(nodes):
        last = (i == len(nodes) - 1)
        connector = '\u2514\u2500 ' if last else '\u251c\u2500 '
        label = node['label'] if node['label'] != '' else '(exact prefix)'
        print(f"{prefix}{connector}{label}  \u00d7{node['count']}")
        child_prefix = prefix + ('    ' if last else '\u2502   ')

        if node['is_leaf']:
            shown = node['members'][:exemplars]
            for stem in shown:
                print(f"{child_prefix}\u2022 {stem_to_name[stem]}")
            extra = node['count'] - len(shown)
            if extra > 0:
                print(f"{child_prefix}\u2026 (+{extra} more)")
        else:
            print_tree(node['children'], child_prefix, exemplars, stem_to_name)


def main():
    ap = argparse.ArgumentParser(
        description="Hierarchical radix grouping of filenames in one directory.")
    ap.add_argument('directory', nargs='?', default='.',
                    help='directory to scan (default: current working directory)')
    ap.add_argument('--levels', type=int, default=5,
                    help='max hierarchy depth, including the extension level (default 5)')
    ap.add_argument('--exemplars', type=int, default=1,
                    help='sample filenames to show per bottom class (default 1)')
    args = ap.parse_args()

    if args.levels < 1:
        ap.error('--levels must be >= 1')
    if args.exemplars < 1:
        ap.error('--exemplars must be >= 1')

    try:
        entries = [e.name for e in os.scandir(args.directory) if e.is_file()]
    except FileNotFoundError:
        ap.error(f"directory not found: {args.directory}")

    if not entries:
        print(f"(no files in {os.path.abspath(args.directory)})")
        return

    # Level 1: group by extension. stem_to_name maps the extension-less stem
    # (used for the radix comparison) back to the real filename (for exemplars).
    ext_groups = OrderedDict()
    for name in entries:
        stem, ext = os.path.splitext(name)
        ext_groups.setdefault(ext, {})[stem] = name

    print(f"{os.path.abspath(args.directory)}  "
          f"[{len(entries)} files, {len(ext_groups)} extensions, levels={args.levels}]")
    print()

    for ext in sorted(ext_groups):
        stem_to_name = ext_groups[ext]
        ext_label = ext if ext else '(no extension)'
        print(f"{ext_label}  \u00d7{len(stem_to_name)}")

        stems = list(stem_to_name)
        if args.levels < 2:
            # The extension itself is the bottom level.
            shown = sorted(stems)[:args.exemplars]
            for stem in shown:
                print(f"    \u2022 {stem_to_name[stem]}")
            extra = len(stems) - len(shown)
            if extra > 0:
                print(f"    \u2026 (+{extra} more)")
        else:
            children = build_subtree(stems, 0, 1, args.levels)
            print_tree(children, '', args.exemplars, stem_to_name)
        print()


if __name__ == '__main__':
    main()


