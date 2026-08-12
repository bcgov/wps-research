"""Which bands of an AOI stack reach the classifier and the export.

One implementation, used by both ``mapping_cmd`` (what the ML stage
receives) and the imagery export handler. They were separate before and
drifted -- the export lost the difference bands for a while without
anything noticing. Sharing the selection makes divergence impossible
rather than merely unlikely.

The stack is laid out ``1..N`` pre, ``N+1..2N`` post, ``2N+1..3N``
anomaly (see ``aoi_stack``), with band names like::

    pre 20260501 20m: B12 2190nm MRAP
    pst 20260803 20m: B11 1610nm MRAP
    anomaly: B9 945nm MRAP (post-pre)/(post+pre)

Nothing here assumes a particular band count or that any specific
spectral band is present. B8 is usually in the MRAP layers but may not
be in future; an exclusion that matches nothing is simply a no-op, and
the number of bands per era is derived from the names rather than
hard-coded.
"""

import re
import sys

# Era prefixes as written by aoi_stack.
_ERA_PRE = 'pre'
_ERA_POST = 'pst'
_ERA_ANOM = 'anomaly'


def band_era(name: str) -> str:
    """'pre', 'pst', 'anomaly', or '' when the name is unrecognised."""
    n = (name or '').strip().lower()
    if n.startswith(_ERA_ANOM):
        return _ERA_ANOM
    if n.startswith(_ERA_PRE):
        return _ERA_PRE
    if n.startswith(_ERA_POST):
        return _ERA_POST
    return ''


def is_b8(name: str) -> bool:
    """True for B8 and B8A, in any era.

    Word-boundary matched so 'B8' cannot match 'B8A' by prefix nor
    'B12' by substring, while still catching both real variants.
    """
    return bool(re.search(r'\bb8a?\b', (name or '').lower()))


def describe_bands(names) -> dict:
    """Counts per era and which spectral codes are present."""
    out = {'total': len(names or []), 'pre': 0, 'pst': 0, 'anomaly': 0,
           'unknown': 0, 'codes': [], 'has_b8': False}
    codes = []
    for nm in names or []:
        era = band_era(nm)
        if era == _ERA_PRE:
            out['pre'] += 1
        elif era == _ERA_POST:
            out['pst'] += 1
        elif era == _ERA_ANOM:
            out['anomaly'] += 1
        else:
            out['unknown'] += 1
        m = re.search(r'\b(b\d+a?)\b', (nm or '').lower())
        if m and m.group(1) not in codes:
            codes.append(m.group(1))
        if is_b8(nm):
            out['has_b8'] = True
    out['codes'] = codes
    return out


def select_bands(names, exclude_b8=False, exclude_pre=False,
                 exclude_diff=False, diff_only=False, override=None,
                 log=None) -> dict:
    """Choose the bands to keep.

    The three exclusions are independent and compose: B8 is dropped
    from every era it appears in, and the era exclusions remove whole
    groups regardless of which spectral bands they contain.

    Guarantees the caller can rely on:

    * An exclusion that matches nothing is a no-op, so a stack without
      B8 behaves exactly as one with B8 excluded.
    * The result is never empty. If a combination would remove every
      band, the era exclusions are backed off (they are the coarse
      ones) and, failing that, nothing is excluded -- with a loud
      message. A wider stack is recoverable; an empty one fails the
      run.

    Returns a dict with ``keep`` (0-based indices, ascending),
    ``dropped``, ``reasons`` (per dropped index), ``summary`` and
    ``degraded`` (True when an exclusion had to be backed off).
    """
    names = list(names or [])
    n = len(names)

    # An explicit override wins outright.
    #
    # The checkboxes describe RULES; the override is a hand-picked
    # list, and a hand-picked list can be any shape at all -- including
    # combinations no rule could express. Trying to reconcile the two
    # would either silently discard the user's picks or invent a rule
    # they did not ask for, so the override is simply used as given and
    # the rules resume the moment a checkbox is touched (which clears
    # it).
    if override:
        keep = sorted({int(i) for i in override
                       if isinstance(i, (int, float))
                       and 0 <= int(i) < n})
        if keep:
            dk = describe_bands([names[i] for i in keep])
            summary = (f'{len(keep)}/{n} band(s) kept by a CUSTOM '
                       f'selection (pre={dk["pre"]} post={dk["pst"]} '
                       f'anomaly={dk["anomaly"]}); the exclusion '
                       f'checkboxes are not applied')
            msg = f'[bands] {summary}'
            sys.stderr.write(msg + '\n')
            if log:
                try:
                    log('  ' + msg)
                except Exception:
                    pass
            return {'keep': keep, 'dropped': [], 'reasons': {},
                    'summary': summary, 'degraded': False,
                    'custom': True,
                    'describe_all': describe_bands(names),
                    'describe_keep': dk}

    # 'Diff only' is a stronger statement than the era exclusions: it
    # names what to KEEP rather than what to drop, so it implies
    # excluding pre AND post and contradicts excluding diff. Resolving
    # that here means every caller gets the same answer, rather than
    # each one re-deriving the implications.
    if diff_only:
        exclude_pre = True
        exclude_diff = False

    def _apply(x_b8, x_pre, x_diff, only_diff=False):
        keep, reasons = [], {}
        for i, nm in enumerate(names):
            era = band_era(nm)
            if x_b8 and is_b8(nm):
                reasons[i] = 'B8'
                continue
            if only_diff and era != _ERA_ANOM:
                reasons[i] = 'not a difference band'
                continue
            if x_pre and era == _ERA_PRE:
                reasons[i] = 'pre-fire'
                continue
            if x_diff and era == _ERA_ANOM:
                reasons[i] = 'difference'
                continue
            keep.append(i)
        return keep, reasons

    keep, reasons = _apply(exclude_b8, exclude_pre, exclude_diff,
                           diff_only)
    degraded = False
    note = ''

    if not keep and n:
        # Back off the era exclusions first: they remove whole groups,
        # so they are the likeliest cause of an empty selection.
        keep, reasons = _apply(exclude_b8, False, False, False)
        degraded = True
        note = ('every band would have been excluded; kept all eras '
                'and applied only the B8 rule')
    if not keep and n:
        keep = list(range(n))
        reasons = {}
        degraded = True
        note = ('every band would have been excluded; kept the full '
                'stack')

    desc_all = describe_bands(names)
    desc_keep = describe_bands([names[i] for i in keep])

    asked = [k for k, v in (('B8', exclude_b8),
                            ('pre-fire', exclude_pre),
                            ('difference', exclude_diff),
                            ('all but difference', diff_only)) if v]
    ineffective = []
    if exclude_b8 and not desc_all['has_b8']:
        ineffective.append('B8 (not present in this stack)')
    if exclude_pre and desc_all['pre'] == 0:
        ineffective.append('pre-fire (no pre bands in this stack)')
    if exclude_diff and desc_all['anomaly'] == 0:
        ineffective.append('difference (no anomaly bands in this stack)')
    if diff_only and desc_all['anomaly'] == 0:
        ineffective.append('diff-only (this stack has no anomaly bands, '
                           'so nothing could be kept)')

    summary = (
        f'{len(keep)}/{n} band(s) kept '
        f'(pre={desc_keep["pre"]} post={desc_keep["pst"]} '
        f'anomaly={desc_keep["anomaly"]})'
        + (f'; excluding {", ".join(asked)}' if asked else '; no exclusions')
        + (f'; no effect: {", ".join(ineffective)}' if ineffective else '')
        + (f'; DEGRADED: {note}' if degraded else ''))

    msg = f'[bands] {summary}'
    sys.stderr.write(msg + '\n')
    if log:
        try:
            log('  ' + msg)
        except Exception:
            pass

    return {'keep': keep, 'dropped': sorted(reasons), 'reasons': reasons,
            'summary': summary, 'degraded': degraded,
            'describe_all': desc_all, 'describe_keep': desc_keep}


def bands_governed_by(names, flag: str) -> list:
    """Indices the given exclusion rule controls.

    Each checkbox owns a set of bands. Toggling it should add or remove
    exactly that set, leaving every other band as the user left it --
    which is what makes an incremental change predictable rather than a
    recomputation that quietly discards hand-picked choices.
    """
    out = []
    for i, nm in enumerate(names or []):
        era = band_era(nm)
        if flag == 'exclude_b8' and is_b8(nm):
            out.append(i)
        elif flag == 'exclude_pre_fire' and era == _ERA_PRE:
            out.append(i)
        elif flag == 'exclude_diff' and era == _ERA_ANOM:
            out.append(i)
        elif flag == 'diff_only' and era != _ERA_ANOM:
            # 'Diff only' governs everything that is NOT a difference
            # band: turning it on removes them, turning it off puts
            # them back.
            out.append(i)
    return out


def apply_flag_change(names, current, flag: str, turned_on: bool,
                      log=None) -> list:
    """Apply ONE rule change to an existing selection.

    *current* is the selection in force; the return value is that
    selection with the bands this flag governs removed (when it was
    switched on) or restored (when switched off). Nothing else moves.

    This is deliberately not a recomputation from all the flags: doing
    that would undo a custom selection every time any box was clicked,
    which is precisely the behaviour being avoided.
    """
    cur = set(int(i) for i in (current or []))
    gov = set(bands_governed_by(names, flag))
    if not gov:
        return sorted(cur)
    before = len(cur)
    if turned_on:
        cur -= gov
    else:
        cur |= gov
    keep = sorted(cur)
    msg = (f'[bands] {flag} {"on" if turned_on else "off"}: '
           f'{"removed" if turned_on else "restored"} '
           f'{len(gov)} band(s) it governs; selection {before} -> '
           f'{len(keep)}')
    sys.stderr.write(msg + '\n')
    if log:
        try:
            log('  ' + msg)
        except Exception:
            pass
    return keep


def selection_tag(exclude_b8=False, exclude_pre=False,
                  exclude_diff=False, diff_only=False,
                  override=None) -> str:
    """Short, stable tag naming a selection, for cache filenames.

    A reduced stack is cached on disk; without the combination in the
    filename a stack built for one set of exclusions would be reused
    for another, silently feeding the classifier the wrong bands.
    """
    if override:
        # Hash the picks: a custom stack must never be served from a
        # cache built for a different selection, and the list itself is
        # too long for a filename.
        import hashlib
        key = ','.join(str(int(i)) for i in sorted(override))
        return 'custom' + hashlib.sha1(
            key.encode('utf-8')).hexdigest()[:8]
    parts = []
    if diff_only:
        # Named separately: a diff-only stack is not the same file as
        # one built with the era exclusions, and sharing a cache name
        # would feed the classifier the wrong bands.
        parts.append('diffonly')
    if exclude_b8:
        parts.append('nob8')
    if exclude_pre and not diff_only:
        parts.append('nopre')
    if exclude_diff and not diff_only:
        parts.append('nodiff')
    return '_'.join(parts) if parts else 'all'


def remap_indices(spec: str, keep, log=None) -> str:
    """Translate 1-based band indices onto the reduced stack.

    ``embed_bands`` holds absolute indices into whatever stack the CLI
    receives. Once bands are removed those indices name different bands
    -- and any index past the new count makes the embedding step fail
    outright -- so they must be remapped, not passed through.

    Indices naming a dropped band are removed. An empty result means
    "all bands", which is the CLI's own default and the right fallback.
    """
    try:
        old_to_new = {orig: j + 1 for j, orig in enumerate(keep)}
        out, dropped = [], []
        for tok in str(spec or '').split(','):
            tok = tok.strip()
            if not tok:
                continue
            try:
                one = int(tok)
            except ValueError:
                continue
            new = old_to_new.get(one - 1)
            if new is None:
                dropped.append(one)
            else:
                out.append(new)
        res = ','.join(str(i) for i in out)
        msg = (f'[bands] embed_bands "{spec}" -> "{res or "all"}"'
               + (f' (dropped {dropped}: excluded band(s))'
                  if dropped else ''))
        sys.stderr.write(msg + '\n')
        if log:
            try:
                log('  ' + msg)
            except Exception:
                pass
        return res
    except Exception as exc:
        sys.stderr.write(
            f'[bands] embed_bands remap failed ({exc}); letting the '
            f'CLI use all bands\n')
        return ''
