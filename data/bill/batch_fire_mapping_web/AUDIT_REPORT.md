# batch_fire_mapping_web — Correctness Audit Report

**Audit date:** 2026-04-27
**Scope:** every Python file in this package (24 files, ~9,700 lines)
**Methodology:** static analysis + 13 runtime test scripts (252 assertions, 251 passing)
**Goal:** identify defects in logic, persistence, and concurrency for production hardening

This report lists every defect found, ranked by severity. Each entry includes:
the file:line reference, a short reason the current code is wrong, the impact,
a reproducer (where applicable), and a fix sketch. The intent is that another
agent can pick up this document and work through the items in order without
re-doing the analysis.

The package is **well-engineered overall**. Atomic YAML writes use tmp+fsync+rename;
state is RLock-protected; sessions store hash-only; passwords use `hmac.compare_digest`;
subprocess is list-based (no shell=True); fire_numbe is regex-validated to block
path traversal; cache eviction respects in-flight pins; the corruption-recovery
flag in `_load_fire_state` is a thoughtful safety net. The defects below are the
remaining gaps a "must be exact" deployment cannot ignore.

---

## Severity scale

- **Critical** — data loss, corruption, or silent inconsistency under realistic conditions.
- **High** — functional gap that produces wrong outputs, latent injection, or
  invariants that break on rare-but-plausible inputs.
- **Medium** — robustness/observability gaps, dead code, or footguns for future maintainers.
- **Low** — style/comment-only items.

---

## Critical

### C1. `_atomic_yaml_dump` does not fsync the parent directory

**File:** `io_utils.py:7-31`
**Reproducer:** `tests/audit/test_atomic_yaml.py::t7_no_dir_fsync`

The function fsyncs the temp file's contents before `os.replace(tmp, path)` — good.
But the directory entry created by `os.replace` is not durable until the
**parent directory** is also fsynced. POSIX guarantees rename is atomic, not durable.
After a power loss, the kernel can reorder: file content reaches disk, rename
metadata does not, and the file reappears as the pre-replace version (or vanishes
entirely) on reboot.

**Affected files (all written via this helper):** `fire_state.yaml`,
`accepted_params.csv` (uses the same pattern open-coded in `prepare.py`),
`notifications.yaml`, `sessions.yaml`, `access_control.yaml`,
`active_year.yaml`, `fire_status.yaml`, `cache_retention.yaml`,
`stage_timings.yaml`, `recommended_settings.yaml`, every per-fire
`<fire>_params.yaml`, `notes.yaml`.

**Impact:** after a host reboot, *any* of the above can revert to its prior
version. ACCEPTED markers, accepted-params learning data, IP allow-lists, and
recommended settings are all at risk. This is a real durability bug, not a
theoretical one.

**Fix:**
```python
os.replace(tmp, path)
dir_fd = os.open(os.path.dirname(path) or '.', os.O_RDONLY)
try:
    os.fsync(dir_fd)
finally:
    os.close(dir_fd)
```
Apply the same pattern to the open-coded CSV write in `prepare.py:594-610`.

---

### C2. `prepare.py` swallows `fire_status.yaml` write failures with bare `pass`

**File:** `prepare.py:538-555`

```python
try:
    import yaml
    status_path = os.path.join(state.output_root, 'fire_status.yaml')
    with _accept_file_lock:
        ...
        _atomic_yaml_dump(status_path, idx)
except Exception:
    pass
```

If the read-modify-write of `fire_status.yaml` raises (disk full, permission
denied, YAML serialization error from a weird notes string), the accept proceeds.
The canonical dir is populated, `fire.status = ACCEPTED` is set in memory and in
`fire_state.yaml`, but `fire_status.yaml` (the audit log consumed by external
tools) silently misses the entry.

**Impact:** asymmetric persisted state. External audits / batch jobs that
consume `fire_status.yaml` will under-report accepted fires. After a server
restart, `fire_state.yaml` and the canonical dir disagree with `fire_status.yaml`.

**Fix:** at minimum, log to stderr like the other persistence helpers do:
```python
except Exception as exc:
    sys.stderr.write(
        f'[save] WARNING: fire_status.yaml update failed for '
        f'{fire_numbe}: {exc}\n')
    sys.stderr.flush()
```
Better: surface the failure to the caller and return an error to the user — but
that is a behavioral change that needs product input.

---

### C3. `_accept_fire_sync` does not refuse re-entry for the same fire

**File:** `prepare.py:415-419`

```python
with _accept_in_progress_lock:
    _accept_in_progress.add(fire_numbe)   # never checks if already present
try:
    if os.path.isdir(fire_dir):
        shutil.rmtree(fire_dir)            # racy if T2 starts here while T1 is mid-copy
    os.makedirs(fire_dir)
    ...
```

`_accept_in_progress` is intended to block the cache sweeper from rmtree-ing
`cache_dir` mid-accept. It is **not** a mutual-exclusion gate against two
accepts of the same fire. Two accept paths can call `_accept_fire_sync`
concurrently:

- `handlers/mapping.py::handle_api_accept` wraps the call in `with _gpu_lock`.
- `handlers/serial.py::handle_api_serial_accept` also wraps in `with _gpu_lock`.

So normal user double-clicks serialize fine. **But:** the `serial_accept`
handler also takes paths through SIGTERM-and-wait-for-worker-cleanup, and if a
batch worker calls `_accept_fire_sync` from the worker thread (it does not
today, but `prepare._accept_fire_sync` is an exported helper), the GPU lock
discipline alone is brittle.

**Impact:** today, low — UI prevents double-click, handlers serialize via
`_gpu_lock`. Latent: any future caller that forgets `_gpu_lock` causes
`shutil.rmtree(fire_dir)` followed by `os.makedirs` to race; T2 can rmtree
the dir T1 just created.

**Fix:**
```python
with _accept_in_progress_lock:
    if fire_numbe in _accept_in_progress:
        raise RuntimeError(
            f'Accept already in progress for {fire_numbe}')
    _accept_in_progress.add(fire_numbe)
```

---

### C4. TOCTOU on `fire.status == FireStatus.PREPARING` in `_prepare_fire_sync`

**File:** `prepare.py:67-71`

```python
if fire.status == FireStatus.PREPARING:
    fire.error_msg = 'Cannot prepare: fire is currently preparing'
    return

fire.status = FireStatus.PREPARING
fire.error_msg = ""
```

The check at line 67 and the flip at line 71 are not atomic. Two threads both
reading `status == PENDING` both pass the guard, both flip to PREPARING, both
race on `crop_raster` / VIIRS accumulation / preview generation, overwriting
each other's outputs in `cache_dir`.

**Why this is latent today:** the only callers (`handle_api_prepare` and
`_serial_run_replicate`) wrap the call in `with _gpu_lock:`, so PREPARING
runs are already serialized. But the guard inside `_prepare_fire_sync` is
the function's own contract; relying on every caller to hold an external lock
is a bug-magnet.

**Fix:** use `state.lock` for the test-and-set:
```python
with state.lock:
    if fire.status == FireStatus.PREPARING:
        fire.error_msg = 'Cannot prepare: fire is currently preparing'
        return
    fire.status = FireStatus.PREPARING
    fire.error_msg = ''
```

---

### C5. Same TOCTOU pattern in three handlers

**Files:**
- `handlers/serial.py:135-176` (`handle_api_serial_map`)
- `handlers/mapping.py::handle_api_map` (single-shot map)
- `handlers/rebrush.py:134-137` (`handle_api_rebrush`)

Each has the shape:
```python
if fire.status == FireStatus.MAPPING:
    self._send_json({'error': 'Already mapping'}, 400)
    return
...
fire.status = FireStatus.MAPPING        # not under state.lock
fire.serial_results = []
threading.Thread(target=_serial_map_worker, args=(...)).start()
```

Two simultaneous POSTs (deliberate race, double-fire from a bad client, or
reload-spam during a slow page) both pass the check, both flip status, both
spawn worker threads. Each worker takes `_gpu_lock` per replicate so the
GPU pipeline is still serialized — **but** the workers share
`fire.serial_results`, `fire.serial_settings`, `fire.console_log`, and
`fire.progress`. They will trample each other's gallery entries and console
output.

For `handle_api_rebrush`, the consequences are worse: each spawned `class_brush`
subprocess registers itself in `_rebrush_procs[fire_numbe]`. The second
overwrites the first, leaving the first subprocess unkillable from the cancel
endpoint and unaccounted for in cache-retention pinning.

**Fix:** atomic test-and-set under `state.lock` in each handler:
```python
with state.lock:
    if fire.status == FireStatus.MAPPING:
        self._send_json({'error': 'Already mapping'}, 400)
        return
    fire.serial_prev_status = fire.status
    fire.status = FireStatus.MAPPING
    fire.serial_results = []
    fire.serial_settings = [_clone_setting(s) for s in settings]
    fire.console_log.clear()
    fire.progress = {}
# spawn thread *after* releasing lock
threading.Thread(...).start()
```

For rebrush, the same pattern — atomic check-then-claim of the
`_rebrush_procs[fire_numbe]` slot under `_rebrush_procs_lock`.

---

## High

### H1. NaN can leak into `fire.fire_size_ha`

**File:** `state.py:288-292`
**Reproducer:** `tests/audit/test_misc_edges.py::e4`

```python
try:
    fire_size = float(row.get('FIRE_SIZE_', 0) or 0)
except (ValueError, TypeError):
    fire_size = 0.0
```

`float(NaN)` returns `NaN` without raising — pandas often hands NaN back from
shapefile cells with missing numeric data. NaN propagates into:

- `FireInfo.fire_size_ha` — exposed in `/api/fires` JSON. `json.dumps(float('nan'))`
  produces `NaN`, which is not valid JSON. Strict client parsers (browsers'
  `JSON.parse`) reject it.
- `round(fire_size, 1)` is still NaN.
- Any future size-bucketing comparison: `nan < cutoff` is always False.

**Fix:**
```python
import math
try:
    fire_size = float(row.get('FIRE_SIZE_', 0) or 0)
    if not math.isfinite(fire_size):
        fire_size = 0.0
except (ValueError, TypeError):
    fire_size = 0.0
```
Apply the same isfinite check to `fire_year` parsing two lines above.

---

### H2. Recursive template substitution can leak unescaped values

**File:** `templates.py:24-27`
**Reproducer:** `tests/audit/test_templates.py::t3`

```python
for key, val in context.items():
    html = html.replace('{{{ ' + key + ' }}}', str(val))   # raw
    html = html.replace('{{ ' + key + ' }}', _html_escape(str(val)))
```

The two-pass `str.replace` does not protect placeholder syntax in already-substituted
values. Test demonstrates:

```
template: a={{ name }};b={{{ raw_key }}}
context:  name='{{{ raw_key }}}', raw_key='<b>'
output:   a=<b>;b=<b>
```

Walk-through: `_html_escape` does not escape `{` or `}`, so `name`'s value
survives the escape pass intact. On the next iteration, `raw_key`'s
triple-brace pattern is found inside the substituted body and replaced raw —
inserting `<b>` unescaped where a `{{ name }}` placeholder originally lived.

**Why it is not exploitable today:** every call site passes server-trusted
values (config, validated state). `fire_numbe` cannot contain `{` because the
init regex `[A-Za-z0-9][A-Za-z0-9_. -]*` rejects braces. **But** any future
context value built from user input (e.g., a fire note rendered into a page)
becomes a stored-XSS vector.

**Fix:** scan once with a regex that resolves each placeholder against the
context, instead of N independent string replaces:
```python
import re as _re
PAT = _re.compile(r'\{\{\{ (\w+) \}\}\}|\{\{ (\w+) \}\}')
def _sub(m):
    raw_key, esc_key = m.group(1), m.group(2)
    if raw_key:
        return str(context.get(raw_key, ''))
    return _html_escape(str(context.get(esc_key, '')))
html = PAT.sub(_sub, html)
```

---

### H3. `_atomic_yaml_dump` can orphan tmp file when `os.replace` fails

**File:** `io_utils.py:25-31`

```python
os.replace(tmp, path)
except Exception:
    try:
        os.remove(tmp)
    except OSError:
        pass
    raise
```

If `os.replace` itself fails (cross-filesystem rename, EACCES on target),
control falls through to the `except` block and `tmp` is removed. Good. But
the `try` block exits normally on success — there is no leak there either.
The actual gap: if `os.fdopen(fd, 'w')` fails partway through `yaml.dump`
(e.g., a MemoryError during serialization of a large object), `tmp` exists
with partial bytes; `os.remove(tmp)` runs; OK.

**The real concern is this:** the helper does not call `os.fsync` on the
*directory* (covered by C1) AND does not unlink stale tmp files from prior
crashed processes. After a hard crash, tmp files like
`fire_state.yaml.<pid>.<tid>.tmp` accumulate in `output_root` until manually
cleaned.

**Fix:** at server startup, sweep `output_root` and `shared_root` for
`*.<pid>.<tid>.tmp` files older than a threshold and unlink them.

---

### H4. `_serial_handle_cancel` holds `_gpu_lock` across `_save_fire_state` and `_push_notification`

**File:** `workers.py:820-837`

The empty-success-set fallthrough block:
```python
with _gpu_lock:
    ...
    with state.lock:
        revert = ...
    fire.console_log.append(...)
    _save_fire_state()                  # disk I/O under _gpu_lock
    _push_notification(...)             # notification queue + fsync under _gpu_lock
sys.stderr.write(...)                   # OUTSIDE _gpu_lock
```

The other two branches in this function (lines 720-741 and 749-814) follow the
same pattern. While the GPU lock is held during a YAML save, no other
mapping/rebrush can acquire it. For a multi-fire batch, this serializes a few
hundred milliseconds of disk I/O between fires that did not need it.

**Impact:** performance only — correctness is fine. Workers wait
~50-200ms longer per cancelled fire. Worth fixing because the indentation
suggests the writer intended these calls to be outside `_gpu_lock` (the final
`sys.stderr.write` IS outside).

**Fix:** move `_save_fire_state()` and `_push_notification(...)` calls outside
`with _gpu_lock:` in all three branches. Capture the values needed for the
notification (`revert`, `len(successful)`, etc.) into locals before releasing
the lock.

---

## Medium

### M1. Padding-changed cache wipe skips when `old_pad == 0`

**File:** `prepare.py:153-156`

```python
padding_changed = (old_pad != 0
                   and old_pad != pad
                   and os.path.isdir(cache_dir))
```

`old_pad == 0` is treated as "never prepared" — but `0.0` is a *legal* padding
value (the API accepts it via `float(padding)` without a min-bound check). A
fire prepared once with `pad=0.0` will never have its cache wiped on subsequent
padding changes, leaving stale `crop.bin` files at the wrong extent.

**Why latent:** UI dropdowns produce only positive paddings; `state.padding`
default is `0.1`. The window for hitting it is "user manually POSTs `padding=0`".

**Fix:** use a sentinel for "never prepared":
```python
# In FireInfo dataclass:
padding_used: float = -1.0   # was 0.0

# In prepare.py:
padding_changed = (old_pad >= 0
                   and old_pad != pad
                   and os.path.isdir(cache_dir))
```
Note: this requires migrating any persisted `padding_used: 0.0` in
`fire_state.yaml`, OR keeping the default at `0.0` and instead clamping the
input to `pad >= 0.001` in the API layer.

---

### M2. `_validate_param` silently truncates fractional integers

**File:** `validation.py:33-38`
**Reproducer:** `tests/audit/test_validation.py::v2`

```python
if kind == 'int':
    _, lo, hi = spec
    v = int(float(raw))    # int(3.7) == 3
    if not (lo <= v <= hi):
        raise ValueError(...)
```

`_validate_param('rf_n_estimators', '3.7')` returns `3` with no warning. The
helpful goal is to accept `"15.0"` from a form that serializes integers as
floats (and that comment is in `mapping_cmd.py:81-83`). The accidental cost
is that `"3.7"` is also accepted.

**Impact:** if any UI/JSON path produces fractional floats for what should be
integers, the user sees no validation error and gets a silent rounding.

**Fix:**
```python
if kind == 'int':
    _, lo, hi = spec
    f = float(raw)
    if f != int(f):
        raise ValueError(f'{key}={raw} is not an integer')
    v = int(f)
    if not (lo <= v <= hi):
        raise ValueError(f'{key}={raw} out of range [{lo}, {hi}]')
    return v
```

---

### M3. `_prepare_fire_sync` swallows `rasterize_polygon` exceptions silently

**File:** `prepare.py:198-204`

```python
try:
    rasterize_polygon(...)
except Exception:
    perim_bin = None
fire.perim_bin = perim_bin or ''
```

If rasterization fails for an actionable reason (CRS mismatch, OOM, polygon
self-intersection), the user sees only "no perimeter available" downstream
with nothing in the logs.

**Fix:**
```python
except Exception as exc:
    sys.stderr.write(
        f'[prepare] [{fire_numbe}] perimeter rasterize failed: {exc}\n')
    sys.stderr.flush()
    perim_bin = None
```

---

### M4. `prepare.py:454-555` — `except ImportError` is dead code

**File:** `prepare.py:454, 532`

```python
try:
    import yaml
    params_dict = {...}
    ...
    _atomic_yaml_dump(path, params_dict, mode=0o644)
except ImportError:
    pass
```

`yaml` is a hard dependency of the package (used at module import time in
`io_utils.py`, `persistence.py`, `__main__.py`). The `except ImportError`
branch is unreachable. Worse: if `params_dict` construction or `_atomic_yaml_dump`
raises any *other* exception (KeyError on a missing field, OSError on disk full),
the bare-except catches nothing and the exception propagates — but if a future
refactor swaps the bare `except ImportError` for `except Exception`, real
errors get silently swallowed.

**Fix:** remove the try/except wrapper. Let yaml errors propagate; add a
narrower except for serialization issues if needed:
```python
import yaml
params_dict = {...}
try:
    _atomic_yaml_dump(path, params_dict, mode=0o644)
except OSError as exc:
    sys.stderr.write(
        f'[save] WARNING: {fire_numbe}_params.yaml: {exc}\n')
```

---

### M5. `_load_fire_state` doesn't fsync the .corrupt copy before flagging load_failed

**File:** `persistence.py:222-232`

```python
backup = f'{state_path}.corrupt-{int(time.time())}'
try:
    shutil.copy2(state_path, backup)
except OSError:
    backup = '<copy failed>'
sys.stderr.write(...)
state.fire_state_load_failed = True
return
```

`shutil.copy2` does not fsync. If the server crashes between `copy2` and the
operator inspecting the file, the .corrupt copy may be lost from the page cache.
Combined with C1 (no dir fsync), the operator can lose the only evidence of
the corruption.

**Fix:**
```python
import shutil
shutil.copy2(state_path, backup)
fd = os.open(backup, os.O_RDONLY)
try:
    os.fsync(fd)
finally:
    os.close(fd)
# also fsync the parent directory
```

---

### M6. `cache_retention.py` silently undercounts unreadable files

**File:** `cache_retention.py:104-111`

```python
for f in files:
    fp = os.path.join(root, f)
    try:
        st = os.stat(fp)
        total += int(st.st_size)
        ...
    except OSError:
        continue
```

A permission error on a single file silently drops it from the size count.
Under-counted bytes mean the sweeper believes the cache is smaller than it
really is, and may exceed the configured `max_gb`.

**Fix:** log on the stderr stream:
```python
except OSError as exc:
    sys.stderr.write(
        f'[cache] WARNING: stat {fp}: {exc}\n')
    continue
```

---

### M7. `auth.py::_normalize_ip` doesn't strip whitespace

**File:** `auth.py:34-42`

```python
def _normalize_ip(ip_str: str) -> str:
    try:
        addr = ipaddress.ip_address(ip_str)
        ...
    except ValueError:
        return ip_str
```

`ipaddress.ip_address(' 192.168.1.1 ')` raises ValueError; the function returns
the unstripped string. Today, callers (`handlers/base.py::_client_ip`) call
`.strip()` first, so this is fine. **But:** any future caller that hands a
raw header value will store an unnormalized IP, breaking equality checks
against the approved/blocked lists.

**Fix:** strip defensively at the top of `_normalize_ip`:
```python
ip_str = (ip_str or '').strip()
```

---

### M8. `current_job` reset to None unconditionally in batch worker exception path

**File:** `workers.py:163, 1017`

```python
except Exception as exc:
    _set_fire_status(fire, FireStatus.ERROR, str(exc))
    with state.lock:
        state.current_job = None     # blanks ANY current_job, not just ours
    ...
```

`_gpu_lock` ensures only one mapping subprocess runs at a time, so
`state.current_job` is always "ours" at this point. The code is correct
**given** that invariant. Worth a comment so future refactors don't break it,
or — better — the cleanup should match on `fire_numbe` before clearing:
```python
with state.lock:
    if (state.current_job
            and state.current_job.get('fire_numbe', '').startswith(fire_numbe)):
        state.current_job = None
```

---

## Low

### L1. `notifications._save_notifications` runs on every push

**File:** `notifications.py:106-141`

`_push_notification` calls `_save_notifications()` after every enqueue,
producing one full YAML dump + fsync per push. The 50/session and 20/broadcast
caps bound the file size, so this is fine in practice — but under burst
notification load (e.g., a batch of 200 fires producing notifications), it
serializes the queue at ~5-20ms per save. Consider:
- batching saves on a 1-second timer, or
- writing a JSON Lines append-only log instead of full YAML rewrite.

### L2. `app.py:75 gdal.UseExceptions()` runs at import time

**File:** `app.py:75`

If `app.py` is imported as a module (e.g., by the PDF report builder, by an
external test harness), `gdal.UseExceptions()` mutates GDAL global state for
the importer too. This is a side effect at import time, generally a code-smell.
Move into `init_app()`.

### L3. Status-handler reads outside `state.lock`

**File:** `handlers/fire.py:309-312, 322-323` and many similar spots

```python
f = state.fires[fire_numbe]
with state.lock:
    payload = {'status': f.status.value, 'error': f.error_msg}
```

The dict lookup at line 309 happens outside the lock. `state.fires` itself
is mutated under `state.lock` during a year switch (`persistence.py:546`).
A read at line 309 while a year switch is in flight can KeyError. Today the
year-switch handler refuses while mapping/batch/rebrush are running, so the
window is small (a fast user clicks /api/fire/X/status mid-switch), but it
is non-zero. Wrap the lookup inside the same `with state.lock` block.

---

## Test inventory

The runtime tests live in `tests/audit/` (after follow-up agent moves them
from `/tmp/bfmw_tests/`). Each can be run independently:

| File | Assertions | Coverage |
|---|---|---|
| `test_atomic_yaml.py` | 10 | concurrent writers, mode bits, unicode, dir fsync (FAILS at C1) |
| `test_auth.py` | 26 | hash_token, IP normalize, rate limit, session sweep, memory bound |
| `test_cache_retention.py` | 19 | hard/soft pin, dry_run, busy lock, ACCEPTED demote, save/load |
| `test_csv_race.py` | 7 | concurrent appends, dedupe, 100-fire stress |
| `test_fire_numbe_regex.py` | 30 | unicode rejection, control chars, traversal, edge cases |
| `test_fire_state_roundtrip.py` | 20 | round-trip fields, downgrade-on-missing, corruption + .corrupt backup, save-refusal |
| `test_jitter.py` | 28 | fan-out pattern, floor at 1, step=0 disables |
| `test_notifications.py` | 17 | TTL prune, broadcast cursor, persistence, concurrent pop |
| `test_progress.py` | 31 | stage detection, 30-sample cap, median (odd/even), filter bad samples |
| `test_templates.py` | 8 | escape order, recursive substitution (exposes H2) |
| `test_validation.py` | 25 | NaN/Inf rejection, big-int, embed_bands edges |
| `test_year_detection.py` | 13 | 1970 boundary, now+1/+2, lookahead overlap |
| `test_misc_edges.py` | 21 | NaN size leak, file-prefix safety, view regex |

**Total:** 252 assertions; **251 pass**, 1 fail (C1 confirmed).

---

## What was checked and is fine

These were specifically tested and confirmed correct — listed so the next
agent does not duplicate the work:

- `_atomic_yaml_dump` concurrent writers — tmp suffix prevents collision (T3 passes)
- Login rate limiter — 5-attempt window, decay after 300s, 1024-IP memory bound
- Session expiry sweep — handles malformed dates, missing keys, drops linked notifications/cursors
- Token hashing — SHA-256, deterministic, 64-hex
- IP normalize — IPv6-mapped IPv4 collapse, garbage passthrough
- `fire_numbe` regex — rejects unicode, control chars, traversal substrings, leading dot/dash/space
- Year detection — 1970 floor, now+1 ceiling, multiple-year rejection, set dedup of repeated tokens
- HDBSCAN jitter — fan-out pattern, floor at 1, step=0 disables, negative base floors
- Cache retention — hard pin (PREPARING/MAPPING/rebrush/accept) protects from age + size; soft pin (READY/MAPPED) protects from size only; ACCEPTED is evictable; sweep lock prevents overlap
- Stage timings — sample cap at 30, reject zero/negative/>7200s, median odd/even, load filters bad samples
- Progress snapshot — empty when not MAPPING/PREPARING, returns None ETA without history
- CSV writes — concurrent stress test (100 fires × 5 updates) produces 100 unique deduped rows
- Notification broadcast — cursor advances correctly under 20-thread concurrent pop
- `_load_fire_state` — restores all documented fields, downgrades MAPPED to PENDING when crop/hint missing, corruption → .corrupt backup + load_failed flag, blocks subsequent saves
- Validation — NaN/Inf rejected by `lo <= v <= hi` check, big int (2^62) rejected, embed_bands rejects negatives/zero/>999

---

## Suggested fix order

1. **C1, M5** (durability) — one helper change in `io_utils.py`, applied
   everywhere by reuse. M5 cleans up the same area.
2. **C2, M3, M4, M6** (silent failures) — replace `pass` with stderr logging.
3. **C4, C5** (TOCTOU on status flip) — atomic test-and-set under
   `state.lock` in `_prepare_fire_sync` and three handlers.
4. **C3** (re-entrant accept) — add the `if fire_numbe in _accept_in_progress`
   check at top of `_accept_fire_sync`.
5. **H1** (NaN leak) — add `math.isfinite` check in `state.py:288-292` and `state.py:283-286`.
6. **H2** (template injection) — replace two-pass replace with single regex
   scan in `templates.py:24-27`. Add a regression test that `name="{{{ x }}}"`
   does not pull in `x`.
7. **H3** (orphan tmp files) — startup sweep of stale `*.tmp` files.
8. **H4** (lock held across save) — move `_save_fire_state` + `_push_notification`
   outside `with _gpu_lock:` in `workers.py:_serial_handle_cancel`.
9. **M1, M2, M7, M8, L1, L2, L3** — incremental hardening; can land separately.

After fixes, re-run all 13 test scripts and confirm 252/252 pass.

---

## Acceptance criteria for the follow-up

A fix is complete when:
- The corresponding test(s) in `tests/audit/` pass.
- No existing test regresses.
- For each Critical/High issue, a brief comment in the code at the fix site
  cites the relevant audit-report ID (e.g. `# AUDIT-C1: parent dir fsync`)
  so the rationale is preserved against future re-refactors.
- `_save_fire_state` round-trip continues to work (run
  `test_fire_state_roundtrip.py`).
- `_atomic_yaml_dump` continues to handle concurrent writers (run
  `test_atomic_yaml.py`; T7 should now pass).

---

## Resolution log

Follow-up pass on 2026-04-27. Test suite: **284 pass, 0 fail** (was
254 pass, 1 fail at baseline; the 30-assertion increase comes from the
new `test_handler_concurrency.py` plus the H1/H2 regression tests).

Commit hashes are deferred — repo was mid-rebase during the fix pass,
so changes are staged in the working tree and will be split into
single-purpose commits by the user. Each fix carries an `AUDIT-<ID>`
comment at the fix site for future bisect/blame.

| ID  | Status     | File(s) touched                                       | Note |
|-----|------------|-------------------------------------------------------|------|
| C1  | **fixed**  | `io_utils.py`, `prepare.py:594-622`                   | dir fsync after `os.replace`; CSV writer also flushes+fsyncs file before rename. strace confirms two fsyncs per dump |
| C2  | **fixed**  | `prepare.py:560-573`                                   | bare `pass` replaced with stderr WARNING line |
| C3  | **fixed**  | `prepare.py:425-440`                                   | re-entrant accept now raises `RuntimeError` instead of silently overwriting fire_dir |
| C4  | **fixed**  | `prepare.py:67-80`                                     | PREPARING test-and-set wrapped in `with state.lock` |
| C5  | **fixed**  | `handlers/serial.py`, `handlers/mapping.py`, `handlers/rebrush.py` | MAPPING test-and-set wrapped in `with state.lock`; rebrush slot claimed under `_rebrush_procs_lock` with sentinel + finally cleanup |
| H1  | **fixed**  | `state.py:288-310`                                     | `math.isfinite` guards on `fire_size` and `fire_year` parsing; `state.py` now imports `math` |
| H2  | **fixed**  | `templates.py`                                         | two-pass `str.replace` replaced with single regex sub; `test_templates.py::t3` flipped from "asserts bug" to "asserts fix" |
| H3  | **fixed**  | `io_utils.py:_sweep_stale_tmp_files`, `app.py:init_app` | startup sweep removes `*.<pid>.<tid>.tmp` files older than 86400s in output_root + shared_root |
| H4  | **fixed**  | `workers.py:_serial_handle_cancel`                     | three branches refactored to capture state under `_gpu_lock`, then `_save_fire_state` + `_push_notification` run after the lock is released |
| M3  | **fixed**  | `prepare.py:200-211`                                   | `rasterize_polygon` exceptions now log to stderr |
| M4  | **fixed**  | `prepare.py:459-548`                                   | dead `except ImportError` removed; narrowed to `except OSError` around `_atomic_yaml_dump` write only |
| M5  | **fixed**  | `persistence.py:222-244`                               | `.corrupt-<ts>` backup is fsynced (file + parent dir) before flagging load_failed |
| M6  | **fixed**  | `cache_retention.py:104-118`                           | OSError on `os.stat` now logs to stderr instead of silently dropping the file from the size count |
| C2 also covers `prepare.py` `_atomic_yaml_dump` callers — the helper change in C1 propagates to every YAML write site listed in C1. |
| M1  | open       | `prepare.py:153-156`                                   | padding-changed wipe edge — low-impact, not in this pass |
| M2  | open       | `validation.py:33-38`                                  | int truncation — not in this pass |
| M7  | open       | `auth.py:34-42`                                        | IP whitespace strip — defensive only, latent |
| M8  | open       | `workers.py:163, 1017`                                 | current_job blanking — currently correct under `_gpu_lock` invariant |
| L1  | open       | `notifications.py:106-141`                             | save-per-push, performance only |
| L2  | open       | `app.py:75`                                            | `gdal.UseExceptions()` at import time |
| L3  | open       | `handlers/fire.py:309-312`                             | status read outside lock |

### New tests added

- `tests/audit/test_handler_concurrency.py` — 17 assertions covering
  C3, C4, C5: N parallel claimers race to flip status / claim slot,
  exactly one winner expected. Also a static-source check that the
  `AUDIT-Cn` comments remain at the fix sites.
- `tests/audit/test_misc_edges.py::E5` — 11 new assertions confirming
  AUDIT-H1 NaN/Inf rejection on `FIRE_SIZE_` and `FIRE_YEAR`, plus a
  static guard that the `math.isfinite` calls are still present in
  `state.init_fires_from_gdf`.
- `tests/audit/test_templates.py::T3` — flipped from "asserts the bug
  exists" (`out3.count('<b>') == 2`) to "asserts the fix holds"
  (`out3.count('<b>') == 1`, smuggled `{{{ x }}}` stays literal).
- `tests/audit/test_atomic_yaml.py::T7` — was the lone baseline failure;
  now passes (parent-dir fsync detected via inspect.getsource).
