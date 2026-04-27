"""Atomic YAML write helper shared by every persistence module."""

import os
import re
import sys
import threading
import time


def _atomic_yaml_dump(path: str, data, mode: int = 0o600):
    """Write YAML atomically via tmp + rename. Sets restrictive permissions.

    Uses a unique tmp suffix (pid + thread id) so concurrent writers to the
    same target path do not clobber each other's tmp file."""
    import yaml
    tmp = f'{path}.{os.getpid()}.{threading.get_ident()}.tmp'
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
        with os.fdopen(fd, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
            # fsync before rename: os.replace is rename-atomic, but without
            # fsync the new file's bytes can still be in the page cache when
            # the directory entry flips. A power loss in that window leaves
            # a zero-length or truncated file after reboot even though the
            # rename "succeeded". Critical for fire_state.yaml et al.
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        # AUDIT-C1: parent dir fsync — see AUDIT_REPORT.md.
        # POSIX rename is atomic, not durable; without fsync on the
        # directory, a power loss after os.replace can leave the rename
        # entry in volatile cache and lose the new file on reboot.
        dir_fd = os.open(os.path.dirname(path) or '.', os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


# AUDIT-H3: tmp suffix is `<basename>.<pid>.<tid>.tmp`. After a hard
# crash these accumulate in output_root/shared_root forever.
_TMP_SUFFIX_RE = re.compile(r'\.\d+\.\d+\.tmp$')


def _sweep_stale_tmp_files(roots, max_age_seconds: int = 86400) -> int:
    """Remove `*.<pid>.<tid>.tmp` files older than max_age_seconds.

    Called at server startup so a previous crash's orphaned tmp files
    don't accumulate in output_root or shared_root. Returns the count
    removed; logs each unlink to stderr."""
    removed = 0
    now = time.time()
    seen = set()
    for root in roots:
        if not root or root in seen:
            continue
        seen.add(root)
        if not os.path.isdir(root):
            continue
        try:
            entries = os.listdir(root)
        except OSError as exc:
            sys.stderr.write(
                f'[startup] WARNING: tmp sweep listdir({root!r}): '
                f'{exc}\n')
            sys.stderr.flush()
            continue
        for name in entries:
            if not _TMP_SUFFIX_RE.search(name):
                continue
            fp = os.path.join(root, name)
            try:
                st = os.stat(fp)
            except OSError:
                continue
            if (now - st.st_mtime) < max_age_seconds:
                continue
            try:
                os.unlink(fp)
                removed += 1
                sys.stderr.write(
                    f'[startup] removed stale tmp: {fp}\n')
            except OSError as exc:
                sys.stderr.write(
                    f'[startup] WARNING: tmp sweep unlink {fp}: '
                    f'{exc}\n')
        sys.stderr.flush()
    return removed
