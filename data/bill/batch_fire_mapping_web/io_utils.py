"""Atomic YAML write helper shared by every persistence module."""

import os
import threading


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
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
