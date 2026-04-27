"""Tiny `{{ key }}` template renderer (replaces Jinja2)."""

import os
import re

_HERE = os.path.dirname(os.path.abspath(__file__))

# AUDIT-H2: single-pass scan that resolves placeholders against the
# context in one walk. The previous two-pass str.replace re-scanned the
# already-substituted body for placeholder syntax, so a value containing
# `{{{ x }}}` could pull `x`'s raw value into a slot that was originally
# `{{ name }}` — a stored-XSS vector once any context value comes from
# user input.
_PLACEHOLDER_PAT = re.compile(r'\{\{\{ (\w+) \}\}\}|\{\{ (\w+) \}\}')


def _html_escape(s: str) -> str:
    """Escape HTML special characters."""
    return (s.replace('&', '&amp;').replace('<', '&lt;')
             .replace('>', '&gt;').replace('"', '&quot;')
             .replace("'", '&#39;'))


def render_template(name: str, context: dict) -> str:
    """Replace ``{{ key }}`` placeholders in a template file.

    Values are HTML-escaped by default.  Use ``{{{ key }}}`` for raw
    (unescaped) insertion when the value is known-safe.
    """
    path = os.path.join(_HERE, 'templates', name)
    with open(path) as f:
        html = f.read()

    def _sub(m):
        raw_key, esc_key = m.group(1), m.group(2)
        if raw_key:
            return str(context.get(raw_key, ''))
        return _html_escape(str(context.get(esc_key, '')))

    return _PLACEHOLDER_PAT.sub(_sub, html)
