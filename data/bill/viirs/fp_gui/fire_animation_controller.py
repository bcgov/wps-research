"""
viirs/fp_gui/fire_animation_controller.py

FireAnimationController: drives the date-by-date animation loop
using tkinter's after() mechanism for play/pause control.
"""

import tkinter as tk
from datetime import date, timedelta
from typing import Optional, Callable, List

from config import DEFAULT_ANIMATION_INTERVAL_MS


class FireAnimationController:
    """
    Steps through a list of calendar dates, calling a user-supplied
    callback on each step.  Supports play / pause / step / reset.

    Parameters
    ----------
    root : tk.Tk or tk.Widget
        Needed for tkinter's .after() scheduling.
    on_frame : callable(date) -> None
        Called with the current date on every animation frame.
    on_finished : callable() -> None, optional
        Called when animation reaches the end date.
    """

    def __init__(
        self,
        root: tk.Tk,
        on_frame: Callable[[date], None],
        on_finished: Optional[Callable[[], None]] = None,
    ):
        self._root = root
        self._on_frame = on_frame
        self._on_finished = on_finished

        self._dates: List[date] = []
        self._index: int = 0
        self._playing: bool = False
        self._interval_ms: int = DEFAULT_ANIMATION_INTERVAL_MS
        self._after_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def current_date(self) -> Optional[date]:
        if 0 <= self._index < len(self._dates):
            return self._dates[self._index]
        return None

    @property
    def current_index(self) -> int:
        return self._index

    @property
    def total_frames(self) -> int:
        return len(self._dates)

    @property
    def interval_ms(self) -> int:
        return self._interval_ms

    @interval_ms.setter
    def interval_ms(self, value: int):
        self._interval_ms = max(50, value)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def set_date_range(self, start: date, end: date):
        """Build the full list of daily dates from start to end inclusive."""
        self._dates = []
        d = start
        while d <= end:
            self._dates.append(d)
            d += timedelta(days=1)
        self._index = 0

    def set_dates(self, dates: List[date]):
        """Set an explicit list of dates to animate through."""
        self._dates = sorted(dates)
        self._index = 0

    # ------------------------------------------------------------------
    # Playback controls
    # ------------------------------------------------------------------

    def play(self):
        """Start or resume the animation."""
        if not self._dates:
            return
        self._playing = True
        self._schedule_next()

    def pause(self):
        """Pause the animation."""
        self._playing = False
        if self._after_id is not None:
            self._root.after_cancel(self._after_id)
            self._after_id = None

    def stop(self):
        """Stop and reset to frame 0."""
        self.pause()
        self._index = 0

    def step_forward(self):
        """Advance one frame manually."""
        if self._index < len(self._dates) - 1:
            self._index += 1
            self._emit_frame()
        elif self._on_finished:
            self._on_finished()

    def step_backward(self):
        """Go back one frame."""
        if self._index > 0:
            self._index -= 1
            self._emit_frame()

    def jump_to(self, index: int):
        """Jump to a specific frame index."""
        self._index = max(0, min(index, len(self._dates) - 1))
        self._emit_frame()

    def jump_by(self, n: int):
        """Jump forward (+) or backward (−) by n frames."""
        new_idx = self._index + n
        new_idx = max(0, min(new_idx, len(self._dates) - 1))
        if new_idx != self._index:
            self._index = new_idx
            self._emit_frame()

    def jump_to_date(self, target: date):
        """Jump to the frame closest to *target*."""
        for i, d in enumerate(self._dates):
            if d >= target:
                self._index = i
                self._emit_frame()
                return
        # If past end, go to last
        if self._dates:
            self._index = len(self._dates) - 1
            self._emit_frame()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _schedule_next(self):
        if not self._playing:
            return
        if self._index >= len(self._dates):
            self._playing = False
            if self._on_finished:
                self._on_finished()
            return

        self._emit_frame()
        self._index += 1

        if self._index < len(self._dates):
            self._after_id = self._root.after(self._interval_ms, self._schedule_next)
        else:
            self._playing = False
            if self._on_finished:
                self._on_finished()

    def _emit_frame(self):
        if 0 <= self._index < len(self._dates):
            self._on_frame(self._dates[self._index])