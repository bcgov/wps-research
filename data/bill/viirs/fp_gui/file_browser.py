"""
viirs/fp_gui/file_browser.py

Custom file/directory browser dialog with:
  - Breadcrumb path bar (max 4 tabs: / is always first, last 3 follow)
  - Back button
  - Directory listing with icons
  - Create New Folder option
  - File type filtering
  - current_value parameter pre-fills the Selected bar
  - Works as drop-in replacement for filedialog.askdirectory / askopenfilename
"""

import os
import tkinter as tk
from tkinter import ttk, simpledialog
from typing import Optional, List, Tuple


class FileBrowserDialog:
    """
    A modal file/directory browser with macOS-inspired navigation.

    Parameters
    ----------
    parent : tk.Tk or tk.Toplevel
    title : str
    initial_dir : str
    mode : 'directory' | 'file'
    filetypes : list of (label, pattern)
    allow_create_folder : bool
    current_value : str
        Pre-fills the Selected bar and opens in that location.
    """

    def __init__(
        self,
        parent: tk.Tk,
        title: str = "Browse",
        initial_dir: str = "",
        mode: str = "directory",
        filetypes: Optional[List[Tuple[str, str]]] = None,
        allow_create_folder: bool = False,
        current_value: str = "",
    ):
        self._parent = parent
        self._mode = mode
        self._filetypes = filetypes or []
        self._allow_create_folder = allow_create_folder
        self._result: Optional[str] = None
        self._initial_selected: str = current_value or ""

        self._history: List[str] = []

        # Resolve starting directory — prefer current_value's location
        start_dir = ""
        if current_value:
            if os.path.isdir(current_value):
                start_dir = current_value
            elif os.path.isfile(current_value):
                start_dir = os.path.dirname(os.path.abspath(current_value))
            elif os.path.isdir(os.path.dirname(current_value)):
                start_dir = os.path.dirname(os.path.abspath(current_value))

        if not start_dir:
            if initial_dir and os.path.isdir(initial_dir):
                start_dir = os.path.abspath(initial_dir)
            elif initial_dir and os.path.isfile(initial_dir):
                start_dir = os.path.dirname(os.path.abspath(initial_dir))
            else:
                start_dir = os.path.expanduser("~")

        self._current_dir = os.path.abspath(start_dir)

        self._win = tk.Toplevel(parent)
        self._win.title(title)
        self._win.transient(parent)
        self._win.grab_set()

        w, h = 720, 520
        sx = parent.winfo_screenwidth()
        sy = parent.winfo_screenheight()
        self._win.geometry(f"{w}x{h}+{(sx-w)//2}+{(sy-h)//2}")
        self._win.resizable(True, True)

        self._win.columnconfigure(0, weight=1)
        self._win.rowconfigure(1, weight=1)

        self._build_ui()
        self._navigate_to(self._current_dir)

        # Pre-fill Selected bar with current_value
        if self._initial_selected:
            self._selected_var.set(self._initial_selected)

        self._win.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self._win.wait_window()

    @property
    def result(self) -> Optional[str]:
        return self._result

    # ==================================================================
    # UI
    # ==================================================================

    def _build_ui(self):
        top = ttk.Frame(self._win, padding=(6, 4))
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        self._back_btn = ttk.Button(
            top, text="\u2190", width=3, command=self._go_back)
        self._back_btn.grid(row=0, column=0, padx=(0, 4))

        self._breadcrumb_frame = ttk.Frame(top)
        self._breadcrumb_frame.grid(row=0, column=1, sticky="ew")

        if self._allow_create_folder:
            ttk.Button(
                top, text="\u2795 New Folder", command=self._create_folder
            ).grid(row=0, column=2, padx=(8, 0))

        # Main listing
        list_frame = ttk.Frame(self._win)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=6, pady=(0, 4))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)

        style = ttk.Style()
        style.configure(
            "Browser.Treeview", rowheight=26, font=("TkDefaultFont", 10))
        style.configure(
            "Browser.Treeview.Heading", font=("TkDefaultFont", 10, "bold"))

        self._tree = ttk.Treeview(
            list_frame, columns=("name", "type", "size"),
            show="headings", style="Browser.Treeview",
            selectmode="browse",
        )
        self._tree.heading("name", text="Name", anchor="w")
        self._tree.heading("type", text="Type", anchor="w")
        self._tree.heading("size", text="Size", anchor="e")
        self._tree.column("name", width=400, minwidth=200, anchor="w")
        self._tree.column("type", width=120, minwidth=80, anchor="w")
        self._tree.column("size", width=100, minwidth=60, anchor="e")

        vsb = ttk.Scrollbar(list_frame, orient="vertical",
                            command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        self._tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")

        self._tree.bind("<Double-1>", self._on_double_click)
        self._tree.bind("<Return>", self._on_double_click)

        # Bottom bar
        bot = ttk.Frame(self._win, padding=(6, 4))
        bot.grid(row=2, column=0, sticky="ew")
        bot.columnconfigure(1, weight=1)

        ttk.Label(bot, text="Selected:").grid(row=0, column=0, padx=(0, 4))
        self._selected_var = tk.StringVar()
        ttk.Entry(bot, textvariable=self._selected_var).grid(
            row=0, column=1, sticky="ew", padx=4)

        btn_frame = ttk.Frame(bot)
        btn_frame.grid(row=0, column=2, padx=(8, 0))

        tk.Button(
            btn_frame, text="  Select  ", bg="#4CAF50", fg="white",
            font=("TkDefaultFont", 9, "bold"), activebackground="#388E3C",
            command=self._on_select,
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(
            btn_frame, text="  Cancel  ", bg="#F44336", fg="white",
            font=("TkDefaultFont", 9, "bold"), activebackground="#C62828",
            command=self._on_cancel,
        ).pack(side=tk.LEFT, padx=4)

        self._tree.bind("<<TreeviewSelect>>", self._on_tree_select)

    # ==================================================================
    # Navigation
    # ==================================================================

    def _navigate_to(self, path: str, push_history: bool = True):
        path = os.path.abspath(path)
        if not os.path.isdir(path):
            return

        if push_history and self._current_dir != path:
            self._history.append(self._current_dir)

        self._current_dir = path
        self._back_btn.configure(
            state=tk.NORMAL if self._history else tk.DISABLED)

        self._refresh_breadcrumbs()
        self._refresh_listing()

        if self._mode == "directory":
            self._selected_var.set(self._current_dir)

    def _go_back(self):
        if self._history:
            prev = self._history.pop()
            self._navigate_to(prev, push_history=False)

    def _refresh_breadcrumbs(self):
        """
        Max 4 breadcrumb buttons.
        First is always  /  (filesystem root).
        Last 3 are the deepest path segments leading to current_dir.
        If path has <=4 segments, show all.  Otherwise: / › … › gp › parent › current
        """
        for w in self._breadcrumb_frame.winfo_children():
            w.destroy()

        # Build full list: [(label, full_path), ...]  root-first
        parts = []
        p = self._current_dir
        while True:
            head, tail = os.path.split(p)
            if tail:
                parts.append((tail, p))
            else:
                if head:
                    parts.append(("/", head))
                break
            p = head
        parts.reverse()

        MAX_CRUMBS = 4
        need_ellipsis = len(parts) > MAX_CRUMBS

        if need_ellipsis:
            # Show: [/]  …  [last 3]
            visible = [parts[0]] + parts[-(MAX_CRUMBS - 1):]
        else:
            visible = parts

        for i, (label, full_path) in enumerate(visible):
            # Separator arrow between buttons
            if i > 0:
                ttk.Label(
                    self._breadcrumb_frame, text="\u203a",
                    font=("TkDefaultFont", 10),
                ).pack(side=tk.LEFT, padx=1)

            # After the root button, insert ellipsis if segments were skipped
            if need_ellipsis and i == 0:
                display = label if len(label) <= 20 else label[:17] + "\u2026"
                ttk.Button(
                    self._breadcrumb_frame, text=display,
                    command=lambda fp=full_path: self._navigate_to(fp),
                ).pack(side=tk.LEFT, padx=1)
                ttk.Label(
                    self._breadcrumb_frame, text="\u203a  \u2026",
                    font=("TkDefaultFont", 10),
                ).pack(side=tk.LEFT, padx=1)
                continue

            display = label if len(label) <= 20 else label[:17] + "\u2026"
            ttk.Button(
                self._breadcrumb_frame, text=display,
                command=lambda fp=full_path: self._navigate_to(fp),
            ).pack(side=tk.LEFT, padx=1)

    def _refresh_listing(self):
        self._tree.delete(*self._tree.get_children())

        try:
            entries = sorted(os.listdir(self._current_dir),
                             key=lambda x: (not os.path.isdir(
                                 os.path.join(self._current_dir, x)),
                                 x.lower()))
        except PermissionError:
            self._tree.insert("", tk.END,
                              values=("(Permission denied)", "", ""))
            return

        for name in entries:
            if name.startswith("."):
                continue
            full = os.path.join(self._current_dir, name)
            if os.path.isdir(full):
                self._tree.insert(
                    "", tk.END,
                    values=(f"\U0001f4c1  {name}", "Folder", ""),
                    tags=("dir",))
            elif self._mode == "file":
                if self._matches_filter(name):
                    try:
                        sz = os.path.getsize(full)
                        sz_str = self._format_size(sz)
                    except OSError:
                        sz_str = ""
                    ext = os.path.splitext(name)[1]
                    self._tree.insert(
                        "", tk.END,
                        values=(f"\U0001f4c4  {name}", ext or "File", sz_str),
                        tags=("file",))

    def _matches_filter(self, name: str) -> bool:
        if not self._filetypes:
            return True
        name_lower = name.lower()
        for _label, pattern in self._filetypes:
            if pattern == "*.*" or pattern == "*":
                return True
            for pat in pattern.split():
                ext = pat.lstrip("*").lower()
                if name_lower.endswith(ext):
                    return True
        return False

    @staticmethod
    def _format_size(n: int) -> str:
        if n < 1024:
            return f"{n} B"
        elif n < 1024 ** 2:
            return f"{n / 1024:.1f} KB"
        elif n < 1024 ** 3:
            return f"{n / 1024**2:.1f} MB"
        else:
            return f"{n / 1024**3:.2f} GB"

    # ==================================================================
    # Events
    # ==================================================================

    def _on_double_click(self, _event):
        sel = self._tree.selection()
        if not sel:
            return
        values = self._tree.item(sel[0], "values")
        tags = self._tree.item(sel[0], "tags")

        name = values[0]
        for prefix in ("\U0001f4c1  ", "\U0001f4c4  "):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        full = os.path.join(self._current_dir, name)

        if "dir" in tags:
            self._navigate_to(full)
        elif "file" in tags and self._mode == "file":
            self._result = full
            self._win.destroy()

    def _on_tree_select(self, _event):
        sel = self._tree.selection()
        if not sel:
            return
        values = self._tree.item(sel[0], "values")
        tags = self._tree.item(sel[0], "tags")

        name = values[0]
        for prefix in ("\U0001f4c1  ", "\U0001f4c4  "):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break

        full = os.path.join(self._current_dir, name)

        if self._mode == "directory" and "dir" in tags:
            self._selected_var.set(full)
        elif self._mode == "file" and "file" in tags:
            self._selected_var.set(full)
        elif self._mode == "directory":
            self._selected_var.set(self._current_dir)

    def _on_select(self):
        sel = self._selected_var.get().strip()
        if not sel:
            sel = self._current_dir

        if self._mode == "directory":
            self._result = sel
        elif self._mode == "file":
            if os.path.isfile(sel):
                self._result = sel
            else:
                return
        self._win.destroy()

    def _on_cancel(self):
        self._result = None
        self._win.destroy()

    def _create_folder(self):
        name = simpledialog.askstring(
            "New Folder", "Folder name:",
            parent=self._win)
        if not name:
            return
        name = name.strip().replace(os.sep, "_")
        new_path = os.path.join(self._current_dir, name)
        try:
            os.makedirs(new_path, exist_ok=True)
            self._refresh_listing()
            self._selected_var.set(new_path)
        except OSError as exc:
            from tkinter import messagebox
            messagebox.showerror("Error", f"Could not create folder:\n{exc}",
                                 parent=self._win)


# ==================================================================
# Convenience wrappers
# ==================================================================

def browse_directory(
    parent: tk.Tk,
    title: str = "Select Directory",
    initial_dir: str = "",
    allow_create_folder: bool = False,
    current_value: str = "",
) -> Optional[str]:
    dlg = FileBrowserDialog(
        parent, title=title, initial_dir=initial_dir,
        mode="directory", allow_create_folder=allow_create_folder,
        current_value=current_value,
    )
    return dlg.result


def browse_file(
    parent: tk.Tk,
    title: str = "Select File",
    initial_dir: str = "",
    filetypes: Optional[List[Tuple[str, str]]] = None,
    allow_create_folder: bool = False,
    current_value: str = "",
) -> Optional[str]:
    dlg = FileBrowserDialog(
        parent, title=title, initial_dir=initial_dir,
        mode="file", filetypes=filetypes,
        allow_create_folder=allow_create_folder,
        current_value=current_value,
    )
    return dlg.result