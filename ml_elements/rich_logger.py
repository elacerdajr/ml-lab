"""
ml_elements/rich_logger.py
--------------------------
Colourful, compact logger for experiments — human‑readable and LLM‑friendly.

Uses `rich` for ANSI styling while keeping output dense (one line per event,
consistent symbol prefixes, no emoji).  Falls back to plain `print()` if
`rich` is not installed.

Usage
-----
    from ml_elements.rich_logger import RichLogger

    log = RichLogger("weak_features_beta")

    log.section("n_train = 200  (1/6)")
    log.ok("catboost   fit=0.8s  cap=99")
    log.warn("figs   fit failed: ValueError")
    log.info("test_seed=%d", 10123)
    log.result("auc=0.927  ap=0.681  brier=0.142")

    with log.timer("DGP sampling"):
        dgp.sample(n, seed)
    # prints:   ⏱ DGP sampling  0.3s

    log.table(rows, columns=["n_train", "AUC", "AP"])
    log.done()
"""

from __future__ import annotations

import contextlib
import sys
import time as _time
from typing import Any, Iterator

# ── optional rich import ────────────────────────────────────────────────────
try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )
    from rich.table import Table
    from rich.text import Text

    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False
    Console = None  # type: ignore[assignment]
    Progress = None
    Table = None
    Text = None
    BarColumn = None
    SpinnerColumn = None
    TextColumn = None
    TimeElapsedColumn = None


# ── colour palette ──────────────────────────────────────────────────────────

_COLORS = {
    "ok": "green",
    "fail": "red",
    "warn": "yellow",
    "info": "bright_black",
    "result": "cyan",
    "section": "bold white",
    "timer": "magenta",
    "saved": "blue",
    "done": "bold green",
}


# ── RichLogger ──────────────────────────────────────────────────────────────


class RichLogger:
    """
    Compact, colourful experiment logger.

    Parameters
    ----------
    title : str or None
        If given, printed as a centred rule on construction.
    show_time : bool
        Prefix each message with ``[HH:MM:SS]``.
    """

    def __init__(self, title: str | None = None, show_time: bool = True) -> None:
        self._show_time = show_time
        if _HAS_RICH:
            self._console = Console(highlight=False, width=140)
        else:
            self._console = None
        if title:
            self.rule(title)

    # ── low‑level ──────────────────────────────────────────────────────

    def _print(self, prefix: str, color: str, msg: str, *args: Any) -> None:
        text = msg % args if args else msg
        ts = f"[dim]{_time.strftime('%H:%M:%S')}[/dim]  " if self._show_time else ""
        if _HAS_RICH:
            self._console.print(f"{ts}[{color}]{prefix}[/{color}] {text}")
        else:
            print(f"{ts}{prefix} {text}")

    def _plain(self, msg: str, *args: Any, color: str = "default") -> None:
        self._print("", color, msg, *args)

    # ── public API ─────────────────────────────────────────────────────

    def rule(self, title: str = "") -> None:
        """Horizontal rule, optionally with centred title text."""
        if _HAS_RICH:
            self._console.rule(title) if title else self._console.rule()
        elif title:
            width = 72
            pad = max(0, (width - len(title) - 2) // 2)
            print(f"{'─' * pad} {title} {'─' * pad}")

    def section(self, msg: str, *args: Any) -> None:
        """Major step / phase announcement (bold, padded)."""
        text = msg % args if args else msg
        line = f"━━ {text} ━━"
        if _HAS_RICH:
            self._console.print(f"[{_COLORS['section']}]{line}[/]")
        else:
            print(f"\n{line}")

    def step(self, msg: str, *args: Any) -> None:
        """Sub‑step marker (same as section but no padding)."""
        text = msg % args if args else msg
        self._print("━", _COLORS["section"], text, *())

    def ok(self, msg: str, *args: Any) -> None:
        """Success."""
        self._print("✔", _COLORS["ok"], msg, *args)

    def fail(self, msg: str, *args: Any) -> None:
        """Failure / error."""
        self._print("✖", _COLORS["fail"], msg, *args)

    def warn(self, msg: str, *args: Any) -> None:
        """Warning."""
        self._print("⚠", _COLORS["warn"], msg, *args)

    def info(self, msg: str, *args: Any) -> None:
        """Neutral / secondary information."""
        self._print("·", _COLORS["info"], msg, *args)

    def result(self, msg: str, *args: Any) -> None:
        """Headline result (bright, for metrics)."""
        self._print("→", _COLORS["result"], msg, *args)

    def saved(self, path: str, detail: str = "") -> None:
        """File‑saved confirmation."""
        msg = f"{path}" + (f"  ({detail})" if detail else "")
        self._print("✓", _COLORS["saved"], msg, *())

    def done(self, msg: str = "") -> None:
        """Completion banner."""
        self._print("✔", _COLORS["done"], msg if msg else "Done.", *())

    # ── structured output ──────────────────────────────────────────────

    @contextlib.contextmanager
    def timer(self, label: str) -> Iterator[None]:
        """Context manager that prints elapsed time on exit.

        >>> with log.timer("CatBoost fit"):
        ...     model.fit(X, y)
        # prints:   ⏱ CatBoost fit  0.8s
        """
        t0 = _time.perf_counter()
        try:
            yield
        finally:
            dt = _time.perf_counter() - t0
            self._print("⏱", _COLORS["timer"], f"{label}  {dt:.1f}s", *())

    @contextlib.contextmanager
    def progress(self, label: str, total: int | None = None) -> Iterator[Any]:
        """Context manager yielding a rich Progress bar.

        >>> with log.progress("Fitting", total=6) as p:
        ...     for name in models:
        ...         model.fit(...)
        ...         p.advance(p.task, advance=1)
        """
        if not _HAS_RICH:
            self.info("%s ...", label)
            yield _DummyProgress(total)
            return

        col_left = [SpinnerColumn(), TextColumn(f"[progress.description]{label}")]
        col_right = []
        if total is not None:
            col_right = [
                BarColumn(bar_width=20),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                "·",
                TimeElapsedColumn(),
            ]
        p = Progress(*col_left, *col_right, console=self._console, transient=False)
        with p:
            yield p

    def table(
        self,
        data: list[dict[str, Any]],
        *,
        columns: list[str] | None = None,
        title: str | None = None,
        caption: str | None = None,
    ) -> None:
        """Render a table from a list of dicts."""
        if not _HAS_RICH:
            if columns is None:
                columns = list(data[0].keys()) if data else []
            print("\t".join(columns))
            for row in data:
                print("\t".join(str(row.get(c, "")) for c in columns))
            return

        if columns is None:
            columns = list(data[0].keys()) if data else []

        tbl = Table(title=title, caption=caption)
        tbl.add_column("#", justify="right", style="dim")
        for c in columns:
            tbl.add_column(c)
        for i, row in enumerate(data, 1):
            tbl.add_row(str(i), *[str(row.get(c, "")) for c in columns])
        self._console.print(tbl)


# ── Dummy progress (no‑rich fallback) ───────────────────────────────────────


class _DummyProgress:
    def __init__(self, total: int | None) -> None:
        self.total = total

    def __enter__(self) -> _DummyProgress:
        self._count = 0
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    def advance(self, task: Any = None, advance: float = 1) -> None:
        self._count += advance
        sys.stdout.write(f"\r  ... {int(self._count)}/{self.total or '?'}")
        sys.stdout.flush()

    def update(self, task: Any = None, description: str = "") -> None:
        sys.stdout.write(f"\r  ... {description}")
        sys.stdout.flush()