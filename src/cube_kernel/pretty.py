"""Pretty-printing utilities for cube kernel matrices."""

from __future__ import annotations

import sys

import numpy as np
from numpy.typing import NDArray

# Ensure Unicode box-drawing characters reach the terminal on Windows.
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

Matrix = NDArray[np.integer]

_W = 2  # chars per integer value; -1 is the widest we encounter


def _v(n: int) -> str:
    return str(int(n)).rjust(_W)


def _fmt_rows(m: np.ndarray) -> list[str]:
    return [" ".join(_v(x) for x in row) for row in m]


def _box(rows: list[str]) -> list[str]:
    w = max(len(r) for r in rows)
    top = "┌" + "─" * (w + 2) + "┐"
    bot = "└" + "─" * (w + 2) + "┘"
    body = ["│ " + r.ljust(w) + " │" for r in rows]
    return [top] + body + [bot]


def _mat_lines(m: np.ndarray) -> list[str]:
    return _box(_fmt_rows(m))


def _join_horizontal(blocks: list[list[str]], seps: list[str]) -> str:
    """Join blocks of lines side by side; each sep is shown at the vertical midpoint."""
    h = max(len(b) for b in blocks)
    mid = (h - 1) // 2
    widths = [max(len(l) for l in b) for b in blocks]
    padded = [list(b) + [""] * (h - len(b)) for b in blocks]
    lines = []
    for i in range(h):
        parts = []
        for j, block in enumerate(padded):
            if j > 0:
                sep = seps[j - 1]
                parts.append(sep if i == mid else " " * len(sep))
            parts.append(block[i].ljust(widths[j]))
        lines.append("".join(parts))
    return "\n".join(lines)


# ── public format functions ───────────────────────────────────────────────────

def fmt_mat(m: Matrix) -> str:
    """Format any matrix with a Unicode box border."""
    return "\n".join(_mat_lines(np.asarray(m)))


def fmt_equation(*matrices: Matrix, ops: list[str]) -> str:
    """Format matrices side by side with operators between them.

    Example: fmt_equation(T, S, R, ops=["@", "="])
    renders as  T  @  S  =  R
    """
    blocks = [_mat_lines(np.asarray(m)) for m in matrices]
    seps = [f"  {op}  " for op in ops]
    return _join_horizontal(blocks, seps)


def fmt_block_mat(m: Matrix, bs: int = 3) -> str:
    """Format an (N×N) matrix with bs×bs block-separating grid lines."""
    m = np.asarray(m)
    n = m.shape[0]
    nb = n // bs
    bw = (_W + 1) * bs - 1  # inner char-width of one block column

    top = "┌" + ("─" * (bw + 2) + "┬") * (nb - 1) + "─" * (bw + 2) + "┐"
    mid = "├" + ("─" * (bw + 2) + "┼") * (nb - 1) + "─" * (bw + 2) + "┤"
    bot = "└" + ("─" * (bw + 2) + "┴") * (nb - 1) + "─" * (bw + 2) + "┘"

    lines = [top]
    for r in range(n):
        if r > 0 and r % bs == 0:
            lines.append(mid)
        parts = [
            " " + " ".join(_v(m[r, cb * bs + c]) for c in range(bs)) + " "
            for cb in range(nb)
        ]
        lines.append("│" + "│".join(parts) + "│")
    lines.append(bot)
    return "\n".join(lines)


def fmt_cubie_state(state: Matrix) -> str:
    """Format a (3N×3) state as N labeled cubie blocks side by side."""
    state = np.asarray(state)
    n = state.shape[0] // 3
    blocks = []
    for i in range(n):
        label = f"Cubie {i}"
        mat_lines = _mat_lines(state[3 * i : 3 * i + 3])
        lw = max(len(label), max(len(l) for l in mat_lines))
        blocks.append([label.center(lw)] + [l.ljust(lw) for l in mat_lines])
    widths = [max(len(l) for l in b) for b in blocks]
    h = max(len(b) for b in blocks)
    padded = [list(b) + [""] * (h - len(b)) for b in blocks]
    lines = []
    for i in range(h):
        lines.append("     ".join(b[i].ljust(w) for b, w in zip(padded, widths)))
    return "\n".join(lines)


# ── print wrappers ────────────────────────────────────────────────────────────

def show(m: Matrix, label: str = "") -> None:
    """Print a matrix with an optional label."""
    if label:
        print(label)
    print(fmt_mat(m))


def show_equation(*matrices: Matrix, ops: list[str], label: str = "") -> None:
    """Print matrices with operators between them."""
    if label:
        print(label)
    print(fmt_equation(*matrices, ops=ops))


def show_block(m: Matrix, bs: int = 3, label: str = "") -> None:
    """Print a block matrix with grid lines."""
    if label:
        print(label)
    print(fmt_block_mat(m, bs))


def show_cubie_state(state: Matrix, label: str = "") -> None:
    """Print a (3N×3) cubie state."""
    if label:
        print(label)
    print(fmt_cubie_state(state))


def fmt_face_state(state: Matrix) -> str:
    """Format a (27×3) face state as a 3×3 grid of labeled cubie blocks."""
    state = np.asarray(state)
    row_strs = []
    for row_idx in range(3):
        blocks = []
        for col_idx in range(3):
            i = row_idx * 3 + col_idx
            label = f"Cubie {i}"
            mat_lines = _mat_lines(state[3 * i : 3 * i + 3])
            lw = max(len(label), max(len(l) for l in mat_lines))
            blocks.append([label.center(lw)] + [l.ljust(lw) for l in mat_lines])
        widths = [max(len(l) for l in b) for b in blocks]
        h = max(len(b) for b in blocks)
        padded = [list(b) + [""] * (h - len(b)) for b in blocks]
        row_strs.append(
            "\n".join("     ".join(b[i].ljust(w) for b, w in zip(padded, widths)) for i in range(h))
        )
    return "\n\n".join(row_strs)


def show_face_state(state: Matrix, label: str = "") -> None:
    """Print a (27×3) face state as a 3×3 grid of cubie blocks."""
    if label:
        print(label)
    print(fmt_face_state(state))
