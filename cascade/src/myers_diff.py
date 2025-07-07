"""myers_diff.py
=================================
Efficient *Myers shortest-edit script* implementation operating on **lists**
of hashable objects (for our first use-case: *lines* of text).

The module offers two public helpers:

```
diff(a: Sequence[T], b: Sequence[T]) -> list[tuple[str, T]]
```
    Return a linearised edit script where each tuple consists of

    * an **operation tag** – one of
        * ``"equal"``   – the element comes from *both* sequences and is
                          unchanged,
        * ``"delete"``  – the element appears **only** in *a* (the *first*
                          sequence),
        * ``"insert"``  – the element appears **only** in *b* (the *second*
                          sequence).

    * the element itself.

and a tiny **CLI** that behaves similarly to the Unix `diff` utility but much
simpler – it prints the edit script using the classical *prefix notation*:

    "  "  unchanged line
    "- "  deletion (present only in *a*)
    "+ " insertion (present only in *b*)

The core algorithm follows Eugene W. Myers' 1986 paper *"An O(ND) Difference
Algorithm and Its Variations"*.  Its asymptotic complexity is

    • *time*   O((N + M) · D)
    • *memory* O(N + M + D²)

where

    N  = |a|,  M = |b|,  D = size of the shortest edit script.

The implementation is almost a textbook translation with two practical
adaptations:

1.  **Pythonic data structures** – we store the *frontier* `V` of the search
    as a dict mapping *diagonal* `k → max x` instead of a list because the
    index range grows with each `d` step (negative `k` values would require
    shifting).

2.  **Trace recording** – to reconstruct the edit path we keep a *snapshot* of
    `V` for every edit distance `d`.  This increases memory by O(D²) but makes
    the back-tracking logic simple and – for the file sizes anticipated in
    this repository – perfectly acceptable.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, TypeVar

# Optional heavy import – only when `--tokens` is requested.
try:
    from transformers import AutoTokenizer  # noqa: F401 – optional
except ModuleNotFoundError:  # pragma: no cover – transformers not installed
    AutoTokenizer = None  # type: ignore[assignment]


T = TypeVar("T", bound=object)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diff(a: Sequence[T], b: Sequence[T]) -> List[Tuple[str, T]]:
    """Return *linearised* Myers edit script mapping *a* → *b*.

    Each entry in the returned list is a tuple ``(tag, element)`` where

    * *tag* ∈ {``"equal"``, ``"delete"``, ``"insert"``}
    * *element* is the underlying sequence item (``a[i]`` or ``b[j]``).
    """

    # Reconstruct the *path* through the edit graph.
    path = _shortest_edit_path(a, b)

    # Linearise the path to a conventional edit script.
    script: List[Tuple[str, T]] = []
    for step in path:
        op, x, y = step  # type: ignore[misc]

        if op == "equal":
            script.append(("equal", a[x]))
        elif op == "delete":
            script.append(("delete", a[x]))
        elif op == "insert":
            script.append(("insert", b[y]))
        else:  # pragma: no cover – should never happen
            raise RuntimeError(f"Unknown operation tag: {op}")

    return script


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _main(argv: List[str]) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Myers diff")
    parser.add_argument("file1", type=Path, help="first file to compare")
    parser.add_argument("file2", type=Path, help="second file to compare")
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="diff at *token* level using a HuggingFace tokenizer",
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="HF model name whose tokenizer to use (token mode only)",
    )

    args = parser.parse_args(argv[1:])

    if not args.file1.exists() or not args.file2.exists():
        missing = [str(p) for p in (args.file1, args.file2) if not p.exists()]
        print(f"Error: file(s) not found: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)

    RED = "\033[31m"
    BLUE = "\033[34m"
    RESET = "\033[0m"

    if args.tokens:
        if AutoTokenizer is None:
            print("Error: 'transformers' package not available – cannot token diff.", file=sys.stderr)
            sys.exit(3)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                trust_remote_code=True,
                local_files_only=True,  # prefer local cache; avoid network if possible
            )
        except Exception as e:
            print("Error loading tokenizer:", e, file=sys.stderr)
            print("Ensure the tokenizer is cached locally or that internet access is available.", file=sys.stderr)
            sys.exit(4)

        text_a = args.file1.read_text(encoding="utf-8", errors="replace")
        text_b = args.file2.read_text(encoding="utf-8", errors="replace")

        ids_a: List[int] = tokenizer.encode(text_a, add_special_tokens=False)
        ids_b: List[int] = tokenizer.encode(text_b, add_special_tokens=False)

        # Compute diff
        start_t = time.perf_counter()
        script = diff(ids_a, ids_b)
        duration = time.perf_counter() - start_t

        # Cache decoded single-token strings (expensive for some tokenizers).
        id_cache: Dict[int, str] = {}

        def decode_single(tid: int) -> str:
            if tid not in id_cache:
                id_cache[tid] = tokenizer.decode([tid], skip_special_tokens=False)
            return id_cache[tid]

        tty = sys.stdout.isatty()

        def colour(text: str, c: str) -> str:
            return f"{c}{text}{RESET}" if tty else text

        # Output builder with grouping of consecutive insertions/deletions.
        out_chunks: List[str] = []

        current_group: str | None = None  # 'insert' | 'delete'
        group_tokens: List[str] = []

        def flush_group():
            nonlocal current_group, group_tokens
            if current_group is None:
                return
            content = "".join(group_tokens)
            if current_group == "insert":
                out_chunks.append(colour(f"[+{content}]", BLUE))
            else:  # delete
                out_chunks.append(colour(f"[-{content}]", RED))
            current_group = None
            group_tokens = []

        for tag, tid in script:
            tok_text = decode_single(tid)

            if tok_text == "":
                continue  # skip zero-length tokens

            # Handle newline token specially
            if tok_text == "\n":
                flush_group()
                if tag == "equal":
                    out_chunks.append("\n")
                elif tag == "insert":
                    out_chunks.append(colour("[+⏎]\n", BLUE))
                elif tag == "delete":
                    out_chunks.append(colour("[-⏎]\n", RED))
                continue

            if tag == "equal":
                flush_group()
                out_chunks.append(tok_text)
            elif tag in ("insert", "delete"):
                if current_group == tag:
                    group_tokens.append(tok_text)
                else:
                    flush_group()
                    current_group = tag
                    group_tokens = [tok_text]

        flush_group()

        sys.stdout.write("".join(out_chunks))

        print(f"[diff computed in {duration:.6f} s]", file=sys.stderr)

    else:
        # Line-based diff (previous behaviour)
        lines_a = args.file1.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        lines_b = args.file2.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

        t0 = time.perf_counter()
        script = diff(lines_a, lines_b)
        duration = time.perf_counter() - t0

        for tag, line in script:
            if tag == "equal":
                prefix = "  "
                colour = ""
            elif tag == "delete":
                prefix = "- "
                colour = RED
            elif tag == "insert":
                prefix = "+ "
                colour = BLUE
            else:  # pragma: no cover – unknown tag
                continue

            if sys.stdout.isatty() and colour:
                if line.endswith("\n"):
                    content, nl = line[:-1], "\n"
                else:
                    content, nl = line, ""
                sys.stdout.write(f"{colour}{prefix}{content}{RESET}{nl}")
            else:
                sys.stdout.write(prefix + line)

        print(f"\n[diff computed in {duration:.6f} s]", file=sys.stderr)


# ---------------------------------------------------------------------------
# Internal helpers – Myers algorithm
# ---------------------------------------------------------------------------


def _shortest_edit_path(a: Sequence[T], b: Sequence[T]):  # noqa: C901  (complexity)
    """Compute one *shortest* edit path for transforming *a* into *b*.

    The function returns the path as a list of *steps* in **reverse** order –
    i.e. from the *end* of both sequences back to the *origin (0, 0)*.  Each
    step is a triple ``(tag, x, y)`` with

    * *tag*   – ``"equal"``, ``"delete"``, or ``"insert"``
    * *x, y*  – coordinates *before* executing the step.
    """

    n = len(a)
    m = len(b)

    # Special-case empty inputs early – avoids edge-cases later.
    if n == 0 and m == 0:
        return []
    if n == 0:
        # All *insertions* from *b*.
        return [("insert", 0, j) for j in reversed(range(m))]
    if m == 0:
        return [("delete", i, 0) for i in reversed(range(n))]

    max_d = n + m

    # V maps diagonal k → furthest x on that diagonal for current d.
    v: Dict[int, int] = {0: 0}

    # We store V *snapshots* for each d so we can reconstruct the path later.
    trace: List[Dict[int, int]] = []

    for d in range(max_d + 1):
        # Keep a copy of the current frontier for later back-tracking.
        trace.append(v.copy())

        for k in range(-d, d + 1, 2):
            # Decide whether the current step is an insertion or deletion.
            if k == -d or (k != d and v.get(k - 1, 0) < v.get(k + 1, 0)):
                # Insertion (move down in the edit graph): (x stays) from k+1.
                x_start = v.get(k + 1, 0)
                op = "insert"  # We inserted b[y] to match a[x].
            else:
                # Deletion (move right): take k-1 diagonal.
                x_start = v.get(k - 1, 0) + 1
                op = "delete"  # Deleted a[x] to match b[y].

            y_start = x_start - k

            # Follow the snake (diagonal) – equal elements.
            x = x_start
            y = y_start
            while x < n and y < m and a[x] == b[y]:
                x += 1
                y += 1

            v[k] = x

            if x >= n and y >= m:
                # Reached the bottom-right corner – reconstruct path.
                return _backtrack(trace, a, b, x, y)

        # End for k – proceed to next d.

    # Should never get here – unless inputs are pathological.
    raise RuntimeError("Failed to find edit path – this should not happen.")


def _backtrack(
    trace: List[Dict[int, int]],
    a: Sequence[T],
    b: Sequence[T],
    x_end: int,
    y_end: int,
):
    """Reconstruct the *edit path* from recorded `trace` information.

    Returns a list of steps in *reverse* order (from end to start).  Each step
    is a tuple *(tag, x, y)* representing **coordinates *before* applying the
    operation**.  This convention simplifies translation into an edit script
    later.
    """

    path: List[Tuple[str, int, int]] = []
    x, y = x_end, y_end

    for d in reversed(range(len(trace))):
        v = trace[d]
        k = x - y

        # Determine whether the last step to (x, y) was a deletion or
        # insertion (unless d == 0 – then we are at the origin already).
        if d == 0:
            break

        if k == -d or (k != d and v.get(k - 1, 0) < v.get(k + 1, 0)):
            prev_k = k + 1
            prev_x = v[prev_k]
            prev_y = prev_x - prev_k
            op = "insert"
        else:
            prev_k = k - 1
            prev_x = v[prev_k] + 1
            prev_y = prev_x - prev_k
            op = "delete"

        # Follow snake backwards – these are "equal" steps.
        while x > prev_x and y > prev_y:
            x -= 1
            y -= 1
            path.append(("equal", x, y))

        # Record the edit operation itself.
        if op == "delete":
            x -= 1
            path.append(("delete", x, y))
        else:  # insert
            y -= 1
            path.append(("insert", x, y))

    # Finally, there might be a leading snake from (0, 0) if the first edit
    # distance d where we terminated is > 0.  Process the remaining equals.
    while x > 0 and y > 0:
        x -= 1
        y -= 1
        path.append(("equal", x, y))

    # Either x or y may still be > 0 if one sequence is a proper prefix of
    # the other.
    while x > 0:
        x -= 1
        path.append(("delete", x, y))

    while y > 0:
        y -= 1
        path.append(("insert", x, y))

    return list(reversed(path))


# ---------------------------------------------------------------------------
# Module entry-point when executed as script
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    _main(sys.argv)
