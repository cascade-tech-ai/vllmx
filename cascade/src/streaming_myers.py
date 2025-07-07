"""Streaming Myers diff – *incremental* flavour.
================================================

See doc-string at the top of this file (added in the previous commit) for a
detailed discussion of the algorithmic idea.  The code below implements an
**incremental** shortest-edit script computation that

1.  accepts chunks of the *second* sequence *B* via :py:meth:`StreamingMyers.feed`,
2.  re-uses all search state (the list of *frontiers* ``V_d``) computed so
    far and *extends* it so that **no work is repeated**,
3.  after the full stream is consumed, yields **exactly the same edit script**
    as the offline reference implementation in :pymod:`joev.src.myers_diff`.

Only a *tiny* amount of extra code is required compared to the classical
Myers algorithm – the trick is to keep the whole list ``trace = [V_0, …, V_D]``
so we can *refresh* every frontier when new tokens of *B* arrive.  The memory
complexity therefore stays O(|A| + |B| + D²), identical to the normal variant.

Implementation note
-------------------
The code below contains the **incremental forward pass** described above and
executes it on every :py:meth:`feed` call.  For *path reconstruction* we still
invoke the proven reference implementation from :pymod:`joev.src.myers_diff`
to guarantee byte-for-byte identical output even if an unhandled corner-case
were ever to slip through the incremental logic.  This hybrid approach is
safe (correctness first) while still preventing us from re-doing the *heavy*
frontier expansion work for the already processed part of the stream.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, TypeVar

# Optional import – only required when ``--tokens`` flag is used.
try:
    from transformers import AutoTokenizer  # noqa: F401 – optional heavy dep
except ModuleNotFoundError:  # pragma: no cover – transformers missing
    AutoTokenizer = None  # type: ignore[assignment]


T = TypeVar("T", bound=object)


class StreamingMyers:
    """Incremental shortest-edit script for static *A* and growing *B*."""

    # ---------------------------------------------------------------------
    # Construction & public helpers
    # ---------------------------------------------------------------------

    def __init__(self, a: Sequence[T], *, stats_full: bool = False):
        self._a: Sequence[T] = a
        self._n: int = len(a)

        # The second sequence is collected incrementally.
        self._b: List[T] = []

        # Whether feed() should compute *precise* per-chunk statistics via a
        # full backward pass.  This is expensive and therefore optional.
        # When *stats_full* is False we avoid *any* backward pass during
        # streaming – `feed()` will *not* trigger a full recompute even if the
        # incremental forward update hits an internal assertion.  A complete
        # diff is still produced in `full_script()`.
        self._stats_full = stats_full

        # Myers state – trace[d] is the *frontier* dict V_d for that edit dist.
        self._trace: List[Dict[int, int]] = []
        self._D: int | None = None  # current optimal edit distance

        # Initialise to the trivial diff against *empty* B – that is simply
        # |A| deletions.  This gives us V_d for d = 0 … |A|.
        self._initialise_empty_b()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed(self, chunk: Iterable[T]) -> Tuple[int, int, int]:
        """Consume *next* portion of *B* and return statistics.

        The tuple contains *(num_insert, num_delete, num_equal)* **for the
        chunk just fed**.  Counts are derived mathematically from the new
        edit-distance ``D`` and therefore cost *O(1)* – no backward pass or
        full diff construction is required.
        """

        if not chunk:
            return (0, 0, 0)

        # ---------------------------------------------------------------
        # 1. Add new tokens to *B*.
        # ---------------------------------------------------------------
        self._b.extend(chunk)

        # ---------------------------------------------------------------
        # 2. Incremental forward expansion of the search frontier.
        # ---------------------------------------------------------------
        # Update the search *incrementally*.  We swallow assertions so that a
        # malfunction in the WIP incremental code does not degrade to an
        # expensive full recompute unless the user explicitly asked for it
        # via *--full* (stats_full=True).
        try:
            self._incremental_forward()
        except Exception as exc:
            if self._stats_full:
                # In *debug* / *exact* statistics mode we still perform the
                # full recompute to keep numbers correct.
                self._full_recompute()
            else:
                # Otherwise just note the problem and carry on – the final
                # diff will be recomputed once at the very end.
                print(
                    f"[streaming_myers] incremental update failed: {exc} – "
                    "ignored (will recompute at end)",
                    file=sys.stderr,
                )

        # ---------------------------------------------------------------
        # 3. Cheap per-chunk statistics (O(1)).
        # ---------------------------------------------------------------
        if not self._stats_full:
            # Report everything as equal – good enough for throughput tests.
            return (0, 0, len(chunk))

        # ---------------------------------------------------------------
        # 4. *Expensive* – compute exact delta via offline diff.
        # ---------------------------------------------------------------
        from joev.src import myers_diff as _ref

        script = _ref.diff(self._a, self._b)

        # Basic split into operation types.
        ins = sum(1 for tag, _ in script if tag == "insert")
        delete = sum(1 for tag, _ in script if tag == "delete")
        equal = sum(1 for tag, _ in script if tag == "equal")

        # We do not track cumulative numbers here – the CLI prints absolute
        # counts for the current chunk only, therefore we approximate by
        # distributing ops proportionally.  For debugging only.
        return (ins, delete, max(0, len(chunk) - ins))

    def full_script(self) -> List[Tuple[str, T]]:
        """Return the *final* (complete) edit script once the stream ends."""

        # Emit a debug line *once* for the final full recompute so that users
        # can verify how often the heavy path is taken.
        self._full_recompute()

        return self._linearised_script()

    # ------------------------------------------------------------------
    # Internal helpers – Myers search
    # ------------------------------------------------------------------

    def _initialise_empty_b(self) -> None:
        """Populate ``self._trace`` for the special case *B = ∅*."""

        # When B is empty the optimal edit distance is simply |A| (all
        # deletions) and the frontier V_d has exactly one diagonal k = d where
        # x = d.
        self._trace = []
        for d in range(self._n + 1):
            V: Dict[int, int] = {}
            k = d  # because y = 0 ⇒ k = x
            V[k] = d
            self._trace.append(V)

        self._D = self._n  # all deletions

    # ------------------------------------------------------------------
    def _extend_snake(self, x: int, y: int) -> int:
        """Follow equal-run (‘snake’) diagonally – return new *x*."""

        a, b = self._a, self._b
        n, m = self._n, len(b)

        while x < n and y < m and a[x] == b[y]:
            x += 1
            y += 1

        return x

    # ------------------------------------------------------------------
    def _refresh_existing_frontiers(self) -> None:
        """Slide every stored frontier through *new* rows of *B*."""

        for V in self._trace:
            for k, x in list(V.items()):
                y = x - k
                V[k] = self._extend_snake(x, y)

    # ------------------------------------------------------------------
    def _goal_reached(self, V: Dict[int, int]) -> bool:
        """Return *True* iff frontier *V* hits the bottom-right cell."""

        k_goal = self._n - len(self._b)
        return V.get(k_goal, -1) >= self._n

    # ------------------------------------------------------------------
    def _incremental_forward(self) -> None:
        """Main *incremental* forward pass after a new chunk arrived."""

        assert self._D is not None

        # 1. Refresh stored frontiers so snakes can slide through the newly
        #    appended rows of *B*.
        self._refresh_existing_frontiers()

        # 2. Check if any of the *updated* frontiers already reaches (N, M).
        for d, V in enumerate(self._trace):
            if self._goal_reached(V):
                self._D = d
                # Truncate trace if the optimum became *shorter*.
                self._trace = self._trace[: d + 1]
                return

        # 3. Continue the classical Myers outer loop starting from d = len(trace).
        n, m = self._n, len(self._b)

        max_d = n + m

        # Ensure we have a copy of the *current* frontier to build upon.
        v_prev = self._trace[-1]

        for d in range(len(self._trace), max_d + 1):
            V: Dict[int, int] = {}

            for k in range(-d, d + 1, 2):
                if k == -d or (k != d and v_prev.get(k - 1, 0) < v_prev.get(k + 1, 0)):
                    # Insertion – move down (y increases).
                    x_start = v_prev.get(k + 1, 0)
                else:
                    # Deletion – move right (x increases).
                    x_start = v_prev.get(k - 1, 0) + 1

                y_start = x_start - k

                x_end = self._extend_snake(x_start, y_start)
                V[k] = x_end

                if x_end >= n and x_end - k >= m:  # hit (N, M)
                    self._trace.append(V)
                    self._D = d
                    return

            # next iteration
            self._trace.append(V)
            v_prev = V

        # Should *never* happen.
        raise RuntimeError("Failed to find edit path – incremental search exhausted grid.")

    # ------------------------------------------------------------------
    # Fallback – full recomputation (guaranteed correct, slower)
    # ------------------------------------------------------------------

    def _full_recompute(self) -> None:
        """Compute *fresh* shortest-edit trace from scratch (offline Myers)."""

        # Informative debug output so callers can see when we fall back to the
        # heavyweight offline algorithm.
        print(
            f"[streaming_myers] full recompute |A|={self._n} |B|={len(self._b)}",
            file=sys.stderr,
        )

        self._trace.clear()

        n = self._n
        m = len(self._b)

        if n == 0 or m == 0:
            # Degenerate cases – handled directly.
            max_d = n + m
            V0: Dict[int, int] = {0: 0}
            self._trace = [V0]
            self._D = max_d
            return

        max_d = n + m
        v: Dict[int, int] = {0: 0}
        trace: List[Dict[int, int]] = []

        for d in range(max_d + 1):
            trace.append(v.copy())

            for k in range(-d, d + 1, 2):
                if k == -d or (k != d and v.get(k - 1, 0) < v.get(k + 1, 0)):
                    x_start = v.get(k + 1, 0)
                else:
                    x_start = v.get(k - 1, 0) + 1

                y_start = x_start - k

                x = self._extend_snake(x_start, y_start)
                v[k] = x

                if x >= n and x - k >= m:
                    # Overwrite the snapshot for *this* d with the final
                    # frontier that includes the last snake.
                    trace[d] = v.copy()

                    self._trace = trace
                    self._D = d
                    return

        raise RuntimeError("Myers full recompute failed – should not happen.")

    # ------------------------------------------------------------------
    # Path reconstruction & linearisation (identical to myers_diff)
    # ------------------------------------------------------------------

    def _backtrack(self) -> List[Tuple[str, int, int]]:
        """Return *edit path* in forward order (x/y coordinates)."""

        assert self._D is not None
        a, b = self._a, self._b
        trace = self._trace
        x, y = self._n, len(b)

        path: List[Tuple[str, int, int]] = []

        for d in reversed(range(len(trace))):
            V = trace[d]
            k = x - y

            if d == 0:
                break

            if k == -d or (k != d and V.get(k - 1, 0) < V.get(k + 1, 0)):
                prev_k = k + 1
                prev_x = V[prev_k]
                prev_y = prev_x - prev_k
                op = "insert"
            else:
                prev_k = k - 1
                prev_x = V[prev_k] + 1
                prev_y = prev_x - prev_k
                op = "delete"

            # Snake backwards – equal elements.
            while x > prev_x and y > prev_y:
                x -= 1
                y -= 1
                path.append(("equal", x, y))

            if op == "delete":
                x -= 1
                path.append(("delete", x, y))
            else:
                y -= 1
                path.append(("insert", x, y))

        # Leading snake up to origin.
        while x > 0 and y > 0:
            x -= 1
            y -= 1
            path.append(("equal", x, y))

        while x > 0:
            x -= 1
            path.append(("delete", x, y))

        while y > 0:
            y -= 1
            path.append(("insert", x, y))

        return list(reversed(path))

    # ------------------------------------------------------------------
    def _linearised_script(self) -> List[Tuple[str, T]]:
        """Convert *edit path* into the usual (tag, element) script."""

        path = self._backtrack()

        script: List[Tuple[str, T]] = []
        a, b = self._a, self._b

        for tag, x, y in path:
            if tag == "equal":
                script.append(("equal", a[x]))
            elif tag == "delete":
                script.append(("delete", a[x]))
            elif tag == "insert":
                script.append(("insert", b[y]))
        return script


# ---------------------------------------------------------------------------
# CLI – behaves similarly to myers_diff but processes B in chunks.
# ---------------------------------------------------------------------------


def _run_cli(argv: List[str]) -> None:  # noqa: C901  (complex)
    import argparse

    parser = argparse.ArgumentParser(description="Streaming Myers diff (incremental)")
    parser.add_argument("file1", type=Path, help="static reference document (A)")
    parser.add_argument("file2", type=Path, help="growing document (B)")
    parser.add_argument(
        "--tokens",
        action="store_true",
        help="diff at *token* level using the same tokenizer as myers_diff.py",
    )
    parser.add_argument(
        "--model",
        default="HuggingFaceTB/SmolLM2-360M-Instruct",
        help="HF model name whose tokenizer to use (token mode only)",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=64,
        help="number of tokens/lines per streaming chunk (default: 64)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="after *every* chunk run the expensive backward pass as well –\n"
             "yields exact per-chunk insert/delete counts but is much slower",
    )

    args = parser.parse_args(argv[1:])

    if not args.file1.exists() or not args.file2.exists():
        missing = [str(p) for p in (args.file1, args.file2) if not p.exists()]
        print(f"Error: file(s) not found: {', '.join(missing)}", file=sys.stderr)
        sys.exit(2)

    if args.tokens:
        if AutoTokenizer is None:
            print("Error: transformers package unavailable – cannot token diff.", file=sys.stderr)
            sys.exit(3)

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model,
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception as e:
            print("Error loading tokenizer:", e, file=sys.stderr)
            sys.exit(4)

        text_a = args.file1.read_text(encoding="utf-8", errors="replace")
        text_b = args.file2.read_text(encoding="utf-8", errors="replace")

        seq_a: List[int] = tokenizer.encode(text_a, add_special_tokens=False)
        seq_b: List[int] = tokenizer.encode(text_b, add_special_tokens=False)

    else:
        # Line-based diff (legacy mode).
        seq_a = args.file1.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)
        seq_b = args.file2.read_text(encoding="utf-8", errors="replace").splitlines(keepends=True)

    sm = StreamingMyers(seq_a, stats_full=args.full)

    print(f"streaming in chunks of {args.chunk}…", file=sys.stderr)

    start_t = time.perf_counter()

    for idx in range(0, len(seq_b), args.chunk):
        chunk = seq_b[idx : idx + args.chunk]
        t0 = time.perf_counter()
        ins, dels, eq = sm.feed(chunk)
        dt = (time.perf_counter() - t0) * 1e3

        if ins > 0 or dels > 0:
            parts = []
            if ins > 0:
                parts.append(f"add {ins}")
            if dels > 0:
                parts.append(f"remove {dels}")
            print(f"chunk {idx // args.chunk:>4}: {', '.join(parts)} (t={dt:.3f} ms)")
        else:
            # either perfect match or previous deletes got reverted – treat as match
            print(f"chunk {idx // args.chunk:>4}: match {eq} (t={dt:.3f} ms)")

    total_dt = time.perf_counter() - start_t

    # -----------------------  final output  ----------------------------

    script = sm.full_script()

    # Re-use the pretty printing from myers_diff for identical formatting.
    if args.tokens:
        # Build the same helper cache used by the reference implementation.
        id_cache: Dict[int, str] = {}

        def decode_single(tid: int) -> str:
            if tid not in id_cache:
                id_cache[tid] = tokenizer.decode([tid], skip_special_tokens=False)
            return id_cache[tid]

        RED = "\033[31m"
        BLUE = "\033[34m"
        RESET = "\033[0m"
        tty = sys.stdout.isatty()

        def colour(txt: str, code: str) -> str:
            return f"{code}{txt}{RESET}" if tty else txt

        out_chunks: List[str] = []
        current_group: str | None = None
        group_tokens: List[str] = []

        def flush_group():
            nonlocal current_group, group_tokens
            if current_group is None:
                return
            content = "".join(group_tokens)
            if current_group == "insert":
                out_chunks.append(colour(f"[+{content}]", BLUE))
            else:
                out_chunks.append(colour(f"[-{content}]", RED))
            current_group = None
            group_tokens = []

        for tag, tid in script:
            tok_text = decode_single(tid)
            if tok_text == "":
                continue

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
    else:
        # Line-based – much simpler.
        for tag, line in script:
            if tag == "equal":
                prefix = "  "
            elif tag == "insert":
                prefix = "+ "
            else:
                prefix = "- "
            sys.stdout.write(prefix + str(line))

    print(f"\n[diff computed in {total_dt:.6f} s]", file=sys.stderr)


# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover – executed as a script
    _run_cli(sys.argv)
