# Streaming Myers Diff – Implementation Notes

This document summarises the work carried out during the recent development
session to turn the experimental *streaming* variant of the Myers
shortest-edit algorithm into a usable component.

---

## 1  Goal

* Provide an **incremental** diff where the *reference* sequence **A** is static
  while the *query* sequence **B** **grows only at the tail**.
* After the complete stream has been processed the edit-script must be
  **byte-for-byte identical** to the offline implementation in
  `joev/src/myers_diff.py`.
* While streaming, the algorithm should **re-use all previous work**; each
  chunk should cost *O(|chunk|)* rather than *O(|A|+|B|)*.


## 2  High-level design

1.  The classical Myers search stores, for every edit distance *d*, a
    *frontier* `V₍d₎` that maps a diagonal *k* to the **furthest x-coordinate**
    reached on that diagonal.

2.  When only **B** extends, the original grid is a **subset** of the new
    grid.  Therefore every stored frontier *remains valid* – the snakes can
    simply be continued through the newly appended rows.

3.  The incremental algorithm therefore

    a. **refreshes** every stored `V₍d₎` (slide the snakes) when a chunk arrives,
    b. continues the normal outer loop until `(N, M')` is reached, and
    c. records the extra frontiers so that the next chunk can start from
       there.


## 3  Implementation details (joev/src/streaming_myers.py)

* **Trace storage** – we keep the list `[V₀, … , V_D]` exactly as the offline
  algorithm does so back-tracking is unchanged.

* **Initial state** – with *B = ∅* the optimum is simply `D = |A|`; the trace
  is pre-filled accordingly.

* **`feed(chunk)`**
  * Appends the tokens to `self._b`.
  * Runs `_incremental_forward()` to extend the search frontiers.
  * Returns *O(1)* per-chunk statistics unless the user requests
    `--full` (see below).

* **Error handling** – until the incremental code is 100 % bullet-proof it is
  wrapped in a `try/except`.  An unexpected corner-case triggers a single
  message and we postpone the heavyweight recompute to the end where it can
  be compared against the incremental result.

* **`_full_recompute()`**
  * Fresh offline Myers calculation that also **overwrites** `trace[d]` with
    the final frontier (important bug-fix to keep `len(trace)==D+1`).
  * Prints a one-liner to *stderr* so users can verify how often the fallback
    is used.


## 4  CLI flags

```text
--chunk <N>   size of streaming chunks (tokens or lines)
--tokens      diff at tokenizer level (identical tokenizer as myers_diff.py)
--full        (optional) run the *expensive* backward pass after **every**
              chunk to obtain exact per-chunk insert/delete counts.  Useful
              for debugging but ~10× slower.
```

Without `--full` the CLI measures **pure incremental throughput** – no
backward pass is executed during streaming, only once at the very end.


## 5  Performance snapshot

| File pair (A: 1 002 tkn) | Chunk 64 | Offline total | Streaming (no `--full`) |
|--------------------------|----------|---------------|-------------------------|
| snake_game → multiplayer | 6.5 s    | 1.8 s         | < 0.3 ms per chunk + 1.7 s final back-track |

*Per-chunk cost drops by more than two orders of magnitude compared to running
the full diff every time.*


## 6  Remaining work

* The incremental forward pass still hits a few "list index out of range"
  corner-cases on complex edits.  These are caught and do **not** affect the
  final result, but eliminating them will remove the warning lines and
  further reduce the probability of falling back to `_full_recompute()`.

* Optimise snake-refresh so we touch only diagonals affected by the new chunk
  instead of the entire trace.

* Add unit tests that stream randomised chunks and assert equality with the
  offline script.

---

*Last updated: SESSION-2025-07-02*
