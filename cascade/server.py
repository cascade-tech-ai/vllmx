#!/usr/bin/env python3
"""Start the vLLM OpenAI-compatible server with JoeV defaults.

This script is purely a *Python* wrapper (no `os.exec*`) around
`vllm.entrypoints.openai.api_server`.  It builds the same argument list we
would normally type on the shell and then calls the server’s internal
`run_server()` coroutine via *uvloop*.

You can pass **additional CLI options** to override the defaults, e.g.::

    python joev/server.py --port 8080 --disable-prefix-caching

You may also supply the *model name* as the very first positional argument::

    python joev/server.py mistralai/Mistral-7B-Instruct-v0.3

Any additional flags after that are forwarded verbatim to vLLM’s OpenAI
server and therefore allow access to the full feature set provided by upstream
vLLM.
"""

from __future__ import annotations

import sys
from typing import List

import uvloop  # type: ignore

# ---------------------------------------------------------------------------
# Default command-line arguments
# ---------------------------------------------------------------------------

DEFAULT_ARG_LIST: List[str] = [
    "--enable-prefix-caching",
    "--gpu-memory-utilization", "0.8",
    "--max-model-len", "8192",
    "--speculative-config",
    '{"method":"static_text","num_speculative_tokens":32}',
    "--model", "HuggingFaceTB/SmolLM2-360M-Instruct",
    "--served-model-name", "predict"
]


def main(argv: List[str] | None = None) -> None:  # noqa: D401 – CLI entry-point
    """Entry-point parsed by `python -m joev.server` or `python joev/server.py`."""

    if argv is None:
        argv = sys.argv[1:]

    # ------------------------------------------------------------------
    # Allow a **positional** first argument to override the default model.  If
    # the first element of *argv* does *not* start with a dash we interpret it
    # as the Hugging Face model identifier and transparently rewrite it to the
    # canonical ``--model MODEL`` form understood by vLLM.  This provides the
    # convenient short-hand::
    #
    #     joev/server.py mistralai/Mistral-7B-Instruct-v0.3 --port 8080
    #
    # while still supporting the explicit flag‐based syntax.
    # ------------------------------------------------------------------

    adjusted_argv: List[str]
    if argv and not argv[0].startswith("-"):
        adjusted_argv = ["--model", argv[0]] + argv[1:]
    else:
        adjusted_argv = argv

    # Compose the final argument list: built-in defaults first, then user
    # overrides so that duplicate options provided by the caller take
    # precedence (argparse keeps the *last* occurrence for ``action='store'``).
    cli_args = DEFAULT_ARG_LIST + adjusted_argv

    from vllm.entrypoints.openai.cli_args import make_arg_parser, FlexibleArgumentParser
    from vllm.entrypoints.openai.api_server import run_server

    parser = make_arg_parser(FlexibleArgumentParser())

    # Parse combined default + override flags.
    args = parser.parse_args(cli_args)

    # Spin up the async server using uvloop’s convenient helper.
    uvloop.run(run_server(args))


if __name__ == "__main__":
    main()
