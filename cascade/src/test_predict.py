import argparse
import asyncio
import sys
import time
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI


SYSTEM_MESSAGE = "repeat the following code verbatim, no markdown formatting and no other text"

GREEN = "\033[92m"
ORANGE = "\033[38;5;208m"
BLUE = "\033[94m"
RESET = "\033[0m"


@dataclass
class RequestMetrics:
    duration: float
    ttft: float
    output_time: float
    output_tokens: int
    accepted_tokens: int
    chunk_count: int
    tokens_per_sec: float


@dataclass
class RequestResult:
    request_index: int
    metrics: RequestMetrics


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark predicted outputs against a model")
    parser.add_argument("--source", required=True, help="Path to the source document")
    parser.add_argument(
        "--predict",
        default="0",
        help="Path to the predicted output document (defaults to literal '0')",
    )
    parser.add_argument("-n", type=int, default=1, dest="n", help="Number of concurrent requests")
    parser.add_argument("--model", default="gpt-4.1", help="Model name to query")
    parser.add_argument("--base_url", default="", help="Override the API base URL")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--show-stream",
        action="store_true",
        help="Print streamed tokens as they arrive",
    )
    args = parser.parse_args()
    if args.n < 1:
        parser.error("-n must be at least 1")
    return args


async def execute_request(
    request_index: int,
    client: AsyncOpenAI,
    model: str,
    messages: list[dict[str, str]],
    prediction_text: str,
    total_requests: int,
    verbose: bool,
    show_stream: bool,
) -> RequestResult:
    start_time = time.perf_counter()
    first_chunk_time: Optional[float] = None
    chunk_count = 0
    printed_raw = False
    stream_usage = None

    async with client.chat.completions.stream(
        model=model,
        messages=messages,
        prediction={"type": "content", "content": prediction_text},
        stream_options={"include_usage": True},
    ) as stream:
        async for event in stream:
            event_type = getattr(event, "type", None)
            if event_type == "chunk":
                chunk = getattr(event, "chunk", None)
                if chunk is None:
                    continue
                choices = getattr(chunk, "choices", None) or []
                if choices:
                    chunk_count += 1
                usage = getattr(chunk, "usage", None)
                if usage is not None:
                    stream_usage = usage
                continue

            if event_type != "content.delta":
                continue

            if first_chunk_time is None:
                first_chunk_time = time.perf_counter()
            text = getattr(event, "delta", "") or ""
            if not text:
                continue
            if verbose:
                if total_requests > 1:
                    prefix = f"[req {request_index} chunk {chunk_count}] "
                else:
                    prefix = f"[chunk {chunk_count}] "
                print(prefix + text.replace("\n", "\\n"), flush=True)
            elif show_stream and total_requests == 1:
                print(text, end="", flush=True)
                printed_raw = True

        completion = await stream.get_final_completion()

    if printed_raw:
        print()

    end_time = time.perf_counter()
    duration = end_time - start_time
    ttft = duration if first_chunk_time is None else first_chunk_time - start_time
    if ttft < 0:
        ttft = 0.0
    output_time = duration - ttft

    usage = stream_usage or completion.usage
    output_tokens = 0
    accepted_tokens = 0
    if usage is not None:
        output_tokens = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            accepted_value = getattr(details, "accepted_prediction_tokens", None)
            if accepted_value is not None:
                accepted_tokens = accepted_value

    if output_time > 0 and output_tokens:
        tokens_per_sec = output_tokens / output_time
    else:
        tokens_per_sec = 0.0

    metrics = RequestMetrics(
        duration=duration,
        ttft=ttft,
        output_time=output_time,
        output_tokens=output_tokens,
        accepted_tokens=accepted_tokens,
        chunk_count=chunk_count,
        tokens_per_sec=tokens_per_sec,
    )
    return RequestResult(request_index=request_index, metrics=metrics)


async def run_benchmark(
    args: argparse.Namespace,
    source_text: str,
    prediction_text: str,
) -> list[RequestResult]:
    client_kwargs: dict[str, str] = {}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = AsyncOpenAI(**client_kwargs)

    message_template = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": source_text},
    ]

    tasks: list[asyncio.Task[RequestResult]] = []
    try:
        for index in range(1, args.n + 1):
            messages = [dict(item) for item in message_template]
            task = asyncio.create_task(
                execute_request(
                    request_index=index,
                    client=client,
                    model=args.model,
                    messages=messages,
                    prediction_text=prediction_text,
                    total_requests=args.n,
                    verbose=args.verbose,
                    show_stream=args.show_stream,
                ))
            tasks.append(task)
        return await asyncio.gather(*tasks)
    finally:
        await client.close()


def print_request_stats(result: RequestResult) -> None:
    metrics = result.metrics
    output_tokens = metrics.output_tokens
    accepted_tokens = metrics.accepted_tokens
    percent_accepted = (accepted_tokens / output_tokens * 100.0) if output_tokens else 0.0
    percent_text = (
        "percent_accepted="
        f"{GREEN}{percent_accepted:.2f}%{RESET} "
        f"({GREEN}{accepted_tokens}{RESET}/{GREEN}{output_tokens}{RESET})"
    )
    tokens_per_sec_text = (
        "tokens/sec="
        f"{ORANGE}{metrics.tokens_per_sec:.3f}{RESET}"
    )
    output_time_text = (
        "output_time="
        f"{BLUE}{metrics.output_time:.3f}s{RESET}"
    )
    parts = [
        f"[req {result.request_index}]",
        percent_text,
        tokens_per_sec_text,
        output_time_text,
        f"duration={metrics.duration:.3f}s",
        f"ttft={metrics.ttft:.3f}s",
        f"num_output_tokens={output_tokens}",
        f"num_chunks={metrics.chunk_count}",
    ]
    print(" ".join(parts), flush=True)


def print_averages(results: list[RequestResult]) -> None:
    if not results:
        return
    count = len(results)
    total_output_time = sum(item.metrics.output_time for item in results)
    total_output_tokens = sum(item.metrics.output_tokens for item in results)
    total_accepted_tokens = sum(item.metrics.accepted_tokens for item in results)
    avg_output_time = total_output_time / count if count else 0.0
    avg_chunks = sum(item.metrics.chunk_count for item in results) / count
    avg_output_tokens = total_output_tokens / count if count else 0.0
    overall_tokens_sec = (
        total_output_tokens / total_output_time if total_output_time > 0 else 0.0
    )
    overall_percent_accepted = (
        total_accepted_tokens / total_output_tokens * 100.0 if total_output_tokens else 0.0
    )
    percent_text = (
        "percent_accepted="
        f"{GREEN}{overall_percent_accepted:.2f}%{RESET} "
        f"({GREEN}{total_accepted_tokens}{RESET}/{GREEN}{total_output_tokens}{RESET})"
    )
    tokens_per_sec_text = (
        "tokens/sec="
        f"{ORANGE}{overall_tokens_sec:.3f}{RESET}"
    )
    output_time_text = (
        "output_time="
        f"{BLUE}{avg_output_time:.3f}s{RESET}"
    )
    parts = [
        "[avg]",
        percent_text,
        tokens_per_sec_text,
        output_time_text,
        f"num_output_tokens={avg_output_tokens:.3f}",
        f"num_chunks={avg_chunks:.3f}",
    ]
    print(" ".join(parts), flush=True)


def main() -> None:
    args = parse_args()
    source_text = read_text(args.source)
    if args.predict == "0":
        prediction_text = "0"
    else:
        prediction_text = read_text(args.predict)

    try:
        results = asyncio.run(run_benchmark(args, source_text, prediction_text))
    except KeyboardInterrupt:
        print("Interrupted", file=sys.stderr)
        sys.exit(1)

    for result in results:
        print_request_stats(result)
    print_averages(results)


if __name__ == "__main__":
    main()
