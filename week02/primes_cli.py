#!/usr/bin/env python3
"""
primes_cli.py

Notes
-----
- Examples of how to run from terminal: 
python3 week02/primes_cli.py --low 0 --high 100_000_0000 --exec single --time --mode count
python3 week02/primes_cli.py --low 0 --high 100_000_0000 --exec threads --time --mode count
python3 week02/primes_cli.py --low 0 --high 100_000_0000 --exec processes --time --mode count
python3 week02/primes_cli.py --low 0 --high 100_000_0000 --exec distributed --time --mode count --secondary-exec processes --primary 127.0.0.1:50051
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import List, Tuple
from urllib.parse import urlparse
from primes_in_range import get_primes

# gRPC imports
import grpc
import primes_pb2
import primes_pb2_grpc


def iter_ranges(low: int, high: int, chunk: int) -> List[Tuple[int, int]]:
    """Split [low, high) into contiguous chunks."""
    if chunk <= 0:
        raise ValueError("--chunk must be > 0")
    out: List[Tuple[int, int]] = []
    x = low
    while x < high:
        y = min(x + chunk, high)
        out.append((x, y))
        x = y
    return out


def _work_chunk(args: Tuple[int, int, bool]) -> Tuple[int, int, object]:
    a, b, return_list = args
    res = get_primes(a, b, return_list=return_list)
    return (a, b, res)


def _parse_primary_target(primary: str) -> str:
    """
    Accept either 'host:port' or 'http://host:port' and return 'host:port'
    suitable for grpc.insecure_channel().
    """
    if "://" in primary:
        parsed = urlparse(primary)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 9200
        return f"{host}:{port}"
    return primary


# Enum conversion helpers
MODE_TO_PB = {"count": primes_pb2.COUNT, "list": primes_pb2.LIST}
EXEC_TO_PB = {"single": primes_pb2.SINGLE, "threads": primes_pb2.THREADS, "processes": primes_pb2.PROCESSES}


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Prime counting/listing over [low, high) using local threads/processes OR distributed secondary nodes."
    )
    ap.add_argument("--low", type=int, required=True, help="Range start (inclusive).")
    ap.add_argument("--high", type=int, required=True, help="Range end (exclusive). Must be > low.")
    ap.add_argument("--mode", choices=["list", "count"], default="count")
    ap.add_argument("--chunk", type=int, default=500_000)
    ap.add_argument("--exec", choices=["single", "threads", "processes", "distributed"], default="single")
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 4))
    ap.add_argument("--max-print", type=int, default=50)
    ap.add_argument("--time", action="store_true")

    # Distributed options
    ap.add_argument("--primary", default=None, help="Primary coordinator address, e.g. 127.0.0.1:9200 or http://127.0.0.1:9200")
    ap.add_argument("--secondary-exec", choices=["single", "threads", "processes"], default="processes")
    ap.add_argument("--secondary-workers", type=int, default=None)
    ap.add_argument("--include-per-node", action="store_true")
    ap.add_argument("--max-return-primes", type=int, default=5000)

    args = ap.parse_args(argv)

    if args.high <= args.low:
        print("Error: --high must be > --low", file=sys.stderr)
        return 2

    if args.primary and args.exec != "distributed":
        print("Warning: --primary is set but --exec is not 'distributed'; ignoring --primary and running locally.",
              file=sys.stderr)

    return_list = (args.mode == "list")

    if args.exec == "distributed":
        if not args.primary:
            print("Error: --primary is required when --exec distributed", file=sys.stderr)
            return 2

        t0 = time.perf_counter()

        # gRPC coordinator call
        primary_target = _parse_primary_target(args.primary)
        channel = grpc.insecure_channel(primary_target)
        stub = primes_pb2_grpc.CoordinatorServiceStub(channel)

        try:
            grpc_req = primes_pb2.ComputeRequest(
                low=args.low,
                high=args.high,
                mode=MODE_TO_PB[args.mode],
                chunk=args.chunk,
                secondary_exec=EXEC_TO_PB[args.secondary_exec],
                secondary_workers=args.secondary_workers or 0,
                max_return_primes=args.max_return_primes,
                include_per_node=args.include_per_node,
            )

            grpc_resp = stub.Compute(grpc_req, timeout=3600)

        except grpc.RpcError as e:
            print(f"Distributed gRPC error: {e.code().name}: {e.details()}", file=sys.stderr)
            return 1

        t1 = time.perf_counter()

        if not grpc_resp.ok:
            print(f"Distributed error: coordinator returned ok=False", file=sys.stderr)
            return 1

        if args.mode == "count":
            print(grpc_resp.total_primes)
        else:
            primes = list(grpc_resp.primes)
            total = grpc_resp.total_primes
            shown = primes[: args.max_print]
            print(f"Total primes: {total}")
            print(f"First {len(shown)} primes (from returned sample):")
            print(" ".join(map(str, shown)))
            if grpc_resp.primes_truncated or total > len(primes):
                print(f"... (returned primes are capped at {args.max_return_primes})")

        if args.time:
            print(
                f"Elapsed seconds: {t1 - t0:.6f}  "
                f"(exec=distributed, secondary_exec={args.secondary_exec}, chunk={args.chunk})",
                file=sys.stderr,
            )
            if args.include_per_node and grpc_resp.per_node:
                print("Per-node summary:", file=sys.stderr)
                for r in grpc_resp.per_node:
                    print(
                        f"  {r.node_id:>12} slice=[{r.low}, {r.high}) primes={r.total_primes} "
                        f"node_elapsed={r.node_elapsed_s:.3f}s round_trip={r.round_trip_s:.3f}s",
                        file=sys.stderr,
                    )
        return 0

    # Local paths
    ranges = iter_ranges(args.low, args.high, args.chunk)
    t0 = time.perf_counter()
    results: List[Tuple[int, int, object]] = []

    if args.exec == "single":
        for a, b in ranges:
            results.append(_work_chunk((a, b, return_list)))

    elif args.exec == "threads":
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_work_chunk, (a, b, return_list)) for a, b in ranges]
            for f in as_completed(futs):
                results.append(f.result())

    else:  # processes
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_work_chunk, (a, b, return_list)) for a, b in ranges]
            for f in as_completed(futs):
                results.append(f.result())

    t1 = time.perf_counter()
    results.sort(key=lambda x: x[0])

    if args.mode == "count":
        total = 0
        for _, _, res in results:
            total += int(res)  # type: ignore[arg-type]
        print(total)
    else:
        all_primes: List[int] = []
        for _, _, res in results:
            all_primes.extend(list(res))  # type: ignore[arg-type]
        total = len(all_primes)
        shown = all_primes[: args.max_print]
        print(f"Total primes: {total}")
        print(f"First {len(shown)} primes:")
        print(" ".join(map(str, shown)))
        if total > len(shown):
            print(f"... ({total - len(shown)} more not shown)")

    if args.time:
        print(
            f"Elapsed seconds: {t1 - t0:.6f}  "
            f"(exec={args.exec}, workers={args.workers if args.exec!='single' else 1}, chunks={len(ranges)}, chunk_size={args.chunk})",
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

