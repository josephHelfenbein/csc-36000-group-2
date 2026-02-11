#!/usr/bin/env python3
"""
secondary_node.py

A "secondary node" HTTP server that exposes prime-range computation via HTTP.

Key features
------------
- Exposes POST /compute with the same partitioning + thread/process execution model as primes_cli.py
- On startup, optionally registers itself with a primary coordinator (primary_node.py) via POST /register
  so the primary can discover and distribute work across all secondary nodes.

Endpoints
---------
GET  /health
    -> {"ok": true, "status": "healthy"}

GET  /info
    -> basic node metadata (host/port/node_id/cpu_count)

POST /compute
    JSON body:
    {
      "low": 0,                   (required)
      "high": 1000000,            (required; exclusive)
      "mode": "count"|"list",     default "count"
      "chunk": 500000,            default 500000
      "exec": "single"|"threads"|"processes", default "single"
      "workers": 8,               default cpu_count
      "max_return_primes": 5000,  default 5000 (only used when mode="list")
      "include_per_chunk": true   default false (summary only; avoids huge responses)
    }

Notes
-----
- Example of how to run from terminal: python3 week01/secondary_node.py --primary http://127.0.0.1:9200 --node-id kbrown
- For classroom demos, use mode="count" for big ranges to avoid large payloads.
- "threads" may not speed up CPU-bound work in CPython; "processes" usually will.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse
from primes_in_range import get_primes

# --- gRPC imports ---
import grpc
from concurrent import futures

# Import the compiled protobuf modules (generated from primes.proto)
import primes_pb2
import primes_pb2_grpc

# ----------------------------
# Partitioning helpers
# ----------------------------

def iter_ranges(low: int, high: int, chunk: int) -> List[Tuple[int, int]]:
    """Split [low, high) into contiguous chunks."""
    if chunk <= 0:
        raise ValueError("chunk must be > 0")
    out: List[Tuple[int, int]] = []
    x = low
    while x < high:
        y = min(x + chunk, high)
        out.append((x, y))
        x = y
    return out


def _work_chunk(args: Tuple[int, int, bool]) -> Dict[str, Any]:
    """
    Worker for one chunk.
    Returns dict for easy JSON serialization.
    """
    low, high, return_list = args

    t0 = time.perf_counter()
    res = get_primes(low, high, return_list=return_list)
    t1 = time.perf_counter()

    if return_list:
        primes = list(res)  # type: ignore[arg-type]
        return {
            "low": low,
            "high": high,
            "elapsed_s": t1 - t0,
            "prime_count": len(primes),
            "max_prime": primes[-1] if primes else -1,
            "primes": primes,
        }

    count = int(res)  # type: ignore[arg-type]
    return {
        "low": low,
        "high": high,
        "elapsed_s": t1 - t0,
        "prime_count": count,
        "max_prime": -1,  # not computed in count mode to avoid extra work
    }


def compute_partitioned(
    low: int,
    high: int,
    *,
    mode: str = "count",
    chunk: int = 500_000,
    exec_mode: str = "single",
    workers: int | None = None,
    max_return_primes: int = 5000,
    include_per_chunk: bool = False,
) -> Dict[str, Any]:
    """
    Perform partitioned computation over [low, high) using get_primes per chunk.
    """
    if high <= low:
        raise ValueError("high must be > low")
    if mode not in ("count", "list"):
        raise ValueError("mode must be 'count' or 'list'")
    if exec_mode not in ("single", "threads", "processes"):
        raise ValueError("exec must be single|threads|processes")

    if workers is None:
        workers = os.cpu_count() or 4
    workers = max(1, int(workers))

    ranges = iter_ranges(low, high, chunk)
    want_list = (mode == "list")

    t0 = time.perf_counter()
    chunk_results: List[Dict[str, Any]] = []

    if exec_mode == "single":
        for a, b in ranges:
            chunk_results.append(_work_chunk((a, b, want_list)))

    elif exec_mode == "threads":
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work_chunk, (a, b, want_list)) for a, b in ranges]
            for f in as_completed(futs):
                chunk_results.append(f.result())

    else:  # processes
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_work_chunk, (a, b, want_list)) for a, b in ranges]
            for f in as_completed(futs):
                chunk_results.append(f.result())

    t1 = time.perf_counter()

    chunk_results.sort(key=lambda d: int(d["low"]))
    total_primes = sum(int(d["prime_count"]) for d in chunk_results)
    sum_chunk = sum(float(d["elapsed_s"]) for d in chunk_results)

    primes_out: List[int] | None = None
    truncated = False
    max_prime = -1

    if want_list:
        primes_out = []
        for d in chunk_results:
            ps = d.get("primes") or []
            if ps:
                max_prime = max(max_prime, int(ps[-1]))
            if len(primes_out) < max_return_primes:
                remaining = max_return_primes - len(primes_out)
                primes_out.extend(ps[:remaining])
                if len(ps) > remaining:
                    truncated = True
            else:
                truncated = True

    response: Dict[str, Any] = {
        "ok": True,
        "mode": mode,
        "range": [low, high],
        "chunk": chunk,
        "exec": exec_mode,
        "workers": workers if exec_mode != "single" else 1,
        "chunks": len(ranges),
        "total_primes": total_primes,
        "max_prime": max_prime,
        "elapsed_seconds": t1 - t0,
        "sum_chunk_compute_seconds": sum_chunk,
    }

    if include_per_chunk:
        slim = []
        for d in chunk_results:
            slim.append({
                "low": d["low"],
                "high": d["high"],
                "elapsed_s": d["elapsed_s"],
                "prime_count": d["prime_count"],
                "max_prime": d.get("max_prime", -1),
            })
        response["per_chunk"] = slim

    if primes_out is not None:
        response["primes"] = primes_out
        response["primes_truncated"] = truncated
        response["max_return_primes"] = max_return_primes

    return response


# ----------------------------
# Registration with primary
# ----------------------------

def _guess_local_ip_for(primary_url: str) -> str:
    """
    Best-effort: pick the local IP used to reach the primary.
    Works well in a LAN lab environment.
    """
    try:
        u = urlparse(primary_url)
        host = u.hostname or "127.0.0.1"
        port = u.port or (443 if u.scheme == "https" else 80)
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((host, port))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int = 5) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def start_registration_loop_grpc(
    primary_host: str,
    primary_port: int,
    node_id: str,
    host: str,
    port: int,
    *,
    interval_s: int = 3600,
) -> None:
    """
    Background heartbeat: periodically register this worker to the primary via gRPC.
    """
    channel = grpc.insecure_channel(f"{primary_host}:{primary_port}")
    stub = primes_pb2_grpc.CoordinatorServiceStub(channel)

    def loop():
        while True:
            try:
                req = primes_pb2.RegisterNodeRequest(
                    node_id=node_id,
                    host=host,
                    port=port,
                    cpu_count=os.cpu_count() or 1,
                )
                resp = stub.RegisterNode(req)
                if resp.ok:
                    print(f"[registration] node_id={node_id} successfully registered at {time.strftime('%X')}")
                else:
                    print(f"[registration] node_id={node_id} registration failed: {resp}")
            except Exception as e:
                print(f"[registration] error registering node to primary: {e}")
            time.sleep(interval_s)

    th = threading.Thread(target=loop, daemon=True)
    th.start()


# ----------------------------
# HTTP server
# ----------------------------

NODE_META: Dict[str, Any] = {}


# ----------------------------
# gRPC Worker Service
# ----------------------------
class WorkerServicer(primes_pb2_grpc.WorkerServiceServicer):
    """gRPC worker implementation using compute_partitioned."""

    def ComputeRange(self, request, context):
        try:
            mode_map = {
                primes_pb2.Mode.COUNT: "count",
                primes_pb2.Mode.LIST: "list",
            }

            exec_map = {
                primes_pb2.ExecMode.SINGLE: "single",
                primes_pb2.ExecMode.THREADS: "threads",
                primes_pb2.ExecMode.PROCESSES: "processes",
            }

            resp = compute_partitioned(
                request.low,
                request.high,
                mode=mode_map.get(request.mode, "count"),
                chunk=request.chunk,
                exec_mode=exec_map.get(request.exec, "single"),
                workers=request.workers if request.workers > 0 else None,
                max_return_primes=request.max_return_primes,
                include_per_chunk=False,
            )

            return primes_pb2.ComputeRangeResponse(
                ok=resp.get("ok", False),
                total_primes=resp.get("total_primes", 0),
                max_prime=resp.get("max_prime", -1),
                primes=resp.get("primes", []),
                primes_truncated=resp.get("primes_truncated", False),
                elapsed_seconds=resp.get("elapsed_seconds", 0.0),
                sum_chunk_compute_seconds=resp.get("sum_chunk_compute_seconds", 0.0),
            )

        except Exception as e:
            context.set_details(str(e))
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            return primes_pb2.ComputeRangeResponse()

    def Health(self, request, context):
        return primes_pb2.HealthResponse(ok=True, status="healthy")


def main() -> None:
    ap = argparse.ArgumentParser(description="Secondary prime worker node (HTTP server).")
    ap.add_argument("--host", default="127.0.0.1", help="Bind host (default 127.0.0.1).")
    ap.add_argument("--port", type=int, default=9100, help="Bind port (default 9100).")
    ap.add_argument("--node-id", default=None, help="Optional stable node id (default: hostname).")

    ap.add_argument("--primary", default=None, help="Primary coordinator URL, e.g. http://134.74.160.1:9200")
    ap.add_argument("--public-host", default=None, help="Host/IP to advertise to primary (default: auto-detect).")
    ap.add_argument("--register-interval", type=int, default=60, help="Seconds between heartbeats (default 60).")

    args = ap.parse_args()

    node_id = args.node_id or os.uname().nodename

    advertised_host = args.public_host
    if args.primary and not advertised_host:
        advertised_host = _guess_local_ip_for(args.primary)
    if not advertised_host:
        advertised_host = "127.0.0.1"

    NODE_META.update({
        "node_id": node_id,
        "bind_host": args.host,
        "bind_port": args.port,
        "advertised_host": advertised_host,
        "advertised_port": args.port,
        "cpu_count": os.cpu_count() or 1,
        "registered_to": args.primary,
    })

    if args.primary:
        raw = args.primary
        if "://" not in raw:
            raw = f"http://{raw}"
        primary_parsed = urlparse(raw)
        primary_host = primary_parsed.hostname or "127.0.0.1"
        primary_port = primary_parsed.port or 9200  # default coordinator port
        start_registration_loop_grpc(
            primary_host,
            primary_port,
            node_id=node_id,
            host=advertised_host,
            port=args.port,
            interval_s=max(5, int(args.register_interval)),
        )


    # ----------------------------
    # gRPC server startup
    # ----------------------------
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4))
    primes_pb2_grpc.add_WorkerServiceServicer_to_server(WorkerServicer(), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()
    print(f"[secondary_node] node_id={node_id}")
    print(f"[secondary_node] gRPC listening on {args.host}:{args.port}")
    print(f"[secondary_node] advertised as {advertised_host}:{args.port}")
    if args.primary:
        print(f"[secondary_node] registering to primary: {args.primary}")

    # ----------------------------
    # Updated CLI messages
    # ----------------------------
    print("  gRPC services:")
    print("    ComputeRange()")
    print("    Health()")


    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[secondary_node] KeyboardInterrupt received; shutting down gracefully...", flush=True)
        server.stop(0)
    print("[secondary_node] gRPC server stopped.")

if __name__ == "__main__":
    main()
