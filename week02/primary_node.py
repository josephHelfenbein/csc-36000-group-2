#!/usr/bin/env python3
"""
primary_node.py

Primary coordinator that:
1) Maintains an in-memory registry of secondary nodes (registered via gRPC)
2) Distributes prime-range computation requests to registered secondary nodes
3) Aggregates results in memory and returns a final result (count or list sample)

gRPC Services
-------------
CoordinatorService:
    RegisterNode - workers call this to join the cluster
    ListNodes - list active workers
    Compute - fan-out prime computation across workers
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent import futures
from typing import Any, Dict, List, Tuple

import grpc
import primes_pb2
import primes_pb2_grpc


class Registry:
    def __init__(self, ttl_s: int = 3600):
        self.ttl_s = ttl_s
        self.lock = threading.Lock()
        self.nodes: Dict[str, Dict[str, Any]] = {}

    def upsert(self, node: Dict[str, Any]) -> Dict[str, Any]:
        node_id = str(node["node_id"])
        now = time.time()
        record = {
            "node_id": node_id,
            "host": str(node["host"]),
            "port": int(node["port"]),
            "cpu_count": int(node.get("cpu_count", 1)),
            "last_seen": float(node.get("ts", now)),
            "registered_at": now,
        }
        with self.lock:
            if node_id in self.nodes:
                record["registered_at"] = self.nodes[node_id].get("registered_at", now)
            self.nodes[node_id] = record
            return record

    def active_nodes(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self.lock:
            stale = [nid for nid, rec in self.nodes.items() if (now - float(rec.get("last_seen", 0))) > self.ttl_s]
            for nid in stale:
                del self.nodes[nid]
            return list(self.nodes.values())
        
    # method to Remove failed nodes from the registry
    def remove(self, node_id: str) -> None:
        with self.lock:
            if node_id in self.nodes:
                del self.nodes[node_id]


REGISTRY = Registry(ttl_s=3600)


def split_into_slices(low: int, high: int, n: int) -> List[Tuple[int, int]]:
    if n <= 0:
        return []
    total = high - low
    base = total // n
    rem = total % n
    out = []
    start = low
    for i in range(n):
        size = base + (1 if i < rem else 0)
        end = start + size
        if start < end:
            out.append((start, end))
        start = end
    return out


MODE_MAP = {
    primes_pb2.COUNT: "count",
    primes_pb2.LIST: "list",
}

EXEC_MAP = {
    primes_pb2.SINGLE: "single",
    primes_pb2.THREADS: "threads",
    primes_pb2.PROCESSES: "processes",
}

MODE_TO_PB = {v: k for k, v in MODE_MAP.items()}
EXEC_TO_PB = {v: k for k, v in EXEC_MAP.items()}

class CoordinatorServicer(primes_pb2_grpc.CoordinatorServiceServicer):
    """gRPC implementation of the coordinator (formerly HTTP primary_node)."""

    def RegisterNode(self, request, context):
        record = REGISTRY.upsert({
            "node_id": request.node_id,
            "host": request.host,
            "port": request.port,
            "cpu_count": request.cpu_count,
        })
        print(f"[primary_node] Registered node: {request.node_id} "
              f"({request.host}:{request.port}, cpus={request.cpu_count})")
        return primes_pb2.RegisterNodeResponse(
            ok=True,
            node=primes_pb2.NodeInfo(
                node_id=record["node_id"],
                host=record["host"],
                port=record["port"],
                cpu_count=record["cpu_count"],
                last_seen=record["last_seen"],
            ),
        )

    def ListNodes(self, request, context):
        nodes = REGISTRY.active_nodes()
        nodes.sort(key=lambda n: n["node_id"])
        pb_nodes = [
            primes_pb2.NodeInfo(
                node_id=n["node_id"],
                host=n["host"],
                port=n["port"],
                cpu_count=n["cpu_count"],
                last_seen=n["last_seen"],
            )
            for n in nodes
        ]
        return primes_pb2.ListNodesResponse(nodes=pb_nodes)

    def Compute(self, request, context):
        low = request.low
        high = request.high

        if high <= low:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("high must be > low")
            return primes_pb2.ComputeResponse()

        mode_str = MODE_MAP.get(request.mode)
        if mode_str is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("mode must be COUNT or LIST")
            return primes_pb2.ComputeResponse()

        exec_str = EXEC_MAP.get(request.secondary_exec)
        if exec_str is None:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("secondary_exec must be SINGLE, THREADS, or PROCESSES")
            return primes_pb2.ComputeResponse()

        nodes = REGISTRY.active_nodes()
        if not nodes:
            context.set_code(grpc.StatusCode.FAILED_PRECONDITION)
            context.set_details("no active secondary nodes registered")
            return primes_pb2.ComputeResponse()

        chunk = request.chunk if request.chunk > 0 else 500_000
        sec_workers = request.secondary_workers if request.secondary_workers > 0 else 0
        max_return_primes = request.max_return_primes if request.max_return_primes > 0 else 5000
        include_per_node = request.include_per_node

        nodes_sorted = sorted(nodes, key=lambda n: n["node_id"])
        slices = split_into_slices(low, high, len(nodes_sorted))
        nodes_sorted = nodes_sorted[:len(slices)]

        t0 = time.perf_counter()

        per_node_results: List[Dict[str, Any]] = []
        failed_slices: List[Tuple[Dict[str, Any], Tuple[int, int]]] = []

        def call_node(node: Dict[str, Any], sl: Tuple[int, int]) -> Dict[str, Any]:
            host = node["host"]
            port = node["port"]
            channel = grpc.insecure_channel(f"{host}:{port}")
            stub = primes_pb2_grpc.WorkerServiceStub(channel)

            worker_req = primes_pb2.ComputeRangeRequest(
                low=sl[0],
                high=sl[1],
                mode=request.mode,
                chunk=chunk,
                exec=request.secondary_exec,
                workers=sec_workers,
                max_return_primes=max_return_primes if mode_str == "list" else 0,
            )

            t_call0 = time.perf_counter()
            resp = stub.ComputeRange(worker_req, timeout=3600)
            t_call1 = time.perf_counter()

            if not resp.ok:
                raise RuntimeError(f"node {node['node_id']} returned ok=False")

            node_elapsed_s = resp.elapsed_seconds
            print(f"Node ID: {node['node_id']} completed in: {node_elapsed_s:.4f}s")

            return {
                "node_id": node["node_id"],
                "low": sl[0],
                "high": sl[1],
                "round_trip_s": t_call1 - t_call0,
                "node_elapsed_s": node_elapsed_s,
                "total_primes": resp.total_primes,
                "max_prime": resp.max_prime,
                "primes": list(resp.primes),
                "primes_truncated": resp.primes_truncated,
            }

        with ThreadPoolExecutor(max_workers=min(32, len(nodes_sorted))) as ex:
            future_map = {
                ex.submit(call_node, node, sl): (node, sl)
                for node, sl in zip(nodes_sorted, slices)
            }
            for f in as_completed(future_map):
                node, sl = future_map[f]
                try:
                    per_node_results.append(f.result())
                except Exception as e:
                    print(f"[primary_node] Node {node['node_id']} failed: {e}")
                    REGISTRY.remove(node["node_id"])
                    failed_slices.append((node, sl))

        for _failed_node, sl in failed_slices:
            healthy_nodes = REGISTRY.active_nodes()
            if not healthy_nodes:
                continue
            retry_node = healthy_nodes[0]
            try:
                per_node_results.append(call_node(retry_node, sl))
            except Exception as e:
                print(f"[primary_node] Retry failed for slice {sl}: {e}")

        per_node_results.sort(key=lambda r: r["low"])

        total_primes = 0
        max_prime = -1
        primes_sample: List[int] = []
        primes_truncated = False

        for r in per_node_results:
            total_primes += int(r["total_primes"])
            max_prime = max(max_prime, int(r["max_prime"]))
            if mode_str == "list" and r.get("primes") is not None:
                ps = list(r["primes"])
                if len(primes_sample) < max_return_primes:
                    remaining = max_return_primes - len(primes_sample)
                    primes_sample.extend(ps[:remaining])
                    if len(ps) > remaining:
                        primes_truncated = True
                else:
                    primes_truncated = True
                if r.get("primes_truncated"):
                    primes_truncated = True

        t1 = time.perf_counter()

        pb_per_node = []
        if include_per_node:
            for r in per_node_results:
                pb_per_node.append(primes_pb2.PerNodeResult(
                    node_id=r["node_id"],
                    low=r["low"],
                    high=r["high"],
                    total_primes=r["total_primes"],
                    max_prime=r["max_prime"],
                    node_elapsed_s=r["node_elapsed_s"],
                    round_trip_s=r["round_trip_s"],
                ))

        return primes_pb2.ComputeResponse(
            ok=True,
            total_primes=total_primes,
            max_prime=max_prime,
            primes=primes_sample if mode_str == "list" else [],
            primes_truncated=primes_truncated,
            elapsed_seconds=t1 - t0,
            per_node=pb_per_node,
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="Primary coordinator for distributed prime computation (gRPC).")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--ttl", type=int, default=3600, help="Seconds to keep node registrations alive (default 3600).")
    args = ap.parse_args()

    global REGISTRY
    REGISTRY = Registry(ttl_s=max(10, int(args.ttl)))

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4))
    primes_pb2_grpc.add_CoordinatorServiceServicer_to_server(CoordinatorServicer(), server)
    server.add_insecure_port(f"{args.host}:{args.port}")
    server.start()

    print(f"[primary_node] gRPC listening on {args.host}:{args.port}")
    print("  gRPC services:")
    print("    RegisterNode()")
    print("    ListNodes()")
    print("    Compute()")

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        print("\n[primary_node] KeyboardInterrupt received; shutting down gracefully...", flush=True)
        server.stop(0)
    print("[primary_node] gRPC server stopped.")


if __name__ == "__main__":
    main()
