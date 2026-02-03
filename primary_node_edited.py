#!/usr/bin/env python3
"""
primary_node.py

Primary coordinator that:
1) Maintains an in-memory registry of secondary nodes (registered by secondary_node.py)
2) Distributes prime-range computation requests to registered secondary nodes
3) Aggregates results in memory and returns a final result (count or list sample)

Endpoints
---------
GET  /health
GET  /nodes
POST /register
POST /compute
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse


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
            "last_seen": now,
            "expires_at": now + self.ttl_s,
        }
        with self.lock:
            self.nodes[node_id] = record
        return record

    def get_all(self) -> List[Dict[str, Any]]:
        now = time.time()
        with self.lock:
            active = []
            for nid in list(self.nodes.keys()):
                if self.nodes[nid]["expires_at"] < now:
                    del self.nodes[nid]
                else:
                    active.append(self.nodes[nid])
            return active


REGISTRY = Registry()


def _post_json(url: str, data: Dict[str, Any], timeout_s: float = 10.0) -> Dict[str, Any]:
    """Helper to perform a POST with a timeout to detect crashed nodes."""
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}, method="POST"
    )
    # Petrov Ch. 8: Timeouts are essential to distinguish between 'slow' and 'down'
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8"))


def split_into_slices(low: int, high: int, n: int) -> List[Tuple[int, int]]:
    if n <= 1:
        return [(low, high)]
    step = (high - low + 1) // n
    slices = []
    current = low
    for i in range(n):
        nxt = current + step - 1
        if i == n - 1:
            nxt = high
        slices.append((current, nxt))
        current = nxt + 1
    return slices


def distributed_compute(payload: Dict[str, Any]) -> Dict[str, Any]:
    low = int(payload.get("low", 0))
    high = int(payload.get("high", 0))
    mode = str(payload.get("mode", "count"))
    max_return_primes = int(payload.get("max_return_primes", 100))

    nodes = REGISTRY.get_all()
    if not nodes:
        return {"ok": False, "error": "no secondary nodes registered"}

    nodes_sorted = sorted(nodes, key=lambda n: n["node_id"])
    slices = split_into_slices(low, high, len(nodes_sorted))
    nodes_sorted = nodes_sorted[:len(slices)]

    t0 = time.perf_counter()
    total_primes = 0
    primes_sample: List[int] = []
    node_reports = []

    # Fault Tolerance Logic: Retry with Timeout
    def call_node_safe(node: Dict[str, Any], sl: Tuple[int, int]) -> Dict[str, Any]:
        url = f"http://{node['host']}:{node['port']}/compute"
        req_payload = {
            "low": sl[0],
            "high": sl[1],
            "mode": mode,
            "chunk": int(payload.get("chunk", 500_000)),
            "exec": str(payload.get("secondary_exec", "processes")),
            "max_return_primes": max_return_primes,
        }
        
        # Petrov Ch. 8: Handling crash failures via retries
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                return _post_json(url, req_payload, timeout_s=15.0)
            except Exception as e:
                if attempt == max_retries:
                    return {"ok": False, "error": f"Node failure after {max_retries} retries: {str(e)}"}
                time.sleep(0.5) # Small backoff

    with ThreadPoolExecutor(max_workers=len(nodes_sorted)) as executor:
        future_to_node = {
            executor.submit(call_node_safe, nodes_sorted[i], slices[i]): nodes_sorted[i]
            for i in range(len(nodes_sorted))
        }

        for future in as_completed(future_to_node):
            node = future_to_node[future]
            try:
                res = future.result()
                if res and res.get("ok"):
                    total_primes += res.get("count", 0)
                    primes_sample.extend(res.get("primes", []))
                    node_reports.append({"node_id": node["node_id"], "status": "success"})
                else:
                    node_reports.append({
                        "node_id": node["node_id"], 
                        "status": "failed", 
                        "error": res.get("error") if res else "No response"
                    })
            except Exception as e:
                node_reports.append({"node_id": node["node_id"], "status": "exception", "error": str(e)})

    # Return partial results if some nodes failed
    return {
        "ok": True,
        "total_count": total_primes,
        "primes": primes_sample[:max_return_primes] if mode == "list" else [],
        "execution_time_s": time.perf_counter() - t0,
        "system_health": {
            "nodes_attempted": len(nodes_sorted),
            "nodes_responded": sum(1 for r in node_reports if r["status"] == "success"),
            "details": node_reports
        }
    }


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, data: Dict[str, Any], code: int = 200):
        content = json.dumps(data).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        parts = urlparse(self.path)
        if parts.path == "/health":
            return self._send_json({"ok": True, "status": "primary up"})
        if parts.path == "/nodes":
            return self._send_json({"ok": True, "nodes": REGISTRY.get_all()})
        return self._send_json({"ok": False, "error": "not found"}, code=404)

    def do_POST(self):
        parts = urlparse(self.path)
        content_length = int(self.headers.get("Content-Length", 0))
        raw_data = self.rfile.read(content_length)
        try:
            payload = json.loads(raw_data)
        except:
            return self._send_json({"ok": False, "error": "invalid json"}, code=400)

        if parts.path == "/register":
            reg = REGISTRY.upsert(payload)
            return self._send_json({"ok": True, "registered": reg})

        if parts.path == "/compute":
            try:
                for k in ["low", "high", "mode"]:
                    if k not in payload:
                        raise ValueError(f"missing field: {k}")
                resp = distributed_compute(payload)
                return self._send_json(resp, code=200)
            except Exception as e:
                return self._send_json({"ok": False, "error": str(e)}, code=400)

        return self._send_json({"ok": False, "error": "not found"}, code=404)

    def log_message(self, fmt, *args):
        return


def main() -> None:
    ap = argparse.ArgumentParser(description="Primary coordinator for distributed prime computation.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9200)
    ap.add_argument("--ttl", type=int, default=3600)
    args = ap.parse_args()

    global REGISTRY
    REGISTRY = Registry(ttl_s=max(10, int(args.ttl)))

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[primary_node] listening on http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()