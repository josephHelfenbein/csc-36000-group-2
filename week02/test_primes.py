#!/usr/bin/env python3
"""
test_primes.py

Run with:
    cd week02 && python -m pytest test_primes.py -v
"""

from __future__ import annotations

import os
import sys
import time
import threading
from concurrent import futures

import pytest
import grpc

sys.path.insert(0, os.path.dirname(__file__))

import primes_pb2
import primes_pb2_grpc
from primes_in_range import get_primes
from secondary_node import compute_partitioned, WorkerServicer
from primary_node import CoordinatorServicer, Registry, REGISTRY

def _start_grpc_server(servicer, add_fn):
    """Start a gRPC server on a random port. Returns (server, port)."""
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    add_fn(servicer, server)
    port = server.add_insecure_port("[::]:0")
    server.start()
    return server, port

class TestUnitPrimes:
    """Pure-logic tests using get_primes and compute_partitioned."""

    def test_count_known_range(self):
        """get_primes(0, 100) in count mode should return 25."""
        result = get_primes(0, 100, return_list=False)
        assert result == 25

    def test_list_known_range(self):
        """get_primes(0, 20) in list mode should return [2,3,5,7,11,13,17,19]."""
        result = get_primes(0, 20, return_list=True)
        assert result == [2, 3, 5, 7, 11, 13, 17, 19]

    def test_list_truncation_flag(self):
        """compute_partitioned with max_return_primes=5 should truncate."""
        resp = compute_partitioned(
            0, 100, mode="list", max_return_primes=5, exec_mode="single",
        )
        assert resp["primes_truncated"] is True
        assert len(resp["primes"]) == 5
        # Should be the first 5 primes
        assert resp["primes"] == [2, 3, 5, 7, 11]

    def test_bad_input_high_le_low(self):
        """compute_partitioned(100, 50) should raise ValueError."""
        with pytest.raises(ValueError, match="high must be > low"):
            compute_partitioned(100, 50)


class TestGRPCErrors:
    @pytest.fixture(autouse=True)
    def _coordinator_server(self):
        import primary_node

        self._original_registry = primary_node.REGISTRY
        primary_node.REGISTRY = Registry(ttl_s=300)

        server, port = _start_grpc_server(
            CoordinatorServicer(),
            primes_pb2_grpc.add_CoordinatorServiceServicer_to_server,
        )
        self.channel = grpc.insecure_channel(f"localhost:{port}")
        self.stub = primes_pb2_grpc.CoordinatorServiceStub(self.channel)
        yield
        server.stop(0)
        self.channel.close()
        primary_node.REGISTRY = self._original_registry

    def test_no_active_workers_error(self):
        """Compute with no workers → FAILED_PRECONDITION."""
        with pytest.raises(grpc.RpcError) as exc_info:
            self.stub.Compute(primes_pb2.ComputeRequest(
                low=0, high=100,
                mode=primes_pb2.COUNT,
                secondary_exec=primes_pb2.PROCESSES,
            ))
        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        assert "no active secondary nodes" in exc_info.value.details()

    def test_bad_range_error_via_grpc(self):
        """Compute(high=0, low=100) → INVALID_ARGUMENT."""
        with pytest.raises(grpc.RpcError) as exc_info:
            self.stub.Compute(primes_pb2.ComputeRequest(
                low=100, high=0,
                mode=primes_pb2.COUNT,
                secondary_exec=primes_pb2.PROCESSES,
            ))
        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert "high must be > low" in exc_info.value.details()

class TestIntegration:
    """Full end-to-end test with real gRPC servers."""

    @pytest.fixture(autouse=True)
    def _cluster(self):
        """Start 1 primary + 2 workers, register workers, tear down after."""
        import primary_node

        self._original_registry = primary_node.REGISTRY
        primary_node.REGISTRY = Registry(ttl_s=300)

        self.primary_server, self.primary_port = _start_grpc_server(
            CoordinatorServicer(),
            primes_pb2_grpc.add_CoordinatorServiceServicer_to_server,
        )

        self.worker_servers = []
        self.worker_ports = []
        for i in range(2):
            srv, port = _start_grpc_server(
                WorkerServicer(),
                primes_pb2_grpc.add_WorkerServiceServicer_to_server,
            )
            self.worker_servers.append(srv)
            self.worker_ports.append(port)

        coord_channel = grpc.insecure_channel(f"localhost:{self.primary_port}")
        coord_stub = primes_pb2_grpc.CoordinatorServiceStub(coord_channel)
        for i, port in enumerate(self.worker_ports):
            resp = coord_stub.RegisterNode(primes_pb2.RegisterNodeRequest(
                node_id=f"worker-{i}",
                host="localhost",
                port=port,
                cpu_count=2,
            ))
            assert resp.ok

        nodes_resp = coord_stub.ListNodes(primes_pb2.Empty())
        assert len(nodes_resp.nodes) == 2

        self.channel = coord_channel
        self.stub = coord_stub
        yield

        self.channel.close()
        for srv in self.worker_servers:
            srv.stop(0)
        self.primary_server.stop(0)
        primary_node.REGISTRY = self._original_registry

    def test_integration_count(self):
        """Compute(0, 10000, COUNT) with 2 workers → 1229 primes."""
        resp = self.stub.Compute(primes_pb2.ComputeRequest(
            low=0,
            high=10000,
            mode=primes_pb2.COUNT,
            secondary_exec=primes_pb2.PROCESSES,
            chunk=500_000,
        ))
        assert resp.ok
        assert resp.total_primes == 1229

    def test_integration_list(self):
        """Compute(0, 10000, LIST) with 2 workers → correct sorted primes."""
        resp = self.stub.Compute(primes_pb2.ComputeRequest(
            low=0,
            high=10000,
            mode=primes_pb2.LIST,
            secondary_exec=primes_pb2.SINGLE,
            chunk=500_000,
            max_return_primes=10000,
        ))
        assert resp.ok
        assert resp.total_primes == 1229

        primes = list(resp.primes)
        assert len(primes) == 1229

        assert primes == sorted(primes)

        assert primes[0] == 2
        assert primes[-1] == 9973

        expected = get_primes(0, 10000, return_list=True)
        assert primes == expected

    def test_integration_per_node(self):
        """Verify per-node results are returned when requested."""
        resp = self.stub.Compute(primes_pb2.ComputeRequest(
            low=0,
            high=10000,
            mode=primes_pb2.COUNT,
            secondary_exec=primes_pb2.SINGLE,
            chunk=500_000,
            include_per_node=True,
        ))
        assert resp.ok
        assert len(resp.per_node) == 2

        node_ids = {r.node_id for r in resp.per_node}
        assert node_ids == {"worker-0", "worker-1"}

        per_node_total = sum(r.total_primes for r in resp.per_node)
        assert per_node_total == resp.total_primes == 1229
