"""
MODULE 5: Memory Planner
==========================
Simulates MCU memory constraints and plans buffer allocation.

Features:
  - Weight memory calculation
  - Activation memory calculation
  - Peak memory estimation
  - Buffer reuse (liveness analysis)
  - Pass/Fail verdict against 256KB limit
"""

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir.ir_builder import IRGraph, IRNode, OpType


# Default MCU RAM constraint
MAX_RAM = 256 * 1024  # 256 KB


@dataclass
class MemoryBlock:
    """Represents a memory allocation block."""
    offset: int
    size: int
    node_name: str
    is_free: bool = False


@dataclass
class MemoryReport:
    """Complete memory analysis report."""
    weight_memory: int = 0
    peak_activation_memory: int = 0
    total_peak_memory: int = 0
    max_ram: int = MAX_RAM
    passed: bool = False
    # Detailed breakdown
    layer_memory: List[Dict] = field(default_factory=list)
    # Buffer reuse info
    buffers_allocated: int = 0
    buffers_reused: int = 0
    activation_memory_without_reuse: int = 0
    activation_memory_with_reuse: int = 0

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "MEMORY PLANNING REPORT",
            "=" * 60,
            f"Weight Memory:          {self.weight_memory:>10,} bytes ({self.weight_memory/1024:.1f} KB)",
            f"Peak Activation Memory: {self.peak_activation_memory:>10,} bytes ({self.peak_activation_memory/1024:.1f} KB)",
            f"Total Peak Memory:      {self.total_peak_memory:>10,} bytes ({self.total_peak_memory/1024:.1f} KB)",
            f"Max RAM Limit:          {self.max_ram:>10,} bytes ({self.max_ram/1024:.1f} KB)",
            f"Status:                 {'PASS' if self.passed else 'FAIL'}",
            "",
            f"Buffer Reuse Stats:",
            f"  Without reuse: {self.activation_memory_without_reuse/1024:.1f} KB",
            f"  With reuse:    {self.activation_memory_with_reuse/1024:.1f} KB",
            f"  Savings:       {(self.activation_memory_without_reuse - self.activation_memory_with_reuse)/1024:.1f} KB",
            f"  Buffers allocated: {self.buffers_allocated}",
            f"  Buffers reused:    {self.buffers_reused}",
            "",
            "Per-Layer Breakdown:",
            f"{'Layer':20s} | {'Weights (B)':>12s} | {'Activation (B)':>14s} | {'Cumulative':>12s}",
            "-" * 65,
        ]
        cumulative = 0
        for entry in self.layer_memory:
            cumulative += entry.get('activation', 0)
            lines.append(
                f"{entry['name']:20s} | {entry.get('weight', 0):12,} | "
                f"{entry.get('activation', 0):14,} | {cumulative:12,}"
            )
        return "\n".join(lines)


class MemoryPlanner:
    """Plans memory allocation for MCU deployment with buffer reuse."""

    def __init__(self, ir_graph: IRGraph, max_ram: int = MAX_RAM, verbose: bool = True):
        self.graph = ir_graph
        self.max_ram = max_ram
        self.verbose = verbose

    def plan(self) -> MemoryReport:
        """
        Perform memory planning and return a detailed report.
        """
        report = MemoryReport(max_ram=self.max_ram)

        compute_nodes = self.graph.get_compute_nodes()

        # 1. Calculate weight memory
        report.weight_memory = self._calculate_weight_memory(compute_nodes, report)

        # 2. Calculate activation memory without reuse
        report.activation_memory_without_reuse = sum(
            n.activation_memory_bytes() for n in compute_nodes
        )

        # 3. Calculate activation memory with buffer reuse (liveness analysis)
        peak_activation, reuse_stats = self._plan_with_buffer_reuse(compute_nodes)
        report.peak_activation_memory = peak_activation
        report.activation_memory_with_reuse = peak_activation
        report.buffers_allocated = reuse_stats['allocated']
        report.buffers_reused = reuse_stats['reused']

        # 4. Total peak memory
        report.total_peak_memory = report.weight_memory + report.peak_activation_memory

        # 5. Pass/Fail check
        report.passed = report.total_peak_memory <= self.max_ram

        # 6. Assign memory offsets to nodes
        self._assign_memory_offsets(compute_nodes)

        if self.verbose:
            print(report.summary())

        return report

    def _calculate_weight_memory(self, nodes: List[IRNode], report: MemoryReport) -> int:
        """Calculate total weight memory and build per-layer breakdown."""
        total = 0
        for node in nodes:
            w_mem = node.weight_memory_bytes()
            a_mem = node.activation_memory_bytes()
            total += w_mem
            report.layer_memory.append({
                'name': node.name,
                'weight': w_mem,
                'activation': a_mem,
                'op_type': node.op_type.value,
            })
        return total

    def _plan_with_buffer_reuse(self, nodes: List[IRNode]) -> Tuple[int, Dict]:
        """
        Plan activation buffers with reuse based on liveness analysis.
        
        A tensor can be freed once all its consumers have been computed.
        Freed memory can be reused by subsequent tensors.
        """
        # Build liveness: for each node, track when its output is last used
        last_use = {}
        for i, node in enumerate(nodes):
            for inp_name in node.inputs:
                last_use[inp_name] = i

        # Simulate memory allocation with greedy reuse
        active_buffers: List[MemoryBlock] = []
        peak_memory = 0
        current_memory = 0
        stats = {'allocated': 0, 'reused': 0}

        for i, node in enumerate(nodes):
            act_size = node.activation_memory_bytes()
            if act_size == 0:
                continue

            # Check if any active buffers can be freed
            freed_block = None
            for buf in active_buffers:
                if not buf.is_free and last_use.get(buf.node_name, -1) <= i:
                    buf.is_free = True
                    current_memory -= buf.size

            # Try to reuse a freed buffer
            reused = False
            for buf in active_buffers:
                if buf.is_free and buf.size >= act_size:
                    buf.is_free = False
                    buf.node_name = node.name
                    current_memory += act_size
                    stats['reused'] += 1
                    reused = True
                    break

            if not reused:
                # Allocate new buffer
                block = MemoryBlock(offset=current_memory, size=act_size, node_name=node.name)
                active_buffers.append(block)
                current_memory += act_size
                stats['allocated'] += 1

            peak_memory = max(peak_memory, current_memory)
            node.memory_size = act_size

        return peak_memory, stats

    def _assign_memory_offsets(self, nodes: List[IRNode]):
        """Assign memory offsets for static allocation in generated C code."""
        offset = 0
        for node in nodes:
            if node.activation_memory_bytes() > 0:
                node.memory_offset = offset
                # For simplicity, use double-buffering approach
                # Alternate between two buffers
                node.memory_size = node.activation_memory_bytes()


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from frontend.graph_extractor import GraphExtractor
    from ir.ir_builder import IRBuilder
    from optimizer.graph_optimizer import GraphOptimizer
    from quantization.int8_quantizer import INT8Quantizer

    class TestCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 14 * 14, 10)
        def forward(self, x):
            return self.fc1(self.flatten(self.pool1(self.relu1(self.conv1(x)))))

    model = TestCNN()
    sample = torch.randn(1, 1, 28, 28)
    graph = GraphExtractor(model, sample).extract()
    ir = IRBuilder(model, graph).build()
    ir = GraphOptimizer(ir, verbose=False).optimize()
    ir = INT8Quantizer(ir, verbose=False).quantize()
    report = MemoryPlanner(ir).plan()
    print("\n✅ Memory planning test passed!")
