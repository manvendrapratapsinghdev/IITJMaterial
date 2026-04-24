"""
MODULE 6: Latency Estimator (Hardware Aware)
=============================================
Estimates inference latency based on a simple cycle model.

Cycle models:
  Conv2D: output_channels * input_channels * kernel_size^2 * output_width * output_height
  Linear: in_features * out_features

Assumes 80 MHz MCU clock for time estimation.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir.ir_builder import IRGraph, IRNode, OpType

# Default MCU clock frequency
DEFAULT_CLOCK_MHZ = 80


@dataclass
class LayerLatency:
    """Latency information for a single layer."""
    name: str
    op_type: str
    cycles: int
    time_us: float  # microseconds
    percentage: float = 0.0


@dataclass
class LatencyReport:
    """Complete latency estimation report."""
    total_cycles: int = 0
    total_time_us: float = 0.0
    total_time_ms: float = 0.0
    clock_mhz: int = DEFAULT_CLOCK_MHZ
    layer_latencies: List[LayerLatency] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 70,
            "LATENCY ESTIMATION REPORT",
            "=" * 70,
            f"MCU Clock: {self.clock_mhz} MHz",
            f"Total Cycles:  {self.total_cycles:>15,}",
            f"Total Time:    {self.total_time_us:>15,.1f} µs ({self.total_time_ms:.2f} ms)",
            "",
            f"{'Layer':20s} | {'Op Type':20s} | {'Cycles':>12s} | {'Time (µs)':>10s} | {'%':>6s}",
            "-" * 75,
        ]
        for ll in self.layer_latencies:
            lines.append(
                f"{ll.name:20s} | {ll.op_type:20s} | {ll.cycles:12,} | "
                f"{ll.time_us:10.1f} | {ll.percentage:5.1f}%"
            )
        return "\n".join(lines)


class LatencyEstimator:
    """Hardware-aware latency estimator for MCU deployment."""

    def __init__(self, ir_graph: IRGraph, clock_mhz: int = DEFAULT_CLOCK_MHZ, verbose: bool = True):
        self.graph = ir_graph
        self.clock_mhz = clock_mhz
        self.verbose = verbose

    def estimate(self) -> LatencyReport:
        """Estimate latency for the entire model."""
        report = LatencyReport(clock_mhz=self.clock_mhz)

        for node in self.graph.get_compute_nodes():
            if node.eliminated:
                continue
            cycles = self._estimate_node_cycles(node)
            time_us = cycles / self.clock_mhz  # cycles / (MHz) = µs
            ll = LayerLatency(
                name=node.name, op_type=node.op_type.value,
                cycles=cycles, time_us=time_us,
            )
            report.layer_latencies.append(ll)
            report.total_cycles += cycles

        report.total_time_us = report.total_cycles / self.clock_mhz
        report.total_time_ms = report.total_time_us / 1000.0

        # Calculate percentages
        for ll in report.layer_latencies:
            ll.percentage = (ll.cycles / report.total_cycles * 100) if report.total_cycles > 0 else 0

        if self.verbose:
            print(report.summary())

        return report

    def _estimate_node_cycles(self, node: IRNode) -> int:
        """Estimate cycles for a single node based on operation type."""
        if node.op_type in (OpType.CONV2D, OpType.FUSED_CONV_RELU):
            return self._conv2d_cycles(node)
        elif node.op_type in (OpType.LINEAR, OpType.FUSED_LINEAR_RELU):
            return self._linear_cycles(node)
        elif node.op_type == OpType.MAXPOOL2D:
            return self._maxpool2d_cycles(node)
        elif node.op_type == OpType.AVGPOOL2D:
            return self._avgpool2d_cycles(node)
        elif node.op_type in (OpType.RELU,):
            return self._relu_cycles(node)
        elif node.op_type == OpType.FLATTEN:
            return 1  # Essentially free (just pointer manipulation)
        elif node.op_type == OpType.SOFTMAX:
            return self._softmax_cycles(node)
        else:
            return 0

    def _conv2d_cycles(self, node: IRNode) -> int:
        """Conv2D: OC * IC * K^2 * OH * OW"""
        p = node.params
        oc = p.get('out_channels', 1)
        ic = p.get('in_channels', 1)
        ks = p.get('kernel_size', (3, 3))
        k = ks[0] if isinstance(ks, tuple) else ks

        if node.output_shape and len(node.output_shape) == 4:
            _, _, oh, ow = node.output_shape
        else:
            oh, ow = 1, 1

        # INT8 multiply-accumulate is ~1 cycle on most MCUs
        # Add 1 cycle for ReLU if fused
        mac_cycles = oc * ic * k * k * oh * ow
        if node.op_type == OpType.FUSED_CONV_RELU:
            mac_cycles += oc * oh * ow  # ReLU comparison
        return mac_cycles

    def _linear_cycles(self, node: IRNode) -> int:
        """Linear: in_features * out_features"""
        p = node.params
        in_f = p.get('in_features', 1)
        out_f = p.get('out_features', 1)
        cycles = in_f * out_f
        if node.op_type == OpType.FUSED_LINEAR_RELU:
            cycles += out_f
        return cycles

    def _maxpool2d_cycles(self, node: IRNode) -> int:
        """MaxPool2D: C * OH * OW * K^2 comparisons"""
        if node.output_shape and len(node.output_shape) == 4:
            _, c, oh, ow = node.output_shape
        else:
            return 0
        ks = node.params.get('kernel_size', (2, 2))
        k = ks[0] if isinstance(ks, tuple) else ks
        return c * oh * ow * k * k

    def _avgpool2d_cycles(self, node: IRNode) -> int:
        """AvgPool2D: similar to MaxPool but with add+divide"""
        if node.output_shape and len(node.output_shape) == 4:
            _, c, oh, ow = node.output_shape
        else:
            return 0
        ks = node.params.get('kernel_size', (2, 2))
        k = ks[0] if isinstance(ks, tuple) else ks
        return c * oh * ow * (k * k + 1)  # +1 for division

    def _relu_cycles(self, node: IRNode) -> int:
        """ReLU: one comparison per element"""
        if node.output_shape:
            total = 1
            for d in node.output_shape:
                total *= d
            return total
        return 0

    def _softmax_cycles(self, node: IRNode) -> int:
        """Softmax approximation: ~10 cycles per element (exp + div)"""
        if node.output_shape:
            total = 1
            for d in node.output_shape:
                total *= d
            return total * 10
        return 0


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from frontend.graph_extractor import GraphExtractor
    from ir.ir_builder import IRBuilder
    from optimizer.graph_optimizer import GraphOptimizer

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
    report = LatencyEstimator(ir).estimate()
    print(f"\n✅ Latency estimation test passed!")
