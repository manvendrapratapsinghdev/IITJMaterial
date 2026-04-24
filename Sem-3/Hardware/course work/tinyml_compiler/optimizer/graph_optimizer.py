"""
MODULE 3: Graph Optimizer
==========================
Implements graph-level optimizations:
  1. Dead Node Elimination
  2. Operator Fusion (Conv+ReLU, Linear+ReLU)
  3. Constant Folding
  4. Dropout Removal (inference mode)
"""

import copy
from typing import List, Set
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir.ir_builder import IRGraph, IRNode, OpType


class GraphOptimizer:
    """Applies optimization passes to an IRGraph."""

    def __init__(self, ir_graph: IRGraph, verbose: bool = True):
        self.graph = ir_graph
        self.verbose = verbose
        self.stats = {'eliminated': 0, 'fused': 0, 'folded': 0}

    def optimize(self) -> IRGraph:
        """Run all optimization passes in sequence."""
        self._log("Starting graph optimization...")
        original_count = len(self.graph.get_compute_nodes())

        self._remove_dropout()
        self._dead_node_elimination()
        self._operator_fusion()
        self._constant_folding()

        final_count = len(self.graph.get_compute_nodes())
        self._log(f"Optimization complete: {original_count} → {final_count} compute nodes")
        self._log(f"Stats: eliminated={self.stats['eliminated']}, "
                  f"fused={self.stats['fused']}, folded={self.stats['folded']}")
        return self.graph

    def _remove_dropout(self):
        """Remove Dropout layers (no-op during inference)."""
        for node in self.graph.nodes:
            if node.op_type == OpType.DROPOUT and not node.eliminated:
                self._bypass_node(node)
                self.stats['eliminated'] += 1
                self._log(f"  Removed dropout: {node.name}")

    def _dead_node_elimination(self):
        """Remove nodes not contributing to the output."""
        # Find output node
        output_node = None
        for n in self.graph.nodes:
            if n.op_type == OpType.OUTPUT:
                output_node = n
                break
        if output_node is None:
            return

        # BFS backward from output to find live nodes
        live_nodes: Set[str] = set()
        queue = [output_node.name]
        while queue:
            current = queue.pop(0)
            if current in live_nodes:
                continue
            live_nodes.add(current)
            node = self.graph.get_node(current)
            if node:
                for inp in node.inputs:
                    queue.append(inp)

        # Eliminate dead nodes
        for node in self.graph.nodes:
            if node.name not in live_nodes and not node.eliminated:
                node.eliminated = True
                self.stats['eliminated'] += 1
                self._log(f"  Dead node eliminated: {node.name}")

    def _operator_fusion(self):
        """Fuse Conv+ReLU and Linear+ReLU pairs."""
        nodes = self.graph.nodes
        for i, node in enumerate(nodes):
            if node.eliminated:
                continue

            # Check for Conv2D + ReLU fusion
            if node.op_type == OpType.CONV2D:
                relu = self._find_single_consumer_of_type(node, OpType.RELU)
                if relu:
                    self._fuse_nodes(node, relu, OpType.FUSED_CONV_RELU)

            # Check for Linear + ReLU fusion
            elif node.op_type == OpType.LINEAR:
                relu = self._find_single_consumer_of_type(node, OpType.RELU)
                if relu:
                    self._fuse_nodes(node, relu, OpType.FUSED_LINEAR_RELU)

    def _find_single_consumer_of_type(self, node: IRNode, target_type: OpType):
        """Find if a node has exactly one consumer of a given type."""
        consumers = [n for n in node.outputs if not self.graph.get_node(n).eliminated] \
                    if node.outputs else []
        if len(consumers) == 1:
            consumer = self.graph.get_node(consumers[0])
            if consumer and consumer.op_type == target_type and not consumer.eliminated:
                return consumer
        return None

    def _fuse_nodes(self, primary: IRNode, secondary: IRNode, fused_type: OpType):
        """Fuse two consecutive nodes into one."""
        primary.op_type = fused_type
        primary.output_shape = secondary.output_shape
        primary.outputs = list(secondary.outputs)
        # Update all nodes that had secondary as input
        for node in self.graph.nodes:
            node.inputs = [primary.name if inp == secondary.name else inp for inp in node.inputs]
        # Update adjacency
        if secondary.name in self.graph.adjacency:
            self.graph.adjacency[primary.name] = self.graph.adjacency[secondary.name]
        secondary.eliminated = True
        self.stats['fused'] += 1
        self._log(f"  Fused: {primary.name} + {secondary.name} → {fused_type.value}")

    def _bypass_node(self, node: IRNode):
        """Remove a node by connecting its inputs directly to its outputs."""
        if not node.inputs:
            return
        predecessor_name = node.inputs[0]
        # Update successors to point to predecessor
        for succ_name in node.outputs:
            succ = self.graph.get_node(succ_name)
            if succ:
                succ.inputs = [predecessor_name if inp == node.name else inp for inp in succ.inputs]
        # Update predecessor's outputs
        pred = self.graph.get_node(predecessor_name)
        if pred:
            pred.outputs = [s if s != node.name else succ_name 
                           for s in pred.outputs for succ_name in ([s] if s != node.name else node.outputs)]
        # Update adjacency
        if predecessor_name in self.graph.adjacency:
            self.graph.adjacency[predecessor_name] = [
                s if s != node.name else succ 
                for s in self.graph.adjacency.get(predecessor_name, [])
                for succ in ([s] if s != node.name else node.outputs)
            ]
        node.eliminated = True

    def _constant_folding(self):
        """Fold constant computations at compile time."""
        # In a TinyML context, most constants are already folded as weights
        # This pass handles BatchNorm folding into preceding Conv2d
        for node in self.graph.nodes:
            if node.eliminated or node.op_type != OpType.BATCHNORM2D:
                continue
            if not node.inputs:
                continue
            pred = self.graph.get_node(node.inputs[0])
            if pred and pred.op_type in (OpType.CONV2D, OpType.FUSED_CONV_RELU) and not pred.eliminated:
                self._fold_batchnorm_into_conv(pred, node)

    def _fold_batchnorm_into_conv(self, conv_node: IRNode, bn_node: IRNode):
        """Fold BatchNorm parameters into Conv2d weights."""
        import numpy as np
        if conv_node.weight is None or bn_node.weight is None:
            return

        gamma = bn_node.weight
        beta = bn_node.bias
        mean = bn_node.params.get('running_mean')
        var = bn_node.params.get('running_var')
        eps = bn_node.params.get('eps', 1e-5)

        if mean is None or var is None:
            return

        std = np.sqrt(var + eps)
        scale = gamma / std

        # Fold into conv weights: W_new = W * scale
        # For each output channel
        for i in range(conv_node.weight.shape[0]):
            conv_node.weight[i] *= scale[i]

        # Fold into bias: b_new = (b - mean) * scale + beta
        if conv_node.bias is not None:
            conv_node.bias = (conv_node.bias - mean) * scale + beta
        else:
            conv_node.bias = (-mean) * scale + beta

        # Bypass BatchNorm node
        self._bypass_node(bn_node)
        self.stats['folded'] += 1
        self._log(f"  Folded BatchNorm {bn_node.name} into {conv_node.name}")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Optimizer] {msg}")


if __name__ == "__main__":
    import torch
    import torch.nn as nn
    from frontend.graph_extractor import GraphExtractor
    from ir.ir_builder import IRBuilder

    class TestCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 14 * 14, 64)
            self.relu2 = nn.ReLU()
            self.fc2 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.flatten(x)
            x = self.relu2(self.fc1(x))
            return self.fc2(x)

    model = TestCNN()
    sample = torch.randn(1, 1, 28, 28)
    graph = GraphExtractor(model, sample).extract()
    ir = IRBuilder(model, graph).build()
    print("Before optimization:")
    print(ir.summary())
    opt = GraphOptimizer(ir)
    ir = opt.optimize()
    print("\nAfter optimization:")
    print(ir.summary())
    print("\n✅ Optimizer test passed!")
