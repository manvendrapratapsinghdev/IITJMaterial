"""
MODULE 2: Intermediate Representation (IR) Builder
====================================================
Converts the extracted torch.fx graph into a custom IR suitable
for compiler optimizations and code generation.
"""

import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from enum import Enum
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from frontend.graph_extractor import ExtractedGraph, GraphNode


class OpType(Enum):
    PLACEHOLDER = "placeholder"
    OUTPUT = "output"
    CONV2D = "conv2d"
    LINEAR = "linear"
    RELU = "relu"
    MAXPOOL2D = "maxpool2d"
    AVGPOOL2D = "avgpool2d"
    ADAPTIVE_AVGPOOL2D = "adaptive_avgpool2d"
    FLATTEN = "flatten"
    SOFTMAX = "softmax"
    BATCHNORM2D = "batchnorm2d"
    DROPOUT = "dropout"
    ADD = "add"
    FUSED_CONV_RELU = "fused_conv_relu"
    FUSED_LINEAR_RELU = "fused_linear_relu"
    UNKNOWN = "unknown"


_MODULE_TYPE_MAP = {
    'Conv2d': OpType.CONV2D, 'Linear': OpType.LINEAR, 'ReLU': OpType.RELU,
    'MaxPool2d': OpType.MAXPOOL2D, 'AvgPool2d': OpType.AVGPOOL2D,
    'AdaptiveAvgPool2d': OpType.ADAPTIVE_AVGPOOL2D, 'Flatten': OpType.FLATTEN,
    'Softmax': OpType.SOFTMAX, 'BatchNorm2d': OpType.BATCHNORM2D,
    'Dropout': OpType.DROPOUT, 'relu': OpType.RELU, 'flatten': OpType.FLATTEN,
    'softmax': OpType.SOFTMAX, 'max_pool2d': OpType.MAXPOOL2D,
    'adaptive_avg_pool2d': OpType.ADAPTIVE_AVGPOOL2D, 'add': OpType.ADD,
}


@dataclass
class IRNode:
    """Intermediate Representation node."""
    name: str
    op_type: OpType
    inputs: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    weight: Optional[np.ndarray] = None
    bias: Optional[np.ndarray] = None
    input_shape: Optional[Tuple] = None
    output_shape: Optional[Tuple] = None
    params: Dict[str, Any] = field(default_factory=dict)
    # Quantization data
    quantized: bool = False
    weight_int8: Optional[np.ndarray] = None
    bias_int32: Optional[np.ndarray] = None
    weight_scale: Optional[float] = None
    weight_zero_point: Optional[int] = None
    input_scale: Optional[float] = None
    input_zero_point: Optional[int] = None
    output_scale: Optional[float] = None
    output_zero_point: Optional[int] = None
    # Memory planning
    memory_offset: Optional[int] = None
    memory_size: Optional[int] = None
    eliminated: bool = False

    def weight_memory_bytes(self) -> int:
        total = 0
        if self.weight is not None:
            total += self.weight_int8.nbytes if (self.quantized and self.weight_int8 is not None) else self.weight.nbytes
        if self.bias is not None:
            total += self.bias_int32.nbytes if (self.quantized and self.bias_int32 is not None) else self.bias.nbytes
        return total

    def activation_memory_bytes(self) -> int:
        if self.output_shape is None:
            return 0
        element_size = 1 if self.quantized else 4
        num_elements = 1
        for dim in self.output_shape:
            num_elements *= dim
        return num_elements * element_size


@dataclass
class IRGraph:
    """Complete IR graph representation."""
    nodes: List[IRNode] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)
    model_name: str = ""

    def get_node(self, name: str) -> Optional[IRNode]:
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_compute_nodes(self) -> List[IRNode]:
        return [n for n in self.nodes if n.op_type not in (OpType.PLACEHOLDER, OpType.OUTPUT) and not n.eliminated]

    def get_parameterized_nodes(self) -> List[IRNode]:
        return [n for n in self.get_compute_nodes() if n.weight is not None]

    def total_weight_memory(self) -> int:
        return sum(n.weight_memory_bytes() for n in self.nodes)

    def total_parameters(self) -> int:
        total = 0
        for n in self.nodes:
            if n.weight is not None: total += n.weight.size
            if n.bias is not None: total += n.bias.size
        return total

    def summary(self) -> str:
        lines = [
            f"IR Graph: {self.model_name}",
            f"Total nodes: {len(self.nodes)}, Compute: {len(self.get_compute_nodes())}",
            f"Parameters: {self.total_parameters():,}, Weight mem: {self.total_weight_memory()/1024:.1f} KB",
            "", f"{'Name':20s} | {'Op':20s} | {'In Shape':20s} | {'Out Shape':20s} | {'W Shape':15s}",
            "-" * 100,
        ]
        for n in self.nodes:
            if n.eliminated: continue
            ws = str(n.weight.shape) if n.weight is not None else "-"
            lines.append(f"{n.name:20s} | {n.op_type.value:20s} | {str(n.input_shape):20s} | {str(n.output_shape):20s} | {ws:15s}")
        return "\n".join(lines)


class IRBuilder:
    """Converts an ExtractedGraph into an IRGraph with full parameter extraction."""

    def __init__(self, model: nn.Module, extracted_graph: ExtractedGraph):
        self.model = model
        self.extracted_graph = extracted_graph

    def build(self) -> IRGraph:
        ir_graph = IRGraph(model_name=self.extracted_graph.model_name)
        output_shape_map = {gn.name: gn.output_shape for gn in self.extracted_graph.nodes}

        for gnode in self.extracted_graph.nodes:
            ir_node = self._convert_node(gnode, output_shape_map)
            ir_graph.nodes.append(ir_node)

        ir_graph.adjacency = dict(self.extracted_graph.adjacency)
        for name, succs in ir_graph.adjacency.items():
            ir_node = ir_graph.get_node(name)
            if ir_node:
                ir_node.outputs = list(succs)
        return ir_graph

    def _convert_node(self, gnode: GraphNode, shape_map: Dict) -> IRNode:
        op_type = self._resolve_op_type(gnode)
        input_shape = shape_map.get(gnode.inputs[0]) if gnode.inputs else None
        weight, bias, params = self._extract_parameters(gnode, op_type)
        return IRNode(name=gnode.name, op_type=op_type, inputs=list(gnode.inputs),
                      weight=weight, bias=bias, input_shape=input_shape,
                      output_shape=gnode.output_shape, params=params)

    def _resolve_op_type(self, gnode: GraphNode) -> OpType:
        if gnode.op_type == 'placeholder': return OpType.PLACEHOLDER
        if gnode.op_type == 'output': return OpType.OUTPUT
        if gnode.module_type and gnode.module_type in _MODULE_TYPE_MAP:
            return _MODULE_TYPE_MAP[gnode.module_type]
        target_lower = gnode.target.lower()
        for key, op in _MODULE_TYPE_MAP.items():
            if key.lower() in target_lower: return op
        return OpType.UNKNOWN

    def _extract_parameters(self, gnode, op_type):
        weight, bias, params = None, None, {}
        if gnode.op_type != 'call_module':
            if op_type == OpType.MAXPOOL2D: params = dict(gnode.kwargs)
            return weight, bias, params
        try:
            module = self.model
            for attr in gnode.target.split('.'):
                module = getattr(module, attr)
        except AttributeError:
            return weight, bias, params

        if isinstance(module, nn.Conv2d):
            weight = module.weight.detach().cpu().numpy()
            if module.bias is not None: bias = module.bias.detach().cpu().numpy()
            params = {'in_channels': module.in_channels, 'out_channels': module.out_channels,
                      'kernel_size': module.kernel_size, 'stride': module.stride,
                      'padding': module.padding, 'groups': module.groups}
        elif isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy()
            if module.bias is not None: bias = module.bias.detach().cpu().numpy()
            params = {'in_features': module.in_features, 'out_features': module.out_features}
        elif isinstance(module, nn.MaxPool2d):
            ks, st, pd = module.kernel_size, module.stride, module.padding
            params = {'kernel_size': ks if isinstance(ks, tuple) else (ks, ks),
                      'stride': st if isinstance(st, tuple) else (st, st),
                      'padding': pd if isinstance(pd, tuple) else (pd, pd)}
        elif isinstance(module, nn.Flatten):
            params = {'start_dim': module.start_dim, 'end_dim': module.end_dim}
        elif isinstance(module, nn.BatchNorm2d):
            weight = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy()
            params = {'num_features': module.num_features, 'eps': module.eps,
                      'running_mean': module.running_mean.detach().cpu().numpy(),
                      'running_var': module.running_var.detach().cpu().numpy()}
        return weight, bias, params


if __name__ == "__main__":
    from frontend.graph_extractor import GraphExtractor
    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(16 * 14 * 14, 10)
        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.flatten(x)
            return self.fc1(x)
    model = SimpleCNN()
    sample = torch.randn(1, 1, 28, 28)
    graph = GraphExtractor(model, sample).extract()
    ir = IRBuilder(model, graph).build()
    print(ir.summary())
    print("\n✅ IR build successful!")
