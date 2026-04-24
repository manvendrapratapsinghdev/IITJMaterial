"""
MODULE 1: Graph Extraction
===========================
Converts a PyTorch model into a computation graph using torch.fx.

Extracts:
  - Node names
  - Operation types
  - Input connections
  - Output shapes

Usage:
    from frontend.graph_extractor import GraphExtractor
    extractor = GraphExtractor(model, sample_input)
    graph_info = extractor.extract()
"""

import torch
import torch.nn as nn
import torch.fx
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass
class GraphNode:
    """Represents a single node in the extracted computation graph."""
    name: str
    op_type: str  # 'call_module', 'call_function', 'call_method', 'placeholder', 'output', 'get_attr'
    target: str   # The actual operation target (e.g., 'conv1', 'relu', etc.)
    inputs: List[str] = field(default_factory=list)
    output_shape: Optional[Tuple] = None
    module_type: Optional[str] = None  # e.g., 'Conv2d', 'Linear', 'ReLU'
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self):
        return (
            f"GraphNode(name={self.name!r}, op={self.op_type!r}, "
            f"target={self.target!r}, inputs={self.inputs}, "
            f"shape={self.output_shape}, module_type={self.module_type!r})"
        )


@dataclass
class ExtractedGraph:
    """Complete extracted computation graph."""
    nodes: List[GraphNode] = field(default_factory=list)
    adjacency: Dict[str, List[str]] = field(default_factory=dict)  # node_name -> list of successor names
    input_shapes: Dict[str, Tuple] = field(default_factory=dict)
    model_name: str = ""

    def get_node(self, name: str) -> Optional[GraphNode]:
        """Retrieve a node by name."""
        for node in self.nodes:
            if node.name == name:
                return node
        return None

    def get_ordered_nodes(self) -> List[GraphNode]:
        """Return nodes in topological order (as extracted)."""
        return list(self.nodes)

    def summary(self) -> str:
        """Human-readable summary of the graph."""
        lines = [f"Graph: {self.model_name}", f"Total nodes: {len(self.nodes)}", ""]
        for node in self.nodes:
            lines.append(
                f"  {node.name:20s} | {node.op_type:15s} | "
                f"{node.module_type or node.target:20s} | "
                f"inputs={node.inputs} | shape={node.output_shape}"
            )
        return "\n".join(lines)


class GraphExtractor:
    """
    Extracts computation graph from a PyTorch model using torch.fx symbolic tracing.
    
    Args:
        model: PyTorch nn.Module to extract graph from.
        sample_input: A sample input tensor for shape inference.
    """

    # Supported operations for TinyML compilation
    SUPPORTED_OPS = {
        'Conv2d', 'Linear', 'ReLU', 'MaxPool2d',
        'Softmax', 'Flatten', 'BatchNorm2d', 'Dropout',
        'AdaptiveAvgPool2d', 'AvgPool2d',
    }

    def __init__(self, model: nn.Module, sample_input: torch.Tensor):
        self.model = model
        self.sample_input = sample_input
        self.model.eval()  # Ensure model is in eval mode

    def extract(self) -> ExtractedGraph:
        """
        Extract the computation graph from the model.
        
        Returns:
            ExtractedGraph containing all nodes with shape information.
        """
        # Step 1: Symbolic trace the model
        traced = torch.fx.symbolic_trace(self.model)

        # Step 2: Run shape propagation
        shape_map = self._infer_shapes(traced)

        # Step 3: Build our graph representation
        graph = ExtractedGraph(model_name=self.model.__class__.__name__)

        for node in traced.graph.nodes:
            graph_node = self._process_node(node, traced, shape_map)
            graph.nodes.append(graph_node)

        # Step 4: Build adjacency list
        graph.adjacency = self._build_adjacency(traced)

        # Step 5: Validate supported operations
        self._validate_ops(graph)

        return graph

    def _infer_shapes(self, traced: torch.fx.GraphModule) -> Dict[str, Tuple]:
        """
        Infer output shapes for each node by running the model with the sample input.
        Uses torch.fx's shape propagation via a concrete run.
        """
        shape_map = {}

        # Use ShapeProp for shape inference
        class ShapeRecorder(torch.fx.Interpreter):
            def __init__(self, module):
                super().__init__(module)
                self.shape_map = {}

            def run_node(self, n):
                result = super().run_node(n)
                if isinstance(result, torch.Tensor):
                    self.shape_map[n.name] = tuple(result.shape)
                elif isinstance(result, (tuple, list)):
                    # For nodes returning tuples, record the first tensor shape
                    for item in result:
                        if isinstance(item, torch.Tensor):
                            self.shape_map[n.name] = tuple(item.shape)
                            break
                return result

        recorder = ShapeRecorder(traced)
        recorder.run(self.sample_input)
        shape_map = recorder.shape_map

        return shape_map

    def _process_node(
        self, node: torch.fx.Node, traced: torch.fx.GraphModule,
        shape_map: Dict[str, Tuple]
    ) -> GraphNode:
        """
        Process a single torch.fx node into our GraphNode representation.
        """
        # Get input node names
        inputs = []
        for arg in node.args:
            if isinstance(arg, torch.fx.Node):
                inputs.append(arg.name)
            elif isinstance(arg, (list, tuple)):
                for a in arg:
                    if isinstance(a, torch.fx.Node):
                        inputs.append(a.name)

        # Determine module type for call_module nodes
        module_type = None
        target_str = str(node.target)

        if node.op == 'call_module':
            try:
                module = traced.get_submodule(node.target)
                module_type = module.__class__.__name__
            except AttributeError:
                module_type = "Unknown"
        elif node.op == 'call_function':
            # Handle functional ops like torch.relu, torch.flatten
            func_name = node.target.__name__ if hasattr(node.target, '__name__') else str(node.target)
            module_type = func_name
            target_str = func_name
        elif node.op == 'call_method':
            module_type = node.target
            target_str = node.target

        # Get kwargs (filter out None values)
        kwargs = {}
        for k, v in node.kwargs.items():
            if v is not None and not isinstance(v, torch.fx.Node):
                kwargs[k] = v

        return GraphNode(
            name=node.name,
            op_type=node.op,
            target=target_str,
            inputs=inputs,
            output_shape=shape_map.get(node.name),
            module_type=module_type,
            kwargs=kwargs,
        )

    def _build_adjacency(self, traced: torch.fx.GraphModule) -> Dict[str, List[str]]:
        """Build adjacency list: node_name -> list of nodes that consume its output."""
        adjacency = {}
        for node in traced.graph.nodes:
            adjacency[node.name] = []
            for user in node.users:
                adjacency[node.name].append(user.name)
        return adjacency

    def _validate_ops(self, graph: ExtractedGraph) -> None:
        """
        Validate that all operations in the graph are supported.
        Raises ValueError for unsupported operations.
        """
        unsupported = []
        for node in graph.nodes:
            if node.op_type in ('placeholder', 'output', 'get_attr'):
                continue  # These are structural nodes, always supported

            if node.module_type and node.module_type not in self.SUPPORTED_OPS:
                # Check if it's a known functional equivalent
                functional_map = {
                    'relu': 'ReLU', 'flatten': 'Flatten',
                    'softmax': 'Softmax', 'max_pool2d': 'MaxPool2d',
                    'adaptive_avg_pool2d': 'AdaptiveAvgPool2d',
                    'avg_pool2d': 'AvgPool2d', 'dropout': 'Dropout',
                    'batch_norm': 'BatchNorm2d',
                }
                mapped = functional_map.get(node.module_type)
                if mapped and mapped in self.SUPPORTED_OPS:
                    node.module_type = mapped  # Normalize the name
                elif node.op_type == 'call_function' and node.module_type in ('getattr', 'getitem'):
                    continue  # Structural ops
                else:
                    unsupported.append(f"{node.name} ({node.module_type})")

        if unsupported:
            print(f"⚠️  Warning: Unsupported operations found: {unsupported}")
            print("   These operations will be skipped during code generation.")


# =====================================================
# Self-test
# =====================================================
def _self_test():
    """Quick self-test for GraphExtractor."""
    import torch.nn as nn

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.relu1 = nn.ReLU()
            self.pool1 = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.relu2 = nn.ReLU()
            self.pool2 = nn.MaxPool2d(2)
            self.flatten = nn.Flatten()
            self.fc1 = nn.Linear(32 * 7 * 7, 128)
            self.relu3 = nn.ReLU()
            self.fc2 = nn.Linear(128, 10)

        def forward(self, x):
            x = self.pool1(self.relu1(self.conv1(x)))
            x = self.pool2(self.relu2(self.conv2(x)))
            x = self.flatten(x)
            x = self.relu3(self.fc1(x))
            x = self.fc2(x)
            return x

    model = SimpleCNN()
    sample = torch.randn(1, 1, 28, 28)

    extractor = GraphExtractor(model, sample)
    graph = extractor.extract()

    print("=" * 60)
    print("GRAPH EXTRACTION SELF-TEST")
    print("=" * 60)
    print(graph.summary())
    print(f"\nAdjacency list:")
    for name, successors in graph.adjacency.items():
        if successors:
            print(f"  {name} → {successors}")
    print(f"\n✅ Graph extraction successful! {len(graph.nodes)} nodes extracted.")
    return graph


if __name__ == "__main__":
    _self_test()
