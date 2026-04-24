"""
MODULE 4: INT8 Quantization Engine
====================================
Implements post-training INT8 quantization.

Steps:
  1. Compute min/max per tensor
  2. Calculate scale and zero_point
  3. Convert weights to int8
  4. Store quantization parameters per layer

Supports per-tensor quantization.
"""

import numpy as np
from typing import Optional, Tuple, Dict
import sys, os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ir.ir_builder import IRGraph, IRNode, OpType


class QuantizationParams:
    """Stores quantization parameters for a tensor."""
    def __init__(self, scale: float, zero_point: int, min_val: float, max_val: float):
        self.scale = scale
        self.zero_point = zero_point
        self.min_val = min_val
        self.max_val = max_val

    def __repr__(self):
        return f"QParams(scale={self.scale:.6f}, zp={self.zero_point}, range=[{self.min_val:.4f}, {self.max_val:.4f}])"


class INT8Quantizer:
    """
    Post-training INT8 quantization engine.
    
    Quantizes weights and computes activation quantization parameters
    using calibration data.
    """

    def __init__(self, ir_graph: IRGraph, verbose: bool = True):
        self.graph = ir_graph
        self.verbose = verbose
        self.calibration_stats: Dict[str, Dict] = {}  # node_name -> {min, max}

    def quantize(self, calibration_data=None) -> IRGraph:
        """
        Quantize all parameterized layers in the graph.
        
        Args:
            calibration_data: Optional list of input tensors for activation range calibration.
        """
        self._log("Starting INT8 quantization...")

        # Step 1: Quantize weights for all parameterized nodes
        for node in self.graph.get_parameterized_nodes():
            if node.eliminated:
                continue
            self._quantize_weights(node)

        # Step 2: Compute activation quantization params
        # Use default symmetric range if no calibration data
        for node in self.graph.get_compute_nodes():
            if node.eliminated:
                continue
            self._compute_activation_params(node)

        # Print summary
        self._print_quantization_summary()
        return self.graph

    def _quantize_weights(self, node: IRNode):
        """Quantize the weights of a single node to INT8."""
        if node.weight is None:
            return

        weight = node.weight.astype(np.float32)

        # Compute per-tensor quantization parameters
        w_min = float(weight.min())
        w_max = float(weight.max())

        # Symmetric quantization for weights
        abs_max = max(abs(w_min), abs(w_max))
        if abs_max == 0:
            abs_max = 1e-8  # Avoid division by zero

        scale = abs_max / 127.0
        zero_point = 0  # Symmetric quantization

        # Quantize weights
        weight_int8 = np.clip(np.round(weight / scale), -128, 127).astype(np.int8)

        # Quantize bias to int32 (bias_scale = input_scale * weight_scale)
        bias_int32 = None
        if node.bias is not None:
            # For bias, we use a higher precision (int32)
            # bias_scale = input_scale * weight_scale (computed later)
            # For now, store as int32 with weight_scale
            bias_int32 = np.round(node.bias / scale).astype(np.int32)

        # Store quantized data
        node.weight_int8 = weight_int8
        node.bias_int32 = bias_int32
        node.weight_scale = scale
        node.weight_zero_point = zero_point
        node.quantized = True

        # Compute quantization error
        dequant = weight_int8.astype(np.float32) * scale
        error = np.mean(np.abs(weight - dequant))
        self._log(f"  Quantized {node.name}: scale={scale:.6f}, "
                  f"range=[{w_min:.4f}, {w_max:.4f}], "
                  f"mean_abs_error={error:.6f}")

    def _compute_activation_params(self, node: IRNode):
        """Compute activation quantization parameters."""
        # Default: assume activations in range [0, 6] for ReLU-like activations
        # or [-128, 127] for general activations
        if node.op_type in (OpType.RELU, OpType.FUSED_CONV_RELU, OpType.FUSED_LINEAR_RELU):
            # ReLU outputs are always non-negative, use asymmetric [0, 6]
            act_min, act_max = 0.0, 6.0
        elif node.op_type == OpType.SOFTMAX:
            act_min, act_max = 0.0, 1.0
        else:
            # General range
            act_min, act_max = -6.0, 6.0

        # Check calibration stats if available
        if node.name in self.calibration_stats:
            act_min = self.calibration_stats[node.name]['min']
            act_max = self.calibration_stats[node.name]['max']

        # Compute scale and zero_point
        if act_max - act_min == 0:
            scale = 1.0 / 255.0
        else:
            scale = (act_max - act_min) / 255.0

        zero_point = int(np.clip(np.round(-act_min / scale), 0, 255))

        node.input_scale = scale
        node.input_zero_point = zero_point
        node.output_scale = scale
        node.output_zero_point = zero_point

    def calibrate(self, model, sample_inputs):
        """
        Run calibration to determine activation ranges.
        
        Args:
            model: The PyTorch model (for forward pass).
            sample_inputs: List of sample input tensors.
        """
        import torch

        self._log("Running calibration...")
        activation_ranges = {}

        # Register hooks to capture activation ranges
        hooks = []
        def make_hook(name):
            def hook_fn(module, input, output):
                if isinstance(output, torch.Tensor):
                    out_np = output.detach().cpu().numpy()
                    if name not in activation_ranges:
                        activation_ranges[name] = {'min': float('inf'), 'max': float('-inf')}
                    activation_ranges[name]['min'] = min(activation_ranges[name]['min'], float(out_np.min()))
                    activation_ranges[name]['max'] = max(activation_ranges[name]['max'], float(out_np.max()))
            return hook_fn

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:
                hooks.append(module.register_forward_hook(make_hook(name)))

        # Run forward passes
        model.eval()
        with torch.no_grad():
            for inp in sample_inputs:
                model(inp)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Map module names to IR node names
        for node in self.graph.get_compute_nodes():
            for mod_name, ranges in activation_ranges.items():
                if mod_name in node.name or node.name in mod_name:
                    self.calibration_stats[node.name] = ranges
                    break

        self._log(f"  Calibrated {len(activation_ranges)} layers")

    def _print_quantization_summary(self):
        """Print a summary of quantization results."""
        if not self.verbose:
            return

        total_orig = 0
        total_quant = 0
        print("\n" + "=" * 70)
        print("QUANTIZATION SUMMARY")
        print("=" * 70)
        print(f"{'Node':20s} | {'Orig (KB)':>10s} | {'Quant (KB)':>10s} | {'Ratio':>8s}")
        print("-" * 55)

        for node in self.graph.get_parameterized_nodes():
            if node.eliminated:
                continue
            orig_size = node.weight.nbytes + (node.bias.nbytes if node.bias is not None else 0)
            quant_size = node.weight_memory_bytes()
            total_orig += orig_size
            total_quant += quant_size
            ratio = orig_size / quant_size if quant_size > 0 else 0
            print(f"{node.name:20s} | {orig_size/1024:10.2f} | {quant_size/1024:10.2f} | {ratio:7.1f}x")

        print("-" * 55)
        ratio = total_orig / total_quant if total_quant > 0 else 0
        print(f"{'TOTAL':20s} | {total_orig/1024:10.2f} | {total_quant/1024:10.2f} | {ratio:7.1f}x")
        print(f"\nQuantization complete! {total_orig/1024:.1f} KB → {total_quant/1024:.1f} KB")

    def _log(self, msg: str):
        if self.verbose:
            print(f"[Quantizer] {msg}")


def simulate_quantized_inference(ir_graph: IRGraph, input_data: np.ndarray) -> np.ndarray:
    """
    Simulate quantized inference in Python for validation.
    Returns the output as a numpy array.
    """
    activations = {}

    for node in ir_graph.nodes:
        if node.eliminated:
            continue

        if node.op_type == OpType.PLACEHOLDER:
            # Quantize input
            scale = 1.0 / 255.0 * 12.0  # Assume input range [-6, 6]
            zp = 128
            activations[node.name] = np.clip(np.round(input_data / scale) + zp, 0, 255).astype(np.uint8)
            continue

        if node.op_type == OpType.OUTPUT:
            return activations.get(node.inputs[0], input_data)

        # Get input activation
        inp = activations.get(node.inputs[0]) if node.inputs else None
        if inp is None:
            continue

        if node.op_type in (OpType.CONV2D, OpType.FUSED_CONV_RELU):
            out = _sim_conv2d_int8(inp, node)
            if node.op_type == OpType.FUSED_CONV_RELU:
                out = np.maximum(out, 0)
            activations[node.name] = out

        elif node.op_type in (OpType.LINEAR, OpType.FUSED_LINEAR_RELU):
            inp_flat = inp.reshape(inp.shape[0], -1).astype(np.int32)
            w = node.weight_int8.astype(np.int32) if node.weight_int8 is not None else node.weight.astype(np.int32)
            out = inp_flat @ w.T
            if node.bias_int32 is not None:
                out += node.bias_int32
            if node.op_type == OpType.FUSED_LINEAR_RELU:
                out = np.maximum(out, 0)
            activations[node.name] = out

        elif node.op_type == OpType.RELU:
            activations[node.name] = np.maximum(inp, 0)

        elif node.op_type == OpType.MAXPOOL2D:
            activations[node.name] = _sim_maxpool2d(inp, node)

        elif node.op_type == OpType.FLATTEN:
            activations[node.name] = inp.reshape(inp.shape[0], -1)

        elif node.op_type == OpType.SOFTMAX:
            # Approximate softmax for int8
            inp_f = inp.astype(np.float32)
            exp_x = np.exp(inp_f - np.max(inp_f, axis=-1, keepdims=True))
            activations[node.name] = exp_x / np.sum(exp_x, axis=-1, keepdims=True)

        else:
            activations[node.name] = inp

    return input_data


def _sim_conv2d_int8(inp, node):
    """Simplified int8 conv2d simulation."""
    # This is a simplified simulation - full implementation would be more complex
    return inp  # Placeholder for simulation


def _sim_maxpool2d(inp, node):
    """Simplified maxpool2d simulation."""
    if len(inp.shape) != 4:
        return inp
    ks = node.params.get('kernel_size', (2, 2))
    if isinstance(ks, int):
        ks = (ks, ks)
    N, C, H, W = inp.shape
    oH, oW = H // ks[0], W // ks[1]
    out = np.zeros((N, C, oH, oW), dtype=inp.dtype)
    for i in range(oH):
        for j in range(oW):
            out[:, :, i, j] = inp[:, :, i*ks[0]:(i+1)*ks[0], j*ks[1]:(j+1)*ks[1]].max(axis=(-2, -1))
    return out


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
    ir = INT8Quantizer(ir).quantize()
    print("\n✅ Quantization test passed!")
