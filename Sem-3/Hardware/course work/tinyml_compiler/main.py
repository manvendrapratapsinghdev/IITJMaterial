#!/usr/bin/env python3
"""
TinyML Compiler - Main Pipeline
=================================
PyTorch-to-TinyML Lightweight Compiler for MCU Deployment.

Pipeline:
  PyTorch Model → Graph Extraction → IR Builder → Optimizer →
  Quantization → Memory Planner → Latency Estimator → C Code Generator

Usage:
  python main.py                    # Run with default MNIST CNN
  python main.py --model cifar10    # Run with CIFAR-10 model
  python main.py --model custom --model-path model.pt  # Custom model
"""

import argparse
import os
import sys
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from frontend.graph_extractor import GraphExtractor
from ir.ir_builder import IRBuilder
from optimizer.graph_optimizer import GraphOptimizer
from quantization.int8_quantizer import INT8Quantizer
from memory.memory_planner import MemoryPlanner
from latency.latency_estimator import LatencyEstimator
from backend.c_generator import CCodeGenerator
from models.sample_cnn import SimpleCNN_MNIST, TinyCNN_CIFAR10, MODELS


def print_banner():
    """Print a nice banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║          TinyML Compiler v0.1.0                              ║
║          PyTorch → Optimized INT8 C Code for MCU             ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def compile_model(
    model: torch.nn.Module,
    sample_input: torch.Tensor,
    output_dir: str = "generated",
    max_ram: int = 256 * 1024,
    clock_mhz: int = 80,
    verbose: bool = True,
):
    """
    Run the complete TinyML compilation pipeline.
    
    Args:
        model: PyTorch nn.Module to compile.
        sample_input: Sample input tensor for shape inference.
        output_dir: Directory for generated C code.
        max_ram: Maximum MCU RAM in bytes (default 256KB).
        clock_mhz: MCU clock frequency in MHz (default 80).
        verbose: Enable verbose logging.
    
    Returns:
        Dictionary with compilation results and reports.
    """
    results = {}
    total_start = time.time()

    # ============================================
    # Step 1: Graph Extraction
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 1: Graph Extraction")
    print("=" * 60)
    t0 = time.time()

    extractor = GraphExtractor(model, sample_input)
    graph = extractor.extract()

    t1 = time.time()
    print(graph.summary())
    print(f"\nTime  Graph extraction: {(t1-t0)*1000:.1f} ms")
    results['graph'] = graph
    results['num_nodes'] = len(graph.nodes)

    # ============================================
    # Step 2: IR Builder
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 2: IR Construction")
    print("=" * 60)
    t0 = time.time()

    builder = IRBuilder(model, graph)
    ir_graph = builder.build()

    t1 = time.time()
    print(ir_graph.summary())
    print(f"\nTime  IR construction: {(t1-t0)*1000:.1f} ms")
    results['ir_graph_before'] = ir_graph

    # Calculate original model size
    orig_size = sum(p.numel() * p.element_size() for p in model.parameters())
    results['original_size'] = orig_size

    # ============================================
    # Step 3: Graph Optimization
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 3: Graph Optimization")
    print("=" * 60)
    t0 = time.time()

    optimizer = GraphOptimizer(ir_graph, verbose=verbose)
    ir_graph = optimizer.optimize()

    t1 = time.time()
    print(f"\nOptimized IR:")
    print(ir_graph.summary())
    print(f"\nTime  Optimization: {(t1-t0)*1000:.1f} ms")
    results['optimizer_stats'] = optimizer.stats

    # ============================================
    # Step 4: INT8 Quantization
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 4: INT8 Quantization")
    print("=" * 60)
    t0 = time.time()

    quantizer = INT8Quantizer(ir_graph, verbose=verbose)
    ir_graph = quantizer.quantize()

    t1 = time.time()
    quantized_size = ir_graph.total_weight_memory()
    results['quantized_size'] = quantized_size
    print(f"\nTime  Quantization: {(t1-t0)*1000:.1f} ms")

    # ============================================
    # Step 5: Memory Planning
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 5: Memory Planning")
    print("=" * 60)
    t0 = time.time()

    planner = MemoryPlanner(ir_graph, max_ram=max_ram, verbose=verbose)
    memory_report = planner.plan()

    t1 = time.time()
    results['memory_report'] = memory_report
    print(f"\nTime  Memory planning: {(t1-t0)*1000:.1f} ms")

    # ============================================
    # Step 6: Latency Estimation
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 6: Latency Estimation")
    print("=" * 60)
    t0 = time.time()

    estimator = LatencyEstimator(ir_graph, clock_mhz=clock_mhz, verbose=verbose)
    latency_report = estimator.estimate()

    t1 = time.time()
    results['latency_report'] = latency_report
    print(f"\nTime  Latency estimation: {(t1-t0)*1000:.1f} ms")

    # ============================================
    # Step 7: C Code Generation
    # ============================================
    print("\n" + "=" * 60)
    print("STEP 7: C Code Generation")
    print("=" * 60)
    t0 = time.time()

    generator = CCodeGenerator(ir_graph, output_dir=output_dir, verbose=verbose)
    main_file = generator.generate()

    t1 = time.time()
    results['output_dir'] = output_dir
    results['main_file'] = main_file
    print(f"\nTime  Code generation: {(t1-t0)*1000:.1f} ms")

    # ============================================
    # Final Summary
    # ============================================
    total_time = time.time() - total_start
    print_final_summary(model, results, memory_report, latency_report, total_time)

    return results


def print_final_summary(model, results, memory_report, latency_report, total_time):
    """Print the final compilation summary."""
    print("\n")
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║                   COMPILATION SUMMARY                       ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Model:           {model.__class__.__name__:40s}  ║")
    print(f"║  Original Size:   {results['original_size']/1024:>10.1f} KB (FP32)                    ║")
    print(f"║  Quantized Size:  {results['quantized_size']/1024:>10.1f} KB (INT8)                    ║")
    ratio = results['original_size'] / results['quantized_size'] if results['quantized_size'] > 0 else 0
    print(f"║  Compression:     {ratio:>10.1f}x                                ║")
    print(f"║  Peak RAM Usage:  {memory_report.total_peak_memory/1024:>10.1f} KB                             ║")
    print(f"║  RAM Limit:       {memory_report.max_ram/1024:>10.1f} KB                             ║")
    status = "PASS" if memory_report.passed else "FAIL"
    print(f"║  Status:          {status:40s}  ║")
    print(f"║  Latency:         {latency_report.total_time_ms:>10.2f} ms @ {latency_report.clock_mhz} MHz             ║")
    print(f"║  Total Cycles:    {latency_report.total_cycles:>10,}                             ║")
    print(f"║  Compile Time:    {total_time*1000:>10.1f} ms                             ║")
    print(f"║  Output:          {results['output_dir']:40s}  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Optimization comparison table
    stats = results.get('optimizer_stats', {})
    print("\nOptimization Comparison:")
    print(f"{'Metric':30s} | {'Before':>12s} | {'After':>12s}")
    print("-" * 60)
    print(f"{'Model Size':30s} | {results['original_size']/1024:>10.1f} KB | {results['quantized_size']/1024:>10.1f} KB")
    print(f"{'Nodes Fused':30s} | {'':>12s} | {stats.get('fused', 0):>12d}")
    print(f"{'Nodes Eliminated':30s} | {'':>12s} | {stats.get('eliminated', 0):>12d}")
    print(f"{'Constants Folded':30s} | {'':>12s} | {stats.get('folded', 0):>12d}")
    print(f"{'Peak RAM':30s} | {'N/A':>12s} | {memory_report.total_peak_memory/1024:>10.1f} KB")


def main():
    parser = argparse.ArgumentParser(description="TinyML Compiler - PyTorch to C for MCU")
    parser.add_argument('--model', type=str, default='mnist',
                        choices=['mnist', 'cifar10'],
                        help='Model to compile (default: mnist)')
    parser.add_argument('--output', type=str, default='generated',
                        help='Output directory for generated C code')
    parser.add_argument('--max-ram', type=int, default=256 * 1024,
                        help='Maximum MCU RAM in bytes (default: 256KB)')
    parser.add_argument('--clock', type=int, default=80,
                        help='MCU clock frequency in MHz (default: 80)')
    parser.add_argument('--quiet', action='store_true',
                        help='Reduce output verbosity')

    args = parser.parse_args()

    print_banner()

    # Select model
    if args.model == 'mnist':
        model = SimpleCNN_MNIST()
        sample_input = SimpleCNN_MNIST.get_sample_input()
        print(f"Using model: SimpleCNN_MNIST (MNIST 28x28)")
    elif args.model == 'cifar10':
        model = TinyCNN_CIFAR10()
        sample_input = TinyCNN_CIFAR10.get_sample_input()
        print(f" Using model: TinyCNN_CIFAR10 (CIFAR-10 32x32)")
    else:
        print(f" Unknown model: {args.model}")
        sys.exit(1)

    # Run compilation
    results = compile_model(
        model=model,
        sample_input=sample_input,
        output_dir=args.output,
        max_ram=args.max_ram,
        clock_mhz=args.clock,
        verbose=not args.quiet,
    )

    print(f"\n Compilation complete! Generated files in: {args.output}/")
    print(f"   To test: gcc -DTEST_MODEL -o model {args.output}/model.c && ./model")


if __name__ == "__main__":
    main()
