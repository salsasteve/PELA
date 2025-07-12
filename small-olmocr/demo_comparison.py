#!/usr/bin/env python3
"""
Demo: Layer Comparison Tools for PELA Compression

This script demonstrates different ways to measure compression quality:
1. Test layer comparison functionality
2. Compare individual layers  
3. Evaluate full model compression
"""

import torch
import torch.nn as nn
from layer_comparison import LayerComparator, test_layer_comparison
from pela_olmocr import PELALinear
import numpy as np


def demo_individual_layer_comparison():
    """Demonstrate comparison of individual layers."""
    print("=" * 60)
    print("🧪 DEMO 1: Individual Layer Comparison")
    print("=" * 60)
    
    # Create test layers with different sizes
    test_configs = [
        (512, 256, 32),   # Medium layer
        (1024, 512, 64),  # Large layer  
        (256, 128, 16),   # Small layer
    ]
    
    comparator = LayerComparator()
    
    for i, (in_feat, out_feat, rank) in enumerate(test_configs):
        print(f"\n📊 Test {i+1}: {in_feat}×{out_feat} layer (rank {rank})")
        
        # Create original layer
        original = nn.Linear(in_feat, out_feat)
        nn.init.xavier_normal_(original.weight)
        
        # Create PELA compressed version
        compressed = PELALinear(original, rank=rank, compress_ratio=3.0, is_approximate=True)
        
        # Compare
        report = comparator.comprehensive_comparison(
            original, compressed, f"test_layer_{i+1}"
        )
        
        # Print key metrics
        if 'weight_metrics' in report:
            wm = report['weight_metrics']
            print(f"  ✅ Weight Cosine Similarity: {wm['cosine_similarity']:.4f}")
            print(f"  ✅ Relative Frobenius Error: {wm['relative_frobenius_error']:.4f}")
        
        if 'forward_metrics' in report:
            fm = report['forward_metrics']
            print(f"  ✅ Forward Pass Cosine Similarity: {fm['forward_cosine_mean']:.4f}")
        
        if 'singular_value_metrics' in report:
            svm = report['singular_value_metrics']
            print(f"  ✅ Energy Preservation: {svm['energy_preservation']:.4f}")
            print(f"  ✅ Rank Ratio: {svm['rank_ratio']:.4f}")
        
        # Calculate parameter reduction
        original_params = original.weight.numel() + (original.bias.numel() if original.bias is not None else 0)
        compressed_params = (compressed.linear_l.weight.numel() + 
                           compressed.linear_r.weight.numel() +
                           (compressed.linear_r.bias.numel() if compressed.linear_r.bias is not None else 0))
        
        reduction = ((original_params - compressed_params) / original_params) * 100
        ratio = original_params / compressed_params
        
        print(f"  📦 Parameter Reduction: {reduction:.1f}% ({ratio:.1f}x compression)")


def demo_compression_quality_analysis():
    """Demonstrate analysis of compression quality across different ranks."""
    print("\n" + "=" * 60)
    print("🔬 DEMO 2: Compression Quality vs Rank Analysis")
    print("=" * 60)
    
    # Test different compression levels
    layer_size = (1024, 512)  # Fixed layer size
    ranks = [16, 32, 64, 128, 256]
    
    print(f"📊 Analyzing {layer_size[0]}×{layer_size[1]} layer with different ranks...")
    
    # Create original layer
    original = nn.Linear(*layer_size)
    nn.init.xavier_normal_(original.weight)
    
    comparator = LayerComparator()
    results = []
    
    for rank in ranks:
        # Create compressed version
        compressed = PELALinear(original, rank=rank, compress_ratio=3.0, is_approximate=True)
        
        # Compare
        report = comparator.comprehensive_comparison(original, compressed, f"rank_{rank}")
        
        # Extract key metrics
        result = {'rank': rank}
        
        if 'weight_metrics' in report:
            result['cosine_similarity'] = report['weight_metrics']['cosine_similarity']
            result['frobenius_error'] = report['weight_metrics']['relative_frobenius_error']
        
        if 'forward_metrics' in report:
            result['forward_similarity'] = report['forward_metrics']['forward_cosine_mean']
            result['forward_mse'] = report['forward_metrics']['forward_mse_mean']
        
        if 'singular_value_metrics' in report:
            result['energy_preservation'] = report['singular_value_metrics']['energy_preservation']
            result['rank_ratio'] = report['singular_value_metrics']['rank_ratio']
        
        # Calculate compression metrics
        original_params = original.weight.numel() + (original.bias.numel() if original.bias is not None else 0)
        compressed_params = (compressed.linear_l.weight.numel() + 
                           compressed.linear_r.weight.numel() +
                           (compressed.linear_r.bias.numel() if compressed.linear_r.bias is not None else 0))
        
        result['compression_ratio'] = original_params / compressed_params
        result['parameter_reduction'] = ((original_params - compressed_params) / original_params) * 100
        
        results.append(result)
    
    # Print results table
    print(f"\n📋 RESULTS TABLE:")
    print(f"{'Rank':<6} {'Compression':<12} {'Cosine Sim':<12} {'Frobenius':<12} {'Energy':<10}")
    print("-" * 60)
    
    for result in results:
        print(f"{result['rank']:<6} "
              f"{result.get('compression_ratio', 0):<12.1f} "
              f"{result.get('cosine_similarity', 0):<12.4f} "
              f"{result.get('frobenius_error', 0):<12.4f} "
              f"{result.get('energy_preservation', 0):<10.4f}")
    
    # Analysis
    print(f"\n🔍 ANALYSIS:")
    best_rank = max(results, key=lambda x: x.get('cosine_similarity', 0))['rank']
    highest_compression = max(results, key=lambda x: x.get('compression_ratio', 0))
    
    print(f"  📈 Best quality (highest cosine similarity): Rank {best_rank}")
    print(f"  📦 Highest compression: Rank {highest_compression['rank']} "
          f"({highest_compression.get('compression_ratio', 0):.1f}x)")
    
    # Find good trade-off (high similarity, decent compression)
    good_tradeoffs = [r for r in results if r.get('cosine_similarity', 0) > 0.95 
                     and r.get('compression_ratio', 0) > 2.0]
    if good_tradeoffs:
        best_tradeoff = max(good_tradeoffs, key=lambda x: x.get('compression_ratio', 0))
        print(f"  ⚖️  Best trade-off: Rank {best_tradeoff['rank']} "
              f"(similarity: {best_tradeoff.get('cosine_similarity', 0):.4f}, "
              f"compression: {best_tradeoff.get('compression_ratio', 0):.1f}x)")


def demo_functional_impact():
    """Demonstrate how compression affects actual function output."""
    print("\n" + "=" * 60)
    print("🎯 DEMO 3: Functional Impact Analysis")  
    print("=" * 60)
    
    # Create a layer and compress it
    layer_size = (512, 256)
    original = nn.Linear(*layer_size)
    nn.init.xavier_normal_(original.weight)
    
    compressed = PELALinear(original, rank=64, compress_ratio=3.0, is_approximate=True)
    
    # Test with different types of inputs
    test_cases = [
        ("Random Normal", lambda: torch.randn(10, layer_size[0])),
        ("Random Uniform", lambda: torch.rand(10, layer_size[0]) * 2 - 1),
        ("Zeros", lambda: torch.zeros(10, layer_size[0])),
        ("Ones", lambda: torch.ones(10, layer_size[0])),
        ("Large Values", lambda: torch.randn(10, layer_size[0]) * 10),
    ]
    
    print(f"📊 Testing functional impact on {layer_size[0]}×{layer_size[1]} layer (rank 64)")
    print(f"{'Input Type':<15} {'Output MSE':<12} {'Cosine Sim':<12} {'Max Diff':<12}")
    print("-" * 55)
    
    with torch.no_grad():
        for test_name, input_generator in test_cases:
            # Generate test input
            x = input_generator()
            
            # Forward pass through both layers
            y_original = original(x)
            y_compressed = compressed(x)
            
            # Calculate metrics
            mse = nn.functional.mse_loss(y_compressed, y_original).item()
            cosine_sim = nn.functional.cosine_similarity(
                y_original.flatten().unsqueeze(0), 
                y_compressed.flatten().unsqueeze(0)
            ).item()
            max_diff = torch.max(torch.abs(y_original - y_compressed)).item()
            
            print(f"{test_name:<15} {mse:<12.6f} {cosine_sim:<12.4f} {max_diff:<12.6f}")
    
    print(f"\n🔍 INTERPRETATION:")
    print(f"  📊 MSE: Lower is better (measures average squared difference)")
    print(f"  📊 Cosine Similarity: Higher is better (1.0 = perfect alignment)")
    print(f"  📊 Max Difference: Shows worst-case output difference")


def main():
    """Run all demonstration examples."""
    print("🚀 PELA Layer Comparison Tool Demonstrations")
    print("=" * 60)
    
    # Demo 1: Basic layer comparison
    demo_individual_layer_comparison()
    
    # Demo 2: Compression quality vs rank
    demo_compression_quality_analysis()
    
    # Demo 3: Functional impact
    demo_functional_impact()
    
    # Demo 4: Test the built-in functionality
    print("\n" + "=" * 60)
    print("🧪 DEMO 4: Built-in Test Functionality")
    print("=" * 60)
    test_layer_comparison()
    
    print("\n" + "=" * 60)
    print("✅ All demonstrations completed!")
    print("=" * 60)
    print("\n💡 NEXT STEPS:")
    print("  1. Run full model compression with evaluation:")
    print("     python pela_olmocr.py")
    print("  2. Compare original vs compressed models:")
    print("     python evaluate_compression.py --original <path> --compressed <path>")
    print("  3. Analyze individual layers in your own models:")
    print("     from layer_comparison import LayerComparator")


if __name__ == "__main__":
    main() 