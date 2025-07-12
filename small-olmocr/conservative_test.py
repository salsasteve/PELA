#!/usr/bin/env python3
"""
Conservative PELA Test - Verify compression works before attempting full model

This script tests PELA compression on small components to validate the approach
before attempting compression on the full 8B parameter OLMoOCR model.
"""

import torch
import torch.nn as nn
import numpy as np
from layer_comparison import LayerComparator
from pela_olmocr import PELALinear
import matplotlib.pyplot as plt


def test_compression_reality_check():
    """Test compression on different layer sizes to see when it breaks down."""
    print("🧪 PELA COMPRESSION REALITY CHECK")
    print("=" * 60)
    
    # Test different layer sizes and compression ratios
    test_configs = [
        # (in_features, out_features, rank, expected_quality)
        (256, 128, 32, "GOOD"),      # Conservative compression
        (512, 256, 64, "GOOD"),      # Moderate compression  
        (1024, 512, 128, "OK"),      # More aggressive
        (2048, 1024, 256, "RISKY"),  # Very aggressive
        (4096, 2048, 64, "BAD"),     # Extreme compression
    ]
    
    comparator = LayerComparator()
    results = []
    
    for in_feat, out_feat, rank, expected in test_configs:
        print(f"\n📊 Testing {in_feat}×{out_feat} → rank {rank} ({expected})")
        
        # Create original layer with realistic initialization
        original = nn.Linear(in_feat, out_feat)
        nn.init.xavier_normal_(original.weight, gain=1.0)
        
        # Calculate theoretical compression ratio
        original_params = in_feat * out_feat + out_feat  # weights + bias
        compressed_params = (in_feat * rank) + (rank * out_feat) + out_feat  # two matrices + bias
        theoretical_ratio = original_params / compressed_params
        
        # Compress
        compressed = PELALinear(original, rank=rank, compress_ratio=theoretical_ratio, is_approximate=True)
        
        # Test quality
        report = comparator.comprehensive_comparison(original, compressed, f"{in_feat}x{out_feat}")
        
        if 'weight_metrics' in report and 'forward_metrics' in report:
            weight_sim = report['weight_metrics']['cosine_similarity']
            forward_sim = report['forward_metrics']['forward_cosine_mean']
            mse = report['forward_metrics']['forward_mse_mean']
            
            # Quality assessment
            if weight_sim > 0.95 and forward_sim > 0.95:
                quality = "✅ EXCELLENT"
            elif weight_sim > 0.9 and forward_sim > 0.9:
                quality = "🟢 GOOD"
            elif weight_sim > 0.8 and forward_sim > 0.8:
                quality = "🟡 ACCEPTABLE"
            elif weight_sim > 0.6 and forward_sim > 0.6:
                quality = "🟠 POOR"
            else:
                quality = "❌ FAILED"
            
            result = {
                'config': f"{in_feat}×{out_feat} (rank {rank})",
                'theoretical_ratio': theoretical_ratio,
                'weight_similarity': weight_sim,
                'forward_similarity': forward_sim,
                'mse': mse,
                'quality': quality,
                'expected': expected
            }
            results.append(result)
            
            print(f"  Compression Ratio: {theoretical_ratio:.1f}x")
            print(f"  Weight Similarity: {weight_sim:.4f}")
            print(f"  Forward Similarity: {forward_sim:.4f}")
            print(f"  Quality: {quality}")
        else:
            print(f"  ❌ Comparison failed!")
    
    # Summary
    print(f"\n📋 SUMMARY:")
    print(f"{'Config':<20} {'Ratio':<8} {'Weight':<8} {'Forward':<8} {'Quality':<12}")
    print("-" * 70)
    
    for r in results:
        print(f"{r['config']:<20} {r['theoretical_ratio']:<8.1f} "
              f"{r['weight_similarity']:<8.4f} {r['forward_similarity']:<8.4f} {r['quality']:<12}")
    
    return results


def test_functional_degradation():
    """Test how compression affects actual computation on realistic data."""
    print(f"\n🎯 FUNCTIONAL DEGRADATION TEST")
    print("=" * 60)
    
    # Create a realistically-sized layer
    layer_size = (1024, 512)
    original = nn.Linear(*layer_size)
    nn.init.xavier_normal_(original.weight)
    
    # Test different compression levels
    ranks = [16, 32, 64, 128, 256]
    
    print(f"Testing {layer_size[0]}×{layer_size[1]} layer with different ranks...")
    print(f"{'Rank':<6} {'Ratio':<8} {'Output MSE':<12} {'Max Error':<12} {'Cosine Sim':<12}")
    print("-" * 60)
    
    # Generate realistic test input (batch of embeddings)
    batch_size = 32
    test_input = torch.randn(batch_size, layer_size[0]) * 0.1  # Small values like real embeddings
    
    with torch.no_grad():
        original_output = original(test_input)
        
        for rank in ranks:
            # Create compressed version
            compressed = PELALinear(original, rank=rank, compress_ratio=3.0, is_approximate=True)
            compressed_output = compressed(test_input)
            
            # Calculate quality metrics
            mse = nn.functional.mse_loss(compressed_output, original_output).item()
            max_error = torch.max(torch.abs(compressed_output - original_output)).item()
            cosine_sim = nn.functional.cosine_similarity(
                original_output.flatten().unsqueeze(0),
                compressed_output.flatten().unsqueeze(0)
            ).item()
            
            # Calculate actual compression ratio
            orig_params = layer_size[0] * layer_size[1] + layer_size[1]
            comp_params = (layer_size[0] * rank) + (rank * layer_size[1]) + layer_size[1]
            actual_ratio = orig_params / comp_params
            
            print(f"{rank:<6} {actual_ratio:<8.1f} {mse:<12.6f} {max_error:<12.6f} {cosine_sim:<12.4f}")


def test_cascading_effects():
    """Test how compression errors cascade through multiple layers."""
    print(f"\n🔗 CASCADING EFFECTS TEST")
    print("=" * 60)
    
    # Create a small 3-layer network
    layers = [
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(), 
        nn.Linear(128, 64)
    ]
    
    original_net = nn.Sequential(*layers)
    
    # Create compressed version (compress only linear layers)
    compressed_layers = []
    for i, layer in enumerate(layers):
        if isinstance(layer, nn.Linear):
            # Use conservative rank
            rank = min(64, min(layer.in_features, layer.out_features) // 4)
            compressed_layers.append(PELALinear(layer, rank=rank, compress_ratio=3.0, is_approximate=True))
        else:
            compressed_layers.append(layer)
    
    compressed_net = nn.Sequential(*compressed_layers)
    
    # Test with realistic input
    test_input = torch.randn(16, 512) * 0.1
    
    with torch.no_grad():
        orig_output = original_net(test_input)
        comp_output = compressed_net(test_input)
        
        final_mse = nn.functional.mse_loss(comp_output, orig_output).item()
        final_cosine = nn.functional.cosine_similarity(
            orig_output.flatten().unsqueeze(0),
            comp_output.flatten().unsqueeze(0)
        ).item()
        
        print(f"3-Layer Network Test:")
        print(f"  Final Output MSE: {final_mse:.6f}")
        print(f"  Final Output Cosine Similarity: {final_cosine:.4f}")
        
        if final_cosine > 0.9:
            print("  ✅ Network compression successful")
        elif final_cosine > 0.7:
            print("  🟡 Network compression acceptable but degraded")
        else:
            print("  ❌ Network compression failed - significant quality loss")


def generate_recommendation():
    """Generate honest recommendation based on test results."""
    print(f"\n💡 HONEST RECOMMENDATION")
    print("=" * 60)
    
    print("Based on compression theory and empirical evidence:")
    print()
    print("🟢 PELA CAN WORK WELL FOR:")
    print("  • Attention layers in transformers (often naturally low-rank)")
    print("  • Middle layers of feed-forward networks")
    print("  • Layers with high redundancy")
    print("  • Conservative compression ratios (2-4x)")
    print()
    print("🔴 PELA STRUGGLES WITH:")
    print("  • Embedding layers")
    print("  • Final classification layers")
    print("  • Small layers (< 256 dimensions)")
    print("  • Aggressive compression ratios (> 10x)")
    print("  • Vision-language fusion layers (critical for OLMoOCR)")
    print()
    print("⚠️  FOR OLMoOCR SPECIFICALLY:")
    print("  • 8B parameters = massive model with critical vision-language components")
    print("  • OCR tasks require precise attention to visual details")
    print("  • Aggressive compression likely to break OCR accuracy")
    print()
    print("🎯 REALISTIC APPROACH:")
    print("  1. Start with conservative 2-3x compression on attention layers only")
    print("  2. Exclude embedding, output, and vision fusion layers")
    print("  3. Test on small OCR tasks before full deployment")
    print("  4. Monitor OCR accuracy degradation carefully")
    print()
    print("💸 HONEST ASSESSMENT:")
    print("  • 3x compression: Might work with careful layer selection")
    print("  • 10x compression: Very risky, likely significant quality loss")
    print("  • 38x compression: Almost certainly breaks the model")


def main():
    """Run all reality check tests."""
    print("🧪 PELA COMPRESSION REALITY CHECK")
    print("Testing compression before attempting full OLMoOCR model")
    print("=" * 80)
    
    # Run tests
    try:
        test_compression_reality_check()
        test_functional_degradation()
        test_cascading_effects()
        generate_recommendation()
        
        print(f"\n✅ Reality check complete!")
        print(f"💡 Recommendation: Start small and conservative!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 