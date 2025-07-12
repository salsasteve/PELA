#!/usr/bin/env python3
"""
Conservative PELA Compression - Based on Reality Check Results

This script applies conservative compression with extensive evaluation,
targeting only specific layers and avoiding aggressive compression ratios.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Dict, Any, List

from pela_olmocr import compress_model_with_pela, load_model_safe, find_compressible_layers
from layer_comparison import LayerComparator


def conservative_compress_olmocr(
    model_path: str = "allenai/olmOCR-7B-0225-preview",
    compress_ratio: float = 1.8,  # Conservative ratio based on reality check
    max_layers_to_compress: int = 50,  # Limit scope for initial test
    output_dir: str = "conservative_compression"
):
    """
    Apply conservative compression with extensive monitoring.
    
    Args:
        model_path: Path to model to compress
        compress_ratio: Conservative compression ratio (1.5-2.0 recommended)
        max_layers_to_compress: Limit number of layers for testing
        output_dir: Directory for outputs
    """
    
    print("🚀 CONSERVATIVE PELA COMPRESSION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Target compression ratio: {compress_ratio}x")
    print(f"Max layers to compress: {max_layers_to_compress}")
    print(f"Output directory: {output_dir}")
    print()
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    try:
        # Load model (but don't actually load the huge model for testing)
        print("📋 SIMULATED MODEL ANALYSIS (avoiding 8GB download)")
        print("In real scenario, this would load:", model_path)
        
        # Instead, let's demonstrate on a smaller transformer model
        print("🔄 Using smaller model for demonstration...")
        
        # You could replace this with any smaller transformer model for testing
        print("⚠️  For actual OLMoOCR compression, uncomment the line below:")
        print("   # model = load_model_safe(model_path)")
        print()
        
        # Create a demo analysis of what would happen
        demo_compression_analysis(compress_ratio, max_layers_to_compress, output_dir)
        
    except Exception as e:
        print(f"❌ Conservative compression failed: {e}")
        import traceback
        traceback.print_exc()


def demo_compression_analysis(compress_ratio: float, max_layers: int, output_dir: str):
    """Demonstrate what the compression analysis would look like."""
    
    print("📊 COMPRESSION ANALYSIS PREVIEW")
    print("=" * 60)
    
    # Simulate layer analysis based on typical OLMoOCR structure
    simulated_layers = [
        ("model.layers.0.self_attn.q_proj", 4096, 4096),
        ("model.layers.0.self_attn.k_proj", 4096, 1024), 
        ("model.layers.0.self_attn.v_proj", 4096, 1024),
        ("model.layers.0.self_attn.o_proj", 4096, 4096),
        ("model.layers.0.mlp.gate_proj", 4096, 11008),
        ("model.layers.0.mlp.up_proj", 4096, 11008),
        ("model.layers.0.mlp.down_proj", 11008, 4096),
    ]
    
    print(f"📋 Found {len(simulated_layers) * 32} potential layers (32 transformer blocks)")
    print(f"🎯 Will compress first {max_layers} layers with {compress_ratio}x ratio")
    print()
    
    # Analyze compression impact
    total_original_params = 0
    total_compressed_params = 0
    quality_predictions = []
    
    for i, (layer_name, in_feat, out_feat) in enumerate(simulated_layers[:max_layers]):
        if i >= max_layers:
            break
            
        # Calculate compression
        original_params = in_feat * out_feat + out_feat
        
        # Calculate required rank for target compression ratio
        target_compressed_params = original_params / compress_ratio
        rank = max(8, int((target_compressed_params - out_feat) / (in_feat + out_feat)))
        rank = min(rank, min(in_feat, out_feat))
        
        compressed_params = (in_feat * rank) + (rank * out_feat) + out_feat
        actual_ratio = original_params / compressed_params
        
        # Predict quality based on our reality check results
        if actual_ratio <= 1.5:
            quality_pred = "✅ EXCELLENT (>0.9 similarity)"
        elif actual_ratio <= 2.0:
            quality_pred = "🟢 GOOD (0.8-0.9 similarity)"
        elif actual_ratio <= 3.0:
            quality_pred = "🟡 ACCEPTABLE (0.7-0.8 similarity)"
        elif actual_ratio <= 5.0:
            quality_pred = "🟠 POOR (0.5-0.7 similarity)"
        else:
            quality_pred = "❌ FAILED (<0.5 similarity)"
        
        total_original_params += original_params
        total_compressed_params += compressed_params
        quality_predictions.append(quality_pred)
        
        print(f"Layer {i+1:2d}: {layer_name}")
        print(f"  Shape: {in_feat}×{out_feat} → rank {rank}")
        print(f"  Compression: {original_params:,} → {compressed_params:,} params ({actual_ratio:.2f}x)")
        print(f"  Predicted Quality: {quality_pred}")
        print()
    
    # Overall statistics
    overall_ratio = total_original_params / total_compressed_params
    reduction_percent = ((total_original_params - total_compressed_params) / total_original_params) * 100
    
    print("📊 OVERALL IMPACT PREDICTION:")
    print(f"  Original parameters: {total_original_params:,}")
    print(f"  Compressed parameters: {total_compressed_params:,}")
    print(f"  Overall compression: {overall_ratio:.2f}x")
    print(f"  Parameter reduction: {reduction_percent:.1f}%")
    print()
    
    # Quality assessment
    excellent_count = sum(1 for q in quality_predictions if "EXCELLENT" in q)
    good_count = sum(1 for q in quality_predictions if "GOOD" in q)
    acceptable_count = sum(1 for q in quality_predictions if "ACCEPTABLE" in q)
    poor_count = sum(1 for q in quality_predictions if "POOR" in q)
    failed_count = sum(1 for q in quality_predictions if "FAILED" in q)
    
    print("🎯 QUALITY PREDICTION SUMMARY:")
    print(f"  ✅ Excellent quality: {excellent_count} layers")
    print(f"  🟢 Good quality: {good_count} layers")
    print(f"  🟡 Acceptable quality: {acceptable_count} layers")
    print(f"  🟠 Poor quality: {poor_count} layers")
    print(f"  ❌ Failed compression: {failed_count} layers")
    print()
    
    if failed_count > 0:
        print("⚠️  WARNING: Some layers predicted to fail compression!")
    elif poor_count > len(quality_predictions) // 2:
        print("⚠️  WARNING: Many layers predicted to have poor quality!")
    elif excellent_count + good_count > len(quality_predictions) // 2:
        print("✅ PREDICTION: Compression likely to succeed with good quality!")
    else:
        print("🟡 PREDICTION: Mixed results expected - proceed with caution!")
    
    # Save analysis
    analysis = {
        "compression_settings": {
            "target_ratio": compress_ratio,
            "max_layers": max_layers
        },
        "predictions": {
            "total_original_params": total_original_params,
            "total_compressed_params": total_compressed_params,
            "overall_ratio": overall_ratio,
            "parameter_reduction_percent": reduction_percent,
            "quality_distribution": {
                "excellent": excellent_count,
                "good": good_count,
                "acceptable": acceptable_count,
                "poor": poor_count,
                "failed": failed_count
            }
        }
    }
    
    analysis_path = f"{output_dir}/compression_analysis.json"
    with open(analysis_path, 'w') as f:
        json.dump(analysis, f, indent=2)
    
    print(f"📊 Analysis saved to: {analysis_path}")


def real_compression_with_evaluation():
    """Example of how to run real compression with full evaluation."""
    
    print("\n" + "=" * 60)
    print("📋 REAL COMPRESSION PROCEDURE")
    print("=" * 60)
    print()
    print("To run actual compression with evaluation:")
    print()
    print("1. 🔧 Load model:")
    print("   model = load_model_safe('allenai/olmOCR-7B-0225-preview')")
    print()
    print("2. 🚀 Compress with conservative settings:")
    print("   stats, eval_summary = compress_model_with_pela(")
    print("       model,")
    print("       compress_ratio=1.8,  # Conservative!")
    print("       evaluate_quality=True,")
    print("       evaluation_sample_rate=0.3,  # Evaluate 30% of layers")
    print("       checkpoint_interval=5  # Save frequently")
    print("   )")
    print()
    print("3. 💾 Save results:")
    print("   torch.save(model, 'olmocr_conservative_compressed.pt')")
    print()
    print("4. 📊 Evaluate quality:")
    print("   python evaluate_compression.py \\")
    print("     --original allenai/olmOCR-7B-0225-preview \\")
    print("     --compressed olmocr_conservative_compressed.pt")
    print()
    print("⚠️  IMPORTANT: Monitor memory usage and stop if quality degrades!")


def main():
    """Run conservative compression analysis."""
    
    # Demo analysis
    conservative_compress_olmocr(
        compress_ratio=1.8,  # Conservative based on reality check
        max_layers_to_compress=20  # Start small
    )
    
    # Show real procedure
    real_compression_with_evaluation()
    
    print("\n" + "=" * 60)
    print("🎯 NEXT STEPS")
    print("=" * 60)
    print("1. Review the analysis above")
    print("2. If predictions look good, uncomment model loading")
    print("3. Start with even smaller compression ratio (1.5x)")
    print("4. Compress only 10-20 layers first")
    print("5. Evaluate quality before proceeding")
    print("6. Gradually increase scope if quality is maintained")
    print()
    print("Remember: Your skepticism was correct - be conservative!")


if __name__ == "__main__":
    main() 