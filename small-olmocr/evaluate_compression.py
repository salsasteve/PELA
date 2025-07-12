#!/usr/bin/env python3
"""
Evaluate PELA Compression Quality on OLMoOCR Model

This script loads both original and compressed models to compare:
- Individual layer differences
- Overall model performance
- Compression quality metrics
"""

import torch
import torch.nn as nn
import gc
import json
import argparse
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Any
import numpy as np
import psutil
import os

from layer_comparison import LayerComparator
from pela_olmocr import compress_model_with_pela, load_model_safe


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def find_linear_layers(model: nn.Module, prefix: str = "") -> Dict[str, nn.Linear]:
    """Find all Linear layers in the model."""
    linear_layers = {}
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, nn.Linear):
            linear_layers[full_name] = module
        elif hasattr(module, 'linear_l') and hasattr(module, 'linear_r'):
            # This is a PELA layer - treat as a single unit
            linear_layers[full_name] = module
        else:
            linear_layers.update(find_linear_layers(module, full_name))
    return linear_layers


def sample_layers_for_comparison(layer_dict: Dict[str, nn.Linear], 
                               max_layers: int = 20) -> Dict[str, nn.Linear]:
    """Sample a subset of layers for detailed comparison."""
    if len(layer_dict) <= max_layers:
        return layer_dict
    
    # Sample strategically: include some large and some small layers
    sorted_layers = sorted(layer_dict.items(), 
                          key=lambda x: x[1].weight.numel(), reverse=True)
    
    # Take some of the largest, some medium, and some smallest
    large_count = max_layers // 3
    medium_count = max_layers // 3
    small_count = max_layers - large_count - medium_count
    
    selected = {}
    
    # Largest layers
    for i in range(min(large_count, len(sorted_layers))):
        name, layer = sorted_layers[i]
        selected[name] = layer
    
    # Medium layers
    mid_start = len(sorted_layers) // 3
    for i in range(mid_start, min(mid_start + medium_count, len(sorted_layers))):
        name, layer = sorted_layers[i]
        selected[name] = layer
    
    # Smallest layers
    for i in range(max(0, len(sorted_layers) - small_count), len(sorted_layers)):
        name, layer = sorted_layers[i]
        selected[name] = layer
    
    return selected


def compare_models(original_model_path: str, compressed_model_path: str,
                  sample_size: int = 20, output_dir: str = "compression_analysis") -> Dict[str, Any]:
    """
    Compare original and compressed models.
    
    Args:
        original_model_path: Path to original model
        compressed_model_path: Path to compressed model
        sample_size: Number of layers to compare in detail
        output_dir: Directory to save analysis results
        
    Returns:
        Comprehensive comparison results
    """
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    print("🔍 Loading models for comparison...")
    print(f"📊 Memory before loading: {get_memory_usage():.2f} GB")
    
    # Load original model
    print("Loading original model...")
    original_model = load_model_safe(original_model_path)
    print(f"📊 Memory after original model: {get_memory_usage():.2f} GB")
    
    # Load compressed model
    print("Loading compressed model...")
    compressed_model = torch.load(compressed_model_path, map_location='cpu', weights_only=False)
    print(f"📊 Memory after compressed model: {get_memory_usage():.2f} GB")
    
    # Find linear layers
    print("🔍 Finding linear layers...")
    original_layers = find_linear_layers(original_model)
    compressed_layers = find_linear_layers(compressed_model)
    
    print(f"Found {len(original_layers)} original layers and {len(compressed_layers)} compressed layers")
    
    # Sample layers for comparison
    if len(original_layers) > sample_size:
        print(f"Sampling {sample_size} layers for detailed comparison...")
        sampled_original = sample_layers_for_comparison(original_layers, sample_size)
    else:
        sampled_original = original_layers
    
    # Initialize comparator
    comparator = LayerComparator()
    
    # Compare layers
    print("🔍 Comparing layers...")
    comparison_reports = []
    
    for layer_name, original_layer in tqdm(sampled_original.items(), desc="Comparing layers"):
        if layer_name in compressed_layers:
            compressed_layer = compressed_layers[layer_name]
            
            # Check if compressed layer has PELA structure
            if hasattr(compressed_layer, 'linear_l') and hasattr(compressed_layer, 'linear_r'):
                try:
                    report = comparator.comprehensive_comparison(
                        original_layer, compressed_layer, layer_name
                    )
                    comparison_reports.append(report)
                except Exception as e:
                    print(f"❌ Failed to compare {layer_name}: {e}")
            elif isinstance(compressed_layer, nn.Linear):
                # Regular linear layer (not compressed)
                print(f"ℹ️  Layer {layer_name} was not compressed")
            else:
                print(f"⚠️  Layer {layer_name} has unknown type: {type(compressed_layer)}")
        
        # Memory cleanup
        if len(comparison_reports) % 5 == 0:
            gc.collect()
    
    # Calculate overall statistics
    print("📊 Calculating overall statistics...")
    overall_stats = calculate_overall_stats(comparison_reports)
    
    # Create visualizations
    print("📊 Creating visualizations...")
    plot_path = f"{output_dir}/layer_comparison.png"
    comparator.plot_comparison_summary(comparison_reports, plot_path)
    
    # Save detailed results
    results = {
        'overall_stats': overall_stats,
        'layer_reports': comparison_reports,
        'model_info': {
            'original_layers': len(original_layers),
            'compressed_layers': len(compressed_layers),
            'sampled_layers': len(comparison_reports)
        }
    }
    
    results_path = f"{output_dir}/comparison_results.json"
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_friendly_results = convert_numpy_to_lists(results)
        json.dump(json_friendly_results, f, indent=2)
    
    print(f"📊 Results saved to {results_path}")
    print(f"📊 Visualization saved to {plot_path}")
    
    return results


def convert_numpy_to_lists(obj):
    """Convert numpy arrays to lists for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_lists(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_lists(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    else:
        return obj


def calculate_overall_stats(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate overall statistics from layer comparison reports."""
    stats = {}
    
    # Weight metrics
    weight_metrics = [r['weight_metrics'] for r in reports 
                     if 'weight_metrics' in r and 'error' not in r['weight_metrics']]
    
    if weight_metrics:
        stats['weight_stats'] = {
            'mean_frobenius_error': np.mean([m['relative_frobenius_error'] for m in weight_metrics]),
            'std_frobenius_error': np.std([m['relative_frobenius_error'] for m in weight_metrics]),
            'mean_cosine_similarity': np.mean([m['cosine_similarity'] for m in weight_metrics]),
            'std_cosine_similarity': np.std([m['cosine_similarity'] for m in weight_metrics]),
            'mean_correlation': np.mean([m['correlation'] for m in weight_metrics]),
            'std_correlation': np.std([m['correlation'] for m in weight_metrics])
        }
    
    # Forward pass metrics
    forward_metrics = [r['forward_metrics'] for r in reports 
                      if 'forward_metrics' in r and 'error' not in r['forward_metrics']]
    
    if forward_metrics:
        stats['forward_stats'] = {
            'mean_mse': np.mean([m['forward_mse_mean'] for m in forward_metrics]),
            'std_mse': np.std([m['forward_mse_mean'] for m in forward_metrics]),
            'mean_cosine_similarity': np.mean([m['forward_cosine_mean'] for m in forward_metrics]),
            'std_cosine_similarity': np.std([m['forward_cosine_mean'] for m in forward_metrics])
        }
    
    # Singular value metrics
    sv_metrics = [r['singular_value_metrics'] for r in reports 
                 if 'singular_value_metrics' in r and 'error' not in r['singular_value_metrics']]
    
    if sv_metrics:
        stats['singular_value_stats'] = {
            'mean_energy_preservation': np.mean([m['energy_preservation'] for m in sv_metrics]),
            'std_energy_preservation': np.std([m['energy_preservation'] for m in sv_metrics]),
            'mean_rank_ratio': np.mean([m['rank_ratio'] for m in sv_metrics]),
            'std_rank_ratio': np.std([m['rank_ratio'] for m in sv_metrics]),
            'mean_spectral_norm_ratio': np.mean([m['spectral_norm_ratio'] for m in sv_metrics]),
            'std_spectral_norm_ratio': np.std([m['spectral_norm_ratio'] for m in sv_metrics])
        }
    
    return stats


def print_summary_report(results: Dict[str, Any]):
    """Print a human-readable summary of the comparison results."""
    print("\n" + "="*60)
    print("🎯 PELA COMPRESSION QUALITY SUMMARY")
    print("="*60)
    
    overall = results['overall_stats']
    
    if 'weight_stats' in overall:
        ws = overall['weight_stats']
        print(f"\n📊 WEIGHT SIMILARITY:")
        print(f"  Average Cosine Similarity: {ws['mean_cosine_similarity']:.4f} ± {ws['std_cosine_similarity']:.4f}")
        print(f"  Average Correlation: {ws['mean_correlation']:.4f} ± {ws['std_correlation']:.4f}")
        print(f"  Average Frobenius Error: {ws['mean_frobenius_error']:.4f} ± {ws['std_frobenius_error']:.4f}")
    
    if 'forward_stats' in overall:
        fs = overall['forward_stats']
        print(f"\n🔄 FORWARD PASS ACCURACY:")
        print(f"  Average Output Cosine Similarity: {fs['mean_cosine_similarity']:.4f} ± {fs['std_cosine_similarity']:.4f}")
        print(f"  Average Output MSE: {fs['mean_mse']:.6f} ± {fs['std_mse']:.6f}")
    
    if 'singular_value_stats' in overall:
        svs = overall['singular_value_stats']
        print(f"\n🔬 SINGULAR VALUE PRESERVATION:")
        print(f"  Average Energy Preservation: {svs['mean_energy_preservation']:.4f} ± {svs['std_energy_preservation']:.4f}")
        print(f"  Average Rank Ratio: {svs['mean_rank_ratio']:.4f} ± {svs['std_rank_ratio']:.4f}")
        print(f"  Average Spectral Norm Ratio: {svs['mean_spectral_norm_ratio']:.4f} ± {svs['std_spectral_norm_ratio']:.4f}")
    
    print(f"\n📋 MODEL INFO:")
    print(f"  Total Original Layers: {results['model_info']['original_layers']}")
    print(f"  Total Compressed Layers: {results['model_info']['compressed_layers']}")
    print(f"  Layers Analyzed: {results['model_info']['sampled_layers']}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description="Evaluate PELA compression quality")
    parser.add_argument("--original", required=True, help="Path to original model")
    parser.add_argument("--compressed", required=True, help="Path to compressed model")
    parser.add_argument("--sample-size", type=int, default=20, help="Number of layers to compare")
    parser.add_argument("--output-dir", default="compression_analysis", help="Output directory")
    
    args = parser.parse_args()
    
    # Run comparison
    results = compare_models(
        args.original,
        args.compressed,
        args.sample_size,
        args.output_dir
    )
    
    # Print summary
    print_summary_report(results)


if __name__ == "__main__":
    main() 