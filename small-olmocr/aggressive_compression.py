#!/usr/bin/env python3
"""
Test aggressive PELA compression settings to achieve meaningful compression
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import Qwen2VLForConditionalGeneration
import os

class AggressivePELA:
    def __init__(self, target_compression=2.0):
        self.target_compression = target_compression
        self.compression_stats = []
    
    def calculate_aggressive_rank(self, layer_shape, target_compression):
        """Calculate rank for aggressive compression"""
        m, n = layer_shape
        
        # For target compression ratio, calculate max allowed rank
        original_params = m * n
        target_params = original_params / target_compression
        
        # SVD rank: r * (m + n)
        max_rank = int(target_params / (m + n))
        
        # Ensure minimum rank for functionality
        min_rank = max(8, min(m, n) // 20)
        
        return max(min_rank, min(max_rank, min(m, n) // 2))
    
    def compress_layer(self, weight_matrix, layer_name, target_compression):
        """Aggressively compress a single layer"""
        original_shape = weight_matrix.shape
        original_params = weight_matrix.numel()
        
        if len(original_shape) != 2:
            return weight_matrix, 1.0  # Skip non-linear layers
        
        # Calculate aggressive rank
        rank = self.calculate_aggressive_rank(original_shape, target_compression)
        
        # Perform SVD
        U, S, Vt = torch.svd(weight_matrix.float())
        
        # Truncate to aggressive rank
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vt_truncated = Vt[:rank, :]
        
        # Reconstruct
        compressed_weight = (U_truncated * S_truncated.unsqueeze(0)) @ Vt_truncated
        
        # Calculate compression ratio
        compressed_params = rank * (original_shape[0] + original_shape[1])
        compression_ratio = original_params / compressed_params
        
        print(f"   {layer_name}: {original_shape} -> rank {rank}")
        print(f"      {original_params:,} -> {compressed_params:,} params ({compression_ratio:.1f}x)")
        
        self.compression_stats.append({
            'layer': layer_name,
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compression_ratio,
            'rank': rank
        })
        
        return compressed_weight, compression_ratio

def test_aggressive_compression():
    """Test what level of compression is actually achievable"""
    
    print("🔥 AGGRESSIVE PELA COMPRESSION TEST")
    print("=" * 60)
    
    # Test different compression targets
    targets = [2.0, 3.0, 5.0, 10.0]
    
    for target in targets:
        print(f"\n🎯 Testing {target}x compression target...")
        print("-" * 40)
        
        # Create a PELA compressor
        pela = AggressivePELA(target_compression=target)
        
        # Test on a few representative layer sizes from OLMoOCR
        test_layers = [
            ("embed_tokens", (152064, 3584)),  # Embedding layer
            ("attention.q_proj", (3584, 3584)),  # Attention projection
            ("mlp.gate_proj", (3584, 18944)),   # MLP gate
            ("mlp.down_proj", (18944, 3584)),   # MLP down
        ]
        
        total_original = 0
        total_compressed = 0
        
        for name, shape in test_layers:
            # Create dummy weight matrix
            weight = torch.randn(shape)
            
            # Compress it
            compressed_weight, ratio = pela.compress_layer(weight, name, target)
            
            original_params = weight.numel()
            compressed_params = original_params / ratio
            
            total_original += original_params
            total_compressed += compressed_params
        
        overall_ratio = total_original / total_compressed
        print(f"\n   📊 Overall compression: {overall_ratio:.1f}x")
        print(f"   📊 Target vs Actual: {target}x -> {overall_ratio:.1f}x")
        
        if overall_ratio < target * 0.8:
            print(f"   ⚠️  Significantly below target!")
        elif overall_ratio >= target * 0.9:
            print(f"   ✅ Close to target!")

def estimate_real_compression():
    """Estimate compression on real OLMoOCR model"""
    print(f"\n🔍 REAL MODEL COMPRESSION ESTIMATE")
    print("=" * 60)
    
    # Layer counts from our debug output
    linear_layers = 347
    total_linear_params = 7.57e9  # From debug
    avg_linear_params = total_linear_params / linear_layers
    
    print(f"📋 Model stats:")
    print(f"   Linear layers: {linear_layers}")
    print(f"   Avg params per layer: {avg_linear_params/1e6:.1f}M")
    
    # Test compression on average layer size
    avg_layer_size = int(np.sqrt(avg_linear_params))  # Assume square-ish
    test_shape = (avg_layer_size, avg_layer_size)
    
    print(f"\n🧪 Testing on representative layer: {test_shape}")
    
    targets = [1.5, 2.0, 3.0, 5.0]
    for target in targets:
        # Estimate rank needed
        m, n = test_shape
        original_params = m * n
        target_params = original_params / target
        max_rank = int(target_params / (m + n))
        
        if max_rank > 0 and max_rank < min(m, n):
            compressed_params = max_rank * (m + n)
            actual_ratio = original_params / compressed_params
            
            # Estimate model-wide compression
            estimated_total_compressed = total_linear_params / actual_ratio
            total_params = 8.11e9  # Including non-linear
            other_params = total_params - total_linear_params
            final_total = estimated_total_compressed + other_params
            
            model_compression = total_params / final_total
            
            print(f"   {target}x target -> {model_compression:.1f}x model compression")
            print(f"      Model size: {final_total/1e9:.1f}B -> {final_total * 2 / 1024**3:.1f}GB (fp16)")
        else:
            print(f"   {target}x target -> Not achievable")

if __name__ == "__main__":
    test_aggressive_compression()
    estimate_real_compression() 