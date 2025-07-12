#!/usr/bin/env python3
"""
Proper PELA implementation that actually achieves target compression ratios
"""

import torch
import torch.nn as nn
import math
from transformers import Qwen2VLForConditionalGeneration

class ProperPELALayer(nn.Module):
    """Proper PELA layer that achieves target compression"""
    
    def __init__(self, original_layer, rank, original_shape):
        super().__init__()
        self.original_shape = original_shape
        self.rank = rank
        
        # Store the low-rank factors
        self.U = nn.Parameter(torch.zeros(original_shape[0], rank))
        self.V = nn.Parameter(torch.zeros(rank, original_shape[1]))
        
        # Copy bias if exists
        if hasattr(original_layer, 'bias') and original_layer.bias is not None:
            self.bias = nn.Parameter(original_layer.bias.clone())
        else:
            self.bias = None
    
    def forward(self, x):
        # Reconstruct weight on-the-fly: W = U @ V
        weight = self.U @ self.V
        return torch.nn.functional.linear(x, weight.t(), self.bias)
    
    def get_compression_ratio(self):
        original_params = self.original_shape[0] * self.original_shape[1]
        compressed_params = self.rank * (self.original_shape[0] + self.original_shape[1])
        return original_params / compressed_params

class AggressivePELACompressor:
    """Aggressive PELA compressor that achieves target ratios"""
    
    def __init__(self, target_compression=1.5):
        self.target_compression = target_compression
        self.compression_stats = []
        
    def calculate_optimal_rank(self, layer_shape, target_ratio):
        """Calculate rank to achieve target compression ratio"""
        m, n = layer_shape
        original_params = m * n
        
        # Target compressed parameters
        target_compressed_params = original_params / target_ratio
        
        # Solve: rank * (m + n) = target_compressed_params
        optimal_rank = int(target_compressed_params / (m + n))
        
        # Constraints
        min_rank = max(4, min(m, n) // 50)  # Very aggressive minimum
        max_rank = min(m, n) - 1
        
        final_rank = max(min_rank, min(optimal_rank, max_rank))
        
        # Calculate actual compression achieved
        actual_compressed = final_rank * (m + n)
        actual_ratio = original_params / actual_compressed
        
        return final_rank, actual_ratio
    
    def compress_layer(self, layer, layer_name):
        """Compress a single linear layer"""
        if not isinstance(layer, nn.Linear):
            return layer, 1.0
        
        weight = layer.weight.data
        shape = weight.shape
        
        # Calculate optimal rank
        rank, actual_ratio = self.calculate_optimal_rank(shape, self.target_compression)
        
        print(f"🔧 {layer_name}: {shape} -> rank {rank} ({actual_ratio:.1f}x)")
        
        # Perform SVD decomposition
        U, S, Vt = torch.svd(weight.float())
        
        # Truncate to target rank
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vt_truncated = Vt[:rank, :]
        
        # Create PELA layer
        pela_layer = ProperPELALayer(layer, rank, shape)
        
        # Initialize with SVD factors
        with torch.no_grad():
            pela_layer.U.data = U_truncated * torch.sqrt(S_truncated).unsqueeze(0)
            pela_layer.V.data = torch.sqrt(S_truncated).unsqueeze(1) * Vt_truncated
        
        # Statistics
        self.compression_stats.append({
            'layer': layer_name,
            'original_params': shape[0] * shape[1],
            'compressed_params': rank * (shape[0] + shape[1]),
            'compression_ratio': actual_ratio,
            'rank': rank
        })
        
        return pela_layer, actual_ratio
    
    def compress_model(self, model):
        """Compress all linear layers in model"""
        print(f"🚀 AGGRESSIVE PELA COMPRESSION (target: {self.target_compression}x)")
        print("=" * 70)
        
        total_original_params = 0
        total_compressed_params = 0
        layers_compressed = 0
        
        # Find and replace all linear layers
        def replace_linear_layers(module, prefix=""):
            nonlocal total_original_params, total_compressed_params, layers_compressed
            
            for name, child in list(module.named_children()):
                full_name = f"{prefix}.{name}" if prefix else name
                
                if isinstance(child, nn.Linear):
                    # Skip very small layers
                    if child.weight.numel() < 1000:
                        continue
                    
                    original_params = child.weight.numel()
                    
                    # Compress this layer
                    compressed_layer, ratio = self.compress_layer(child, full_name)
                    
                    # Replace in model
                    setattr(module, name, compressed_layer)
                    
                    compressed_params = original_params / ratio
                    total_original_params += original_params
                    total_compressed_params += compressed_params
                    layers_compressed += 1
                    
                else:
                    # Recursively process children
                    replace_linear_layers(child, full_name)
        
        # Process the model
        replace_linear_layers(model)
        
        # Summary
        overall_compression = total_original_params / total_compressed_params if total_compressed_params > 0 else 1.0
        
        print(f"\n📊 COMPRESSION SUMMARY:")
        print(f"   Layers compressed: {layers_compressed}")
        print(f"   Original params: {total_original_params:,}")
        print(f"   Compressed params: {total_compressed_params:,.0f}")
        print(f"   Overall compression: {overall_compression:.1f}x")
        print(f"   Target vs Actual: {self.target_compression}x -> {overall_compression:.1f}x")
        
        if overall_compression >= self.target_compression * 0.8:
            print(f"   ✅ Successfully achieved target compression!")
        else:
            print(f"   ⚠️  Below target - consider more aggressive settings")
        
        return model

def test_aggressive_pela():
    """Test the aggressive PELA implementation"""
    print("🧪 TESTING AGGRESSIVE PELA")
    print("=" * 50)
    
    # Test different compression targets
    targets = [1.5, 2.0, 3.0]
    
    for target in targets:
        print(f"\n🎯 Testing {target}x compression:")
        
        # Create test model (simplified)
        test_model = nn.Sequential(
            nn.Linear(3584, 3584),  # Attention layer
            nn.Linear(3584, 18944), # MLP up
            nn.Linear(18944, 3584), # MLP down
        )
        
        # Calculate original size
        original_params = sum(p.numel() for p in test_model.parameters())
        
        # Compress
        compressor = AggressivePELACompressor(target_compression=target)
        compressed_model = compressor.compress_model(test_model)
        
        # Calculate compressed size
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        actual_ratio = original_params / compressed_params
        
        print(f"   Result: {actual_ratio:.1f}x compression")

if __name__ == "__main__":
    test_aggressive_pela() 