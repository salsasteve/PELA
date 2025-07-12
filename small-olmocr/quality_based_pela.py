#!/usr/bin/env python3
"""
Quality-based PELA compression - select ranks based on information retention
"""

import torch
import torch.nn as nn
import numpy as np
from transformers import Qwen2VLForConditionalGeneration

class QualityBasedPELALayer(nn.Module):
    """PELA layer with quality-based rank selection"""
    
    def __init__(self, original_layer, rank, original_shape, retained_energy):
        super().__init__()
        self.original_shape = original_shape
        self.rank = rank
        self.retained_energy = retained_energy
        
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
    
    def get_quality_info(self):
        return {
            'rank': self.rank,
            'compression_ratio': self.get_compression_ratio(),
            'retained_energy': self.retained_energy
        }

class QualityBasedPELACompressor:
    """PELA compressor that selects ranks based on information retention"""
    
    def __init__(self, target_retention=0.55):
        """
        Args:
            target_retention (float): Target information retention (0.5 = 50%, 0.6 = 60%)
        """
        self.target_retention = target_retention
        self.compression_stats = []
        
    def calculate_quality_based_rank(self, weight_matrix, target_retention):
        """Calculate rank based on singular value energy retention"""
        # Perform SVD
        U, S, Vt = torch.svd(weight_matrix.float())
        
        # Calculate energy (squared singular values)
        energy = S.pow(2)
        total_energy = energy.sum()
        
        # Find rank that retains target percentage of energy
        cumulative_energy = torch.cumsum(energy, dim=0)
        energy_ratios = cumulative_energy / total_energy
        
        # Find the minimum rank that achieves target retention
        target_idx = torch.where(energy_ratios >= target_retention)[0]
        
        if len(target_idx) > 0:
            optimal_rank = target_idx[0].item() + 1  # +1 because of 0-indexing
        else:
            optimal_rank = len(S)  # Use all ranks if target not achievable
        
        # Constraints
        min_rank = max(4, min(weight_matrix.shape) // 100)  # Very conservative minimum
        max_rank = min(weight_matrix.shape) - 1
        
        final_rank = max(min_rank, min(optimal_rank, max_rank))
        
        # Calculate actual retention achieved
        if final_rank <= len(energy):
            actual_retention = (energy[:final_rank].sum() / total_energy).item()
        else:
            actual_retention = 1.0
        
        return final_rank, actual_retention
    
    def analyze_layer_quality(self, original_weight, compressed_weight):
        """Analyze how well the compression preserves the original"""
        # Ensure same shape for comparison
        if compressed_weight.shape != original_weight.shape:
            print(f"      ⚠️  Shape mismatch: {original_weight.shape} vs {compressed_weight.shape}")
            return 0.0, 0.0
        
        # Frobenius norm similarity
        diff_norm = torch.norm(original_weight - compressed_weight, 'fro')
        orig_norm = torch.norm(original_weight, 'fro')
        frobenius_sim = 1 - (diff_norm / orig_norm) if orig_norm > 0 else 0.0
        
        # Cosine similarity (flattened)
        orig_flat = original_weight.flatten()
        comp_flat = compressed_weight.flatten()
        
        # Handle zero tensors
        if torch.norm(orig_flat) == 0 or torch.norm(comp_flat) == 0:
            cosine_sim = 0.0
        else:
            cosine_sim = torch.cosine_similarity(orig_flat, comp_flat, dim=0)
        
        return frobenius_sim.item(), cosine_sim.item()
    
    def compress_layer(self, layer, layer_name):
        """Compress a single linear layer based on quality retention"""
        if not isinstance(layer, nn.Linear):
            return layer, 1.0, {}
        
        weight = layer.weight.data
        shape = weight.shape
        
        # Calculate quality-based rank
        rank, actual_retention = self.calculate_quality_based_rank(weight, self.target_retention)
        
        print(f"🔧 {layer_name}: {shape} -> rank {rank}")
        print(f"   📊 Target retention: {self.target_retention:.1%} -> Actual: {actual_retention:.1%}")
        
        # Perform SVD decomposition (convert to float32 for numerical stability)
        U, S, Vt = torch.svd(weight.float())
        
        # Truncate to quality-based rank
        U_truncated = U[:, :rank]
        S_truncated = S[:rank]
        Vt_truncated = Vt[:rank, :]
        
        # Reconstruct compressed weight for quality analysis
        # Ensure proper matrix multiplication: (m x rank) @ (rank x n) = (m x n)
        compressed_weight = U_truncated @ torch.diag(S_truncated) @ Vt_truncated
        
        # Analyze quality preservation
        frobenius_sim, cosine_sim = self.analyze_layer_quality(weight, compressed_weight)
        
        # Create PELA layer
        pela_layer = QualityBasedPELALayer(layer, rank, shape, actual_retention)
        
        # Initialize with SVD factors (preserve original dtype)
        with torch.no_grad():
            U_init = (U_truncated * torch.sqrt(S_truncated).unsqueeze(0)).to(weight.dtype)
            V_init = (torch.sqrt(S_truncated).unsqueeze(1) * Vt_truncated).to(weight.dtype)
            pela_layer.U.data = U_init
            pela_layer.V.data = V_init
        
        # Calculate compression ratio
        original_params = shape[0] * shape[1]
        compressed_params = rank * (shape[0] + shape[1])
        compression_ratio = original_params / compressed_params
        
        print(f"   🎯 Compression: {compression_ratio:.1f}x")
        print(f"   📈 Frobenius similarity: {frobenius_sim:.3f}")
        print(f"   📈 Cosine similarity: {cosine_sim:.3f}")
        
        # Statistics
        quality_stats = {
            'layer': layer_name,
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compression_ratio,
            'rank': rank,
            'target_retention': self.target_retention,
            'actual_retention': actual_retention,
            'frobenius_similarity': frobenius_sim,
            'cosine_similarity': cosine_sim
        }
        
        self.compression_stats.append(quality_stats)
        
        return pela_layer, compression_ratio, quality_stats
    
    def compress_model(self, model):
        """Compress all linear layers in model using quality-based approach"""
        print(f"🎯 QUALITY-BASED PELA COMPRESSION")
        print(f"📊 Target information retention: {self.target_retention:.1%}")
        print("=" * 70)
        
        total_original_params = 0
        total_compressed_params = 0
        layers_compressed = 0
        quality_scores = []
        
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
                    compressed_layer, ratio, stats = self.compress_layer(child, full_name)
                    
                    # Replace in model
                    setattr(module, name, compressed_layer)
                    
                    compressed_params = original_params / ratio
                    total_original_params += original_params
                    total_compressed_params += compressed_params
                    layers_compressed += 1
                    
                    quality_scores.append(stats['cosine_similarity'])
                    
                else:
                    # Recursively process children
                    replace_linear_layers(child, full_name)
        
        # Process the model
        replace_linear_layers(model)
        
        # Summary
        overall_compression = total_original_params / total_compressed_params if total_compressed_params > 0 else 1.0
        avg_quality = np.mean(quality_scores) if quality_scores else 0
        
        print(f"\n📊 COMPRESSION SUMMARY:")
        print(f"   Layers compressed: {layers_compressed}")
        print(f"   Original params: {total_original_params:,}")
        print(f"   Compressed params: {total_compressed_params:,.0f}")
        print(f"   Overall compression: {overall_compression:.1f}x")
        print(f"   Average cosine similarity: {avg_quality:.3f}")
        print(f"   Target retention: {self.target_retention:.1%}")
        
        if avg_quality >= self.target_retention * 0.9:  # Allow some tolerance
            print(f"   ✅ Quality target achieved!")
        else:
            print(f"   ⚠️  Below quality target")
        
        return model

def test_quality_retention():
    """Test different quality retention levels"""
    print("🧪 TESTING QUALITY-BASED PELA")
    print("=" * 50)
    
    # Test different retention targets
    retentions = [0.50, 0.55, 0.60, 0.70]
    
    for retention in retentions:
        print(f"\n🎯 Testing {retention:.0%} information retention:")
        
        # Create test model
        test_model = nn.Sequential(
            nn.Linear(3584, 3584),  # Attention layer
            nn.Linear(3584, 18944), # MLP up
            nn.Linear(18944, 3584), # MLP down
        )
        
        # Calculate original size
        original_params = sum(p.numel() for p in test_model.parameters())
        
        # Compress
        compressor = QualityBasedPELACompressor(target_retention=retention)
        compressed_model = compressor.compress_model(test_model)
        
        # Calculate compressed size
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        actual_ratio = original_params / compressed_params
        
        # Get average quality
        avg_quality = np.mean([s['cosine_similarity'] for s in compressor.compression_stats])
        
        print(f"   Result: {actual_ratio:.1f}x compression, {avg_quality:.3f} avg similarity")

if __name__ == "__main__":
    test_quality_retention() 