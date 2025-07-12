#!/usr/bin/env python3
"""
Layer Comparison Tool - Measure differences between original and compressed layers
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

class LayerComparator:
    """Compare original and compressed layers across multiple metrics."""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def compare_weights(self, original_layer: nn.Linear, compressed_layer: nn.Module) -> Dict[str, float]:
        """
        Compare weight matrices between original and compressed layers.
        
        Args:
            original_layer: Original nn.Linear layer
            compressed_layer: PELA compressed layer
            
        Returns:
            Dictionary of comparison metrics
        """
        # Get original weight matrix
        W_orig = original_layer.weight.detach()
        
        # Reconstruct weight from compressed layer
        if hasattr(compressed_layer, 'linear_l') and hasattr(compressed_layer, 'linear_r'):
            # PELA layer: W ≈ W_r @ W_l
            W_compressed = compressed_layer.linear_r.weight @ compressed_layer.linear_l.weight
        else:
            raise ValueError("Compressed layer doesn't have expected PELA structure")
        
        # Ensure same device and dtype
        W_orig = W_orig.to(W_compressed.device).to(W_compressed.dtype)
        
        # Calculate metrics
        metrics = {}
        
        # 1. Frobenius norm metrics
        diff = W_orig - W_compressed
        frobenius_error = torch.norm(diff, 'fro').item()
        frobenius_orig = torch.norm(W_orig, 'fro').item()
        metrics['frobenius_error'] = frobenius_error
        metrics['relative_frobenius_error'] = frobenius_error / frobenius_orig if frobenius_orig > 0 else 0
        
        # 2. Element-wise metrics
        metrics['mse'] = torch.nn.functional.mse_loss(W_compressed, W_orig).item()
        metrics['mae'] = torch.nn.functional.l1_loss(W_compressed, W_orig).item()
        
        # 3. Statistical metrics
        metrics['max_absolute_error'] = torch.max(torch.abs(diff)).item()
        metrics['std_error'] = torch.std(diff).item()
        
        # 4. Correlation metrics
        W_orig_flat = W_orig.flatten()
        W_comp_flat = W_compressed.flatten()
        correlation = torch.corrcoef(torch.stack([W_orig_flat, W_comp_flat]))[0, 1].item()
        metrics['correlation'] = correlation if not torch.isnan(torch.tensor(correlation)) else 0.0
        
        # 5. Cosine similarity
        cosine_sim = torch.nn.functional.cosine_similarity(
            W_orig_flat.unsqueeze(0), W_comp_flat.unsqueeze(0)
        ).item()
        metrics['cosine_similarity'] = cosine_sim
        
        # 6. Spectral norm metrics
        try:
            spectral_orig = torch.norm(W_orig, ord=2).item()
            spectral_comp = torch.norm(W_compressed, ord=2).item()
            metrics['spectral_error'] = abs(spectral_orig - spectral_comp)
            metrics['relative_spectral_error'] = abs(spectral_orig - spectral_comp) / spectral_orig if spectral_orig > 0 else 0
        except:
            metrics['spectral_error'] = float('nan')
            metrics['relative_spectral_error'] = float('nan')
        
        return metrics
    
    def compare_forward_pass(self, original_layer: nn.Linear, compressed_layer: nn.Module, 
                           num_samples: int = 100, input_range: Tuple[float, float] = (-1.0, 1.0)) -> Dict[str, float]:
        """
        Compare forward pass outputs between original and compressed layers.
        
        Args:
            original_layer: Original nn.Linear layer
            compressed_layer: PELA compressed layer
            num_samples: Number of random input samples to test
            input_range: Range for random input generation
            
        Returns:
            Dictionary of forward pass comparison metrics
        """
        device = next(original_layer.parameters()).device
        in_features = original_layer.in_features
        
        metrics = {}
        errors = []
        cosine_sims = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                # Generate random input
                x = torch.randn(1, in_features, device=device) * (input_range[1] - input_range[0]) + input_range[0]
                
                # Forward pass through both layers
                y_orig = original_layer(x)
                y_comp = compressed_layer(x)
                
                # Calculate metrics for this sample
                mse = torch.nn.functional.mse_loss(y_comp, y_orig).item()
                errors.append(mse)
                
                # Cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    y_orig.flatten().unsqueeze(0), y_comp.flatten().unsqueeze(0)
                ).item()
                cosine_sims.append(cos_sim)
        
        # Aggregate metrics
        metrics['forward_mse_mean'] = np.mean(errors)
        metrics['forward_mse_std'] = np.std(errors)
        metrics['forward_mse_max'] = np.max(errors)
        metrics['forward_cosine_mean'] = np.mean(cosine_sims)
        metrics['forward_cosine_std'] = np.std(cosine_sims)
        metrics['forward_cosine_min'] = np.min(cosine_sims)
        
        return metrics
    
    def compare_singular_values(self, original_layer: nn.Linear, compressed_layer: nn.Module) -> Dict[str, Any]:
        """
        Compare singular value decompositions of original and compressed layers.
        
        Args:
            original_layer: Original nn.Linear layer
            compressed_layer: PELA compressed layer
            
        Returns:
            Dictionary containing singular value analysis
        """
        W_orig = original_layer.weight.detach().cpu()
        W_comp = (compressed_layer.linear_r.weight @ compressed_layer.linear_l.weight).detach().cpu()
        
        # Compute SVDs
        try:
            U_orig, S_orig, Vh_orig = torch.linalg.svd(W_orig, full_matrices=False)
            U_comp, S_comp, Vh_comp = torch.linalg.svd(W_comp, full_matrices=False)
            
            metrics = {}
            metrics['original_rank'] = torch.sum(S_orig > 1e-10).item()
            metrics['compressed_rank'] = torch.sum(S_comp > 1e-10).item()
            metrics['rank_ratio'] = metrics['compressed_rank'] / metrics['original_rank'] if metrics['original_rank'] > 0 else 0
            
            # Compare largest singular values
            min_len = min(len(S_orig), len(S_comp))
            if min_len > 0:
                sv_diff = torch.abs(S_orig[:min_len] - S_comp[:min_len])
                metrics['singular_value_mse'] = torch.mean(sv_diff ** 2).item()
                metrics['singular_value_max_diff'] = torch.max(sv_diff).item()
                
                # Spectral norm preservation
                metrics['spectral_norm_orig'] = S_orig[0].item()
                metrics['spectral_norm_comp'] = S_comp[0].item()
                metrics['spectral_norm_ratio'] = S_comp[0].item() / S_orig[0].item() if S_orig[0] > 0 else 0
                
                # Energy preservation (sum of squared singular values)
                energy_orig = torch.sum(S_orig ** 2).item()
                energy_comp = torch.sum(S_comp ** 2).item()
                metrics['energy_preservation'] = energy_comp / energy_orig if energy_orig > 0 else 0
            
            # Return singular values for plotting
            metrics['singular_values_original'] = S_orig.numpy()
            metrics['singular_values_compressed'] = S_comp.numpy()
            
            return metrics
            
        except Exception as e:
            print(f"SVD comparison failed: {e}")
            return {'error': str(e)}
    
    def comprehensive_comparison(self, original_layer: nn.Linear, compressed_layer: nn.Module,
                               layer_name: str = "layer") -> Dict[str, Any]:
        """
        Perform comprehensive comparison between original and compressed layers.
        
        Args:
            original_layer: Original nn.Linear layer
            compressed_layer: PELA compressed layer
            layer_name: Name of the layer for reporting
            
        Returns:
            Complete comparison report
        """
        print(f"🔍 Analyzing {layer_name}...")
        
        report = {
            'layer_name': layer_name,
            'layer_shape': (original_layer.in_features, original_layer.out_features)
        }
        
        # Weight comparison
        try:
            weight_metrics = self.compare_weights(original_layer, compressed_layer)
            report['weight_metrics'] = weight_metrics
        except Exception as e:
            print(f"❌ Weight comparison failed: {e}")
            report['weight_metrics'] = {'error': str(e)}
        
        # Forward pass comparison
        try:
            forward_metrics = self.compare_forward_pass(original_layer, compressed_layer)
            report['forward_metrics'] = forward_metrics
        except Exception as e:
            print(f"❌ Forward pass comparison failed: {e}")
            report['forward_metrics'] = {'error': str(e)}
        
        # Singular value comparison
        try:
            sv_metrics = self.compare_singular_values(original_layer, compressed_layer)
            report['singular_value_metrics'] = sv_metrics
        except Exception as e:
            print(f"❌ Singular value comparison failed: {e}")
            report['singular_value_metrics'] = {'error': str(e)}
        
        return report
    
    def plot_comparison_summary(self, reports: List[Dict[str, Any]], save_path: str = "layer_comparison.png"):
        """
        Create visualization of comparison results across multiple layers.
        
        Args:
            reports: List of comparison reports from comprehensive_comparison
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Extract metrics
        layer_names = [r['layer_name'] for r in reports if 'weight_metrics' in r and 'error' not in r['weight_metrics']]
        
        if not layer_names:
            print("No valid reports to plot")
            return
        
        # 1. Relative Frobenius Error
        frobenius_errors = [r['weight_metrics']['relative_frobenius_error'] for r in reports 
                           if 'weight_metrics' in r and 'relative_frobenius_error' in r['weight_metrics']]
        if frobenius_errors:
            axes[0, 0].bar(range(len(frobenius_errors)), frobenius_errors)
            axes[0, 0].set_title('Relative Frobenius Error')
            axes[0, 0].set_ylabel('Error')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Cosine Similarity
        cosine_sims = [r['weight_metrics']['cosine_similarity'] for r in reports 
                      if 'weight_metrics' in r and 'cosine_similarity' in r['weight_metrics']]
        if cosine_sims:
            axes[0, 1].bar(range(len(cosine_sims)), cosine_sims)
            axes[0, 1].set_title('Weight Cosine Similarity')
            axes[0, 1].set_ylabel('Similarity')
            axes[0, 1].set_ylim([0, 1])
        
        # 3. Forward Pass MSE
        forward_mses = [r['forward_metrics']['forward_mse_mean'] for r in reports 
                       if 'forward_metrics' in r and 'forward_mse_mean' in r['forward_metrics']]
        if forward_mses:
            axes[0, 2].bar(range(len(forward_mses)), forward_mses)
            axes[0, 2].set_title('Forward Pass MSE')
            axes[0, 2].set_ylabel('MSE')
            axes[0, 2].set_yscale('log')
        
        # 4. Energy Preservation
        energy_preservations = [r['singular_value_metrics']['energy_preservation'] for r in reports 
                              if 'singular_value_metrics' in r and 'energy_preservation' in r['singular_value_metrics']]
        if energy_preservations:
            axes[1, 0].bar(range(len(energy_preservations)), energy_preservations)
            axes[1, 0].set_title('Energy Preservation')
            axes[1, 0].set_ylabel('Ratio')
            axes[1, 0].set_ylim([0, 1])
        
        # 5. Rank Comparison
        rank_ratios = [r['singular_value_metrics']['rank_ratio'] for r in reports 
                      if 'singular_value_metrics' in r and 'rank_ratio' in r['singular_value_metrics']]
        if rank_ratios:
            axes[1, 1].bar(range(len(rank_ratios)), rank_ratios)
            axes[1, 1].set_title('Rank Preservation')
            axes[1, 1].set_ylabel('Ratio')
            axes[1, 1].set_ylim([0, 1])
        
        # 6. Singular Values Comparison (for first layer)
        if reports and 'singular_value_metrics' in reports[0]:
            sv_orig = reports[0]['singular_value_metrics'].get('singular_values_original', [])
            sv_comp = reports[0]['singular_value_metrics'].get('singular_values_compressed', [])
            if len(sv_orig) > 0 and len(sv_comp) > 0:
                min_len = min(len(sv_orig), len(sv_comp), 50)  # Plot first 50
                axes[1, 2].semilogy(sv_orig[:min_len], 'b-', label='Original', alpha=0.7)
                axes[1, 2].semilogy(sv_comp[:min_len], 'r--', label='Compressed', alpha=0.7)
                axes[1, 2].set_title(f'Singular Values: {layer_names[0]}')
                axes[1, 2].set_xlabel('Index')
                axes[1, 2].set_ylabel('Singular Value')
                axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plot saved to {save_path}")
        plt.close()


def test_layer_comparison():
    """Test the layer comparison functionality."""
    from pela_olmocr import PELALinear
    
    print("🧪 Testing layer comparison functionality...")
    
    # Create a test layer
    original_layer = nn.Linear(512, 256)
    pela_layer = PELALinear(original_layer, rank=32, compress_ratio=3.0, is_approximate=True)
    
    # Compare
    comparator = LayerComparator()
    report = comparator.comprehensive_comparison(original_layer, pela_layer, "test_layer")
    
    # Print results
    print("\n📊 COMPARISON RESULTS:")
    if 'weight_metrics' in report:
        wm = report['weight_metrics']
        print(f"Weight Metrics:")
        print(f"  Relative Frobenius Error: {wm.get('relative_frobenius_error', 'N/A'):.6f}")
        print(f"  Cosine Similarity: {wm.get('cosine_similarity', 'N/A'):.6f}")
        print(f"  Correlation: {wm.get('correlation', 'N/A'):.6f}")
    
    if 'forward_metrics' in report:
        fm = report['forward_metrics']
        print(f"Forward Pass Metrics:")
        print(f"  Mean MSE: {fm.get('forward_mse_mean', 'N/A'):.6f}")
        print(f"  Mean Cosine Similarity: {fm.get('forward_cosine_mean', 'N/A'):.6f}")
    
    if 'singular_value_metrics' in report:
        svm = report['singular_value_metrics']
        print(f"Singular Value Metrics:")
        print(f"  Energy Preservation: {svm.get('energy_preservation', 'N/A'):.6f}")
        print(f"  Rank Ratio: {svm.get('rank_ratio', 'N/A'):.6f}")
    
    return report


if __name__ == "__main__":
    test_layer_comparison() 