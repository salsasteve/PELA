#!/usr/bin/env python3
"""
PELA: Parameter-Efficient Learning with Low-rank Approximation for OLMoOCR
Enhanced version with batch processing, memory management, and evaluation capabilities
"""

import torch
import torch.nn as nn
import gc
import psutil
import os
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import json
from pathlib import Path

import copy
import numpy as np
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from transformers import PreTrainedModel, AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2_vl import Qwen2VLForConditionalGeneration
import logging
from tqdm import tqdm
import time

logger = logging.getLogger(__name__)


def load_model_safe(model_path: str, device_map: str = "auto"):
    """
    Safely load a model with proper error handling and device management.
    
    Args:
        model_path: Path to model (local or HuggingFace model name)
        device_map: Device mapping strategy
        
    Returns:
        Loaded model
    """
    try:
        print(f"🔍 Loading model: {model_path}")
        
        # For OLMoOCR models, try Qwen2VL architecture first
        if "olmOCR" in model_path or "olmo" in model_path.lower():
            try:
                model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_path,
                    device_map=device_map,
                    torch_dtype=torch.float16,
                    trust_remote_code=True
                )
                print(f"✅ Loaded as Qwen2VL model")
                return model
            except Exception as e:
                print(f"⚠️  Qwen2VL loading failed: {e}")
                print("🔄 Trying AutoModelForCausalLM...")
        
        # Fallback to general causal LM
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        print(f"✅ Loaded as AutoModelForCausalLM")
        return model
        
    except Exception as e:
        print(f"❌ Failed to load model {model_path}: {e}")
        raise


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    return psutil.Process(os.getpid()).memory_info().rss / 1e9


def cleanup_memory():
    """Aggressive memory cleanup."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def find_compressible_layers(model: nn.Module) -> List[Tuple[str, nn.Linear]]:
    """Find all compressible Linear layers in the model."""
    compressible = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Skip very small layers
            if module.weight.numel() > 1024:  # Only compress layers with >1K parameters
                compressible.append((name, module))
    return compressible


def get_nested_attr(obj, attr_path: str):
    """Get nested attribute using dot notation."""
    attrs = attr_path.split('.')
    for attr in attrs:
        obj = getattr(obj, attr)
    return obj


def set_nested_attr(obj, attr_path: str, value):
    """Set nested attribute using dot notation."""
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)


def compress_layer_optimized(layer: nn.Linear, compress_ratio: float, 
                           is_approximate: bool, device: torch.device) -> Tuple["PELALinear", Dict[str, Any]]:
    """Compress a single layer with optimized memory management."""
    # Calculate rank based on compression ratio
    original_params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
    
    # Conservative rank calculation to avoid OOM
    max_possible_rank = min(layer.in_features, layer.out_features)
    
    # Calculate rank that achieves desired compression ratio
    # original_params = in * out + out
    # compressed_params = in * rank + rank * out + out = rank * (in + out) + out
    # ratio = original_params / compressed_params
    target_compressed_params = original_params / compress_ratio
    # Solve: rank * (in + out) + out = target_compressed_params
    rank = max(1, int((target_compressed_params - layer.out_features) / (layer.in_features + layer.out_features)))
    rank = min(rank, max_possible_rank)
    rank = max(8, rank)  # Minimum rank of 8
    
    # Create compressed layer
    compressed_layer = PELALinear(layer, rank=rank, compress_ratio=compress_ratio, is_approximate=is_approximate)
    
    # Calculate actual compression achieved
    compressed_params = (compressed_layer.linear_l.weight.numel() + 
                        compressed_layer.linear_r.weight.numel() +
                        (compressed_layer.linear_r.bias.numel() if compressed_layer.linear_r.bias is not None else 0))
    
    actual_ratio = original_params / compressed_params
    
    info = {
        'rank': rank,
        'original_params': original_params,
        'compressed_params': compressed_params,
        'actual_ratio': actual_ratio
    }
    
    return compressed_layer, info


def create_smart_batches(compressible_layers: List[Tuple[str, nn.Linear]]) -> List[List[Tuple[str, nn.Linear]]]:
    """Create smart batches based on layer sizes."""
    # Sort by parameter count
    sorted_layers = sorted(compressible_layers, key=lambda x: x[1].weight.numel(), reverse=True)
    
    batches = []
    current_batch = []
    current_batch_params = 0
    
    # Target: ~100M parameters per batch
    target_batch_size = 100_000_000
    
    for layer_name, layer in sorted_layers:
        layer_params = layer.weight.numel()
        
        # Very large layers (>4096 dimensions) get their own batch
        if layer_params > 4_000_000:
            if current_batch:
                batches.append(current_batch)
                current_batch = []
                current_batch_params = 0
            batches.append([(layer_name, layer)])
        
        # Medium layers (>2048 dimensions) go in pairs
        elif layer_params > 1_000_000:
            if len(current_batch) >= 2 or current_batch_params + layer_params > target_batch_size:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [(layer_name, layer)]
                current_batch_params = layer_params
            else:
                current_batch.append((layer_name, layer))
                current_batch_params += layer_params
        
        # Small layers go in groups of up to 5
        else:
            if len(current_batch) >= 5 or current_batch_params + layer_params > target_batch_size:
                if current_batch:
                    batches.append(current_batch)
                current_batch = [(layer_name, layer)]
                current_batch_params = layer_params
            else:
                current_batch.append((layer_name, layer))
                current_batch_params += layer_params
    
    if current_batch:
        batches.append(current_batch)
    
    return batches


def low_rank_approximate(mat_org: torch.Tensor, rank: int = 32) -> Dict[str, torch.Tensor]:
    """
    Learning a low-rank decomposition for the given matrix using SVD.
    
    Args:
        mat_org: The original matrix to decompose
        rank: Target rank for decomposition
        
    Returns:
        Dictionary containing left matrix, right matrix, and reconstruction error
    """
    device = mat_org.device
    original_dtype = mat_org.dtype
    
    # Handle meta tensors (offloaded parameters)
    if device.type == 'meta':
        # For meta tensors, we need to move to a real device first
        target_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        mat_org = mat_org.to(target_device)
        device = target_device
    
    # Convert to float32 for SVD if needed (for numerical stability)
    computation_dtype = torch.float32 if mat_org.dtype == torch.float16 else mat_org.dtype
    if mat_org.dtype != computation_dtype:
        mat_org_compute = mat_org.to(computation_dtype)
    else:
        mat_org_compute = mat_org
    
    # Perform SVD on GPU if available, using PyTorch's native implementation
    with torch.no_grad():  # Save memory during SVD
        try:
            # Use PyTorch SVD - this can run on GPU!
            u, s, vh = torch.linalg.svd(mat_org_compute, full_matrices=False)
        except Exception as e:
            # Fallback to CPU if GPU SVD fails
            logger.warning(f"GPU SVD failed, falling back to CPU: {e}")
            mat_org_cpu = mat_org_compute.cpu()
            u, s, vh = torch.linalg.svd(mat_org_cpu, full_matrices=False)
            # Move results back to target device
            u, s, vh = u.to(device), s.to(device), vh.to(device)
    
    # Adjust rank if necessary
    actual_rank = min(rank, len(s))
    
    # Create low-rank approximation
    # For W = U @ S @ Vt, we want W ≈ W2 @ W1 where:
    # W1: (rank, in_features) for linear_l
    # W2: (out_features, rank) for linear_r
    s_sqrt = torch.sqrt(s[:actual_rank])
    mat_l = vh[:actual_rank, :] * s_sqrt.unsqueeze(1)  # (rank, in)
    mat_r = u[:, :actual_rank] * s_sqrt.unsqueeze(0)   # (out, rank)
    
    # Convert back to original dtype
    mat_l = mat_l.to(original_dtype)
    mat_r = mat_r.to(original_dtype)
    
    # Calculate reconstruction error (keep on same device)
    reconstructed = mat_r @ mat_l
    error = nn.functional.mse_loss(reconstructed.float(), mat_org_compute.float())
    
    # Ensure results are on the target device
    mat_l = mat_l.to(device)
    mat_r = mat_r.to(device)
    
    return {
        'mat_l': mat_l,  # (rank, in_features)
        'mat_r': mat_r,  # (out_features, rank)
        'error': error,
        'rank': actual_rank
    }


class PELALinear(nn.Module):
    """
    PELA replacement for nn.Linear using low-rank approximation.
    
    This module replaces a linear layer with two smaller linear layers
    that approximate the original through low-rank decomposition.
    """
    
    def __init__(self, 
                 original_linear: nn.Linear, 
                 rank: int,
                 compress_ratio: float = 3.0,
                 is_approximate: bool = True):
        super().__init__()
        
        self.in_features = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank = min(rank, min(self.in_features, self.out_features))
        
        # Create two linear layers for low-rank approximation
        self.linear_l = nn.Linear(self.in_features, self.rank, bias=False)
        self.linear_r = nn.Linear(self.rank, self.out_features, bias=original_linear.bias is not None)
        
        # Move to same device as original
        device = next(original_linear.parameters()).device
        self.linear_l = self.linear_l.to(device)
        self.linear_r = self.linear_r.to(device)
        
        if is_approximate:
            self._initialize_from_original(original_linear)
        
    def _initialize_from_original(self, original_linear: nn.Linear):
        """Initialize the low-rank matrices from the original linear layer."""
        with torch.no_grad():
            weight = original_linear.weight
            
            # Time the SVD computation and track device usage
            svd_start = time.time()
            original_device = weight.device
            lr_decomp = low_rank_approximate(weight, self.rank)
            svd_time = time.time() - svd_start
            
            # Set weights - mat_l is (rank, in_features), mat_r is (out_features, rank)
            self.linear_l.weight.data.copy_(lr_decomp['mat_l'])  # (rank, in) -> (rank, in)
            self.linear_r.weight.data.copy_(lr_decomp['mat_r'])  # (out, rank) -> (out, rank)
            
            # Copy bias if it exists
            if original_linear.bias is not None:
                self.linear_r.bias.data.copy_(original_linear.bias)
            
            # Log timing and device info for layers
            device_info = f"on {original_device}" if original_device.type != 'cpu' else "on CPU"
            if svd_time > 2.0:  # Log if SVD took more than 2 seconds
                logger.info(f"SVD took {svd_time:.1f}s {device_info} for {weight.shape} -> rank {lr_decomp['rank']}")
            
            logger.debug(f"Initialized PELA layer: {weight.shape} -> "
                        f"[{self.linear_l.weight.shape}, {self.linear_r.weight.shape}] "
                        f"with rank {lr_decomp['rank']}, error: {lr_decomp['error']:.6f}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the low-rank approximation."""
        return self.linear_r(self.linear_l(x))
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for comparison."""
        pela_params = sum(p.numel() for p in self.parameters())
        original_params = self.in_features * self.out_features
        if hasattr(self.linear_r, 'bias') and self.linear_r.bias is not None:
            original_params += self.out_features
        
        return {
            'original': original_params,
            'pela': pela_params,
            'reduction': original_params - pela_params,
            'compression_ratio': original_params / pela_params if pela_params > 0 else 0
        }


class ModulePELA:
    """
    PELA module replacement manager for OLMoOCR models.
    
    This class handles the replacement of linear layers in OLMoOCR models
    with low-rank approximations while maintaining the model's functionality.
    """
    
    def __init__(self,
                 compress_ratio: float = 3.0,
                 target_modules: Optional[List[str]] = None,
                 exclude_modules: Optional[List[str]] = None,
                 min_rank: int = 8,
                 max_rank: int = 256,
                 is_approximate: bool = True):
        """
        Initialize PELA module manager.
        
        Args:
            compress_ratio: Target compression ratio for rank calculation
            target_modules: Specific modules to target (if None, use defaults)
            exclude_modules: Modules to exclude from compression
            min_rank: Minimum rank for any layer
            max_rank: Maximum rank for any layer
            is_approximate: Whether to initialize with low-rank approximation
        """
        self.compress_ratio = compress_ratio
        self.is_approximate = is_approximate
        self.min_rank = min_rank
        self.max_rank = max_rank
        
        # Default target modules for OLMoOCR (based on LoRA config)
        self.target_modules = target_modules or [
            # Main transformer attention and FF layers (Molmo)
            'att_proj', 'ff_proj', 'attn_out', 'ff_out',
            # Vision transformer layers (Molmo)
            'attention.wq', 'attention.wk', 'attention.wv', 'attention.wo',
            'feed_forward.w1', 'feed_forward.w2',
            # Vision projector (Molmo)
            'vision_backbone.image_projector',
            # Qwen2VL attention and FF layers
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj',
            # Qwen2VL vision layers
            'visual.blocks.*.attn.qkv', 'visual.blocks.*.attn.proj',
            'visual.blocks.*.mlp.fc1', 'visual.blocks.*.mlp.fc2',
            'visual.merger.mlp.0', 'visual.merger.mlp.2'
        ]
        
        self.exclude_modules = exclude_modules or [
            'embed', 'embedding', 'norm', 'ln', 'head', 'classifier'
        ]
        
        self.replacement_stats = {}
    
    def _calculate_rank(self, in_features: int, out_features: int) -> int:
        """Calculate optimal rank based on compression ratio."""
        # Formula: rank = (in_features * out_features) / (compress_ratio * (in_features + out_features))
        rank = int((in_features * out_features) / (self.compress_ratio * (in_features + out_features)))
        return max(self.min_rank, min(self.max_rank, rank))
    
    def _should_replace_module(self, name: str, module: nn.Module) -> bool:
        """Determine if a module should be replaced with PELA."""
        if not isinstance(module, nn.Linear):
            return False
        
        # Check if module is too small
        if module.out_features < 10:
            return False
        
        # Check exclusion list
        if any(exclude in name.lower() for exclude in self.exclude_modules):
            return False
        
        # Check target list
        return any(target in name for target in self.target_modules)
    
    def apply_pela(self, model: PreTrainedModel) -> PreTrainedModel:
        """
        Apply PELA compression to the model.
        
        Args:
            model: The model to compress
            
        Returns:
            Compressed model with PELA layers
        """
        # Create a copy of the model to avoid modifying the original
        compressed_model = copy.deepcopy(model)
        
        # Collect all linear layers that should be replaced
        layers_to_replace = []
        for name, module in compressed_model.named_modules():
            if self._should_replace_module(name, module):
                layers_to_replace.append((name, module))
        
        if not layers_to_replace:
            logger.warning("No layers found to replace with PELA!")
            return compressed_model
        
        print(f"🎯 Found {len(layers_to_replace)} layers to compress")
        print(f"📊 Estimated time: {len(layers_to_replace) * 2:.0f}-{len(layers_to_replace) * 8:.0f} minutes")
        
        # Apply PELA to each layer with progress bar
        start_time = time.time()
        with tqdm(layers_to_replace, desc="🗜️  Compressing layers", unit="layer") as pbar:
            for i, (name, original_module) in enumerate(pbar):
                layer_start = time.time()
                
                # Calculate rank for this layer
                rank = self._calculate_rank(original_module.in_features, original_module.out_features)
                
                # Create PELA replacement
                pela_module = PELALinear(
                    original_module, 
                    rank, 
                    self.compress_ratio, 
                    self.is_approximate
                )
                
                # Replace the module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent_module = compressed_model.get_submodule(parent_name)
                    setattr(parent_module, child_name, pela_module)
                else:
                    setattr(compressed_model, child_name, pela_module)
                
                # Track statistics
                stats = pela_module.get_parameter_count()
                self.replacement_stats[name] = stats
                
                # Update progress bar with timing info
                layer_time = time.time() - layer_start
                elapsed = time.time() - start_time
                avg_time_per_layer = elapsed / (i + 1)
                eta = avg_time_per_layer * (len(layers_to_replace) - i - 1)
                
                pbar.set_postfix({
                    'Layer': f"{original_module.in_features}x{original_module.out_features}",
                    'Rank': rank,
                    'Time': f"{layer_time:.1f}s",
                    'ETA': f"{eta/60:.1f}m"
                })
        
        total_time = time.time() - start_time
        print(f"✅ Compression completed in {total_time/60:.1f} minutes")
        
        return compressed_model
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics about PELA application."""
        if not self.replacement_stats:
            return {}
        
        total_original = sum(stats['original'] for stats in self.replacement_stats.values())
        total_pela = sum(stats['pela'] for stats in self.replacement_stats.values())
        
        return {
            'total_replacements': len(self.replacement_stats),
            'total_original_params': total_original,
            'total_pela_params': total_pela,
            'total_reduction': total_original - total_pela,
            'overall_compression_ratio': total_original / total_pela if total_pela > 0 else 0,
            'per_module_stats': self.replacement_stats
        }


# Convenience function for easy model compression
def compress_olmocr_with_pela(model: PreTrainedModel, 
                             compress_ratio: float = 3.0,
                             **kwargs) -> PreTrainedModel:
    """
    Convenient function to apply PELA compression to an OLMoOCR model.
    
    Args:
        model: OLMoOCR model to compress
        compress_ratio: Target compression ratio
        **kwargs: Additional arguments for ModulePELA
        
    Returns:
        Compressed model with PELA applied
    """
    pela_manager = ModulePELA(compress_ratio=compress_ratio, **kwargs)
    compressed_model = pela_manager.apply_pela(model)
    
    # Print statistics
    stats = pela_manager.get_statistics()
    if stats:
        print(f"\n🎯 PELA Compression Results:")
        print(f"   📊 Modules compressed: {stats['total_replacements']}")
        print(f"   📈 Parameter reduction: {stats['total_reduction']:,} "
              f"({100 * stats['total_reduction'] / stats['total_original_params']:.1f}%)")
        print(f"   🗜️  Compression ratio: {stats['overall_compression_ratio']:.2f}x")
    
    return compressed_model 


def compress_model_with_pela(model, compress_ratio: float = 3.0, 
                           is_approximate: bool = True, 
                           batch_size_strategy: str = "smart",
                           checkpoint_interval: int = 10,
                           checkpoint_dir: str = "./checkpoints",
                           evaluate_quality: bool = False,
                           evaluation_sample_rate: float = 0.1) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Compress model with PELA, enhanced with batching, checkpointing, and optional evaluation.
    
    Args:
        model: Model to compress
        compress_ratio: Target compression ratio  
        is_approximate: Whether to use approximate SVD
        batch_size_strategy: Batching strategy ("smart", "fixed", "none")
        checkpoint_interval: Save checkpoint every N layers
        checkpoint_dir: Directory to save checkpoints
        evaluate_quality: Whether to evaluate compression quality
        evaluation_sample_rate: Fraction of layers to evaluate in detail
        
    Returns:
        Tuple of (compression_stats, evaluation_results)
    """
    from layer_comparison import LayerComparator  # Import here to avoid circular import
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Initialize evaluation if requested
    evaluator = LayerComparator() if evaluate_quality else None
    evaluation_results = [] if evaluate_quality else None
    
    # Find compressible layers
    compressible_layers = find_compressible_layers(model)
    total_layers = len(compressible_layers)
    
    print(f"📊 Found {total_layers} compressible layers")
    print(f"📊 Initial memory usage: {get_memory_usage():.2f} GB")
    
    if evaluate_quality:
        print(f"🔍 Quality evaluation enabled (sampling {evaluation_sample_rate*100:.1f}% of layers)")
    
    # Create batches
    if batch_size_strategy == "smart":
        batches = create_smart_batches(compressible_layers)
    elif batch_size_strategy == "fixed":
        batch_size = 5
        batches = [compressible_layers[i:i+batch_size] for i in range(0, len(compressible_layers), batch_size)]
    else:  # "none"
        batches = [[layer] for layer in compressible_layers]
    
    print(f"📦 Created {len(batches)} batches using {batch_size_strategy} strategy")
    
    # Initialize tracking
    compression_stats = {
        'total_layers': total_layers,
        'total_original_params': 0,
        'total_compressed_params': 0,
        'layer_details': [],
        'memory_usage': [],
        'batch_info': []
    }
    
    # Create checkpoint directory
    Path(checkpoint_dir).mkdir(exist_ok=True)
    
    # Two-level progress bars
    batch_pbar = tqdm(total=len(batches), desc="📦 Processing batches", position=0)
    layer_pbar = tqdm(total=total_layers, desc="🔧 Compressing layers", position=1)
    
    try:
        layers_processed = 0
        
        for batch_idx, batch in enumerate(batches):
            batch_stats = process_batch(
                batch, model, compress_ratio, is_approximate, device,
                layer_pbar, evaluator, evaluation_sample_rate, evaluation_results
            )
            
            compression_stats['batch_info'].append(batch_stats)
            compression_stats['total_original_params'] += batch_stats['batch_original_params']
            compression_stats['total_compressed_params'] += batch_stats['batch_compressed_params']
            compression_stats['layer_details'].extend(batch_stats['layer_details'])
            
            layers_processed += len(batch)
            
            # Memory tracking
            current_memory = get_memory_usage()
            compression_stats['memory_usage'].append({
                'batch': batch_idx,
                'layers_processed': layers_processed,
                'memory_gb': current_memory
            })
            
            # Checkpointing
            if (batch_idx + 1) % checkpoint_interval == 0:
                checkpoint_path = f"{checkpoint_dir}/checkpoint_batch_{batch_idx + 1}.pt"
                save_checkpoint(model, compression_stats, evaluation_results, checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Aggressive memory cleanup
            cleanup_memory()
            
            batch_pbar.update(1)
            batch_pbar.set_postfix({
                'Memory': f"{current_memory:.1f}GB",
                'Layers': f"{layers_processed}/{total_layers}"
            })
    
    finally:
        batch_pbar.close()
        layer_pbar.close()
    
    # Final statistics
    if compression_stats['total_original_params'] > 0:
        overall_ratio = compression_stats['total_original_params'] / compression_stats['total_compressed_params']
        compression_stats['overall_compression_ratio'] = overall_ratio
        reduction_percent = ((compression_stats['total_original_params'] - compression_stats['total_compressed_params']) 
                           / compression_stats['total_original_params']) * 100
        compression_stats['parameter_reduction_percent'] = reduction_percent
        
        print(f"\n✅ COMPRESSION COMPLETE!")
        print(f"📊 Overall compression ratio: {overall_ratio:.2f}x")
        print(f"📊 Parameter reduction: {reduction_percent:.1f}%")
        print(f"📊 Final memory usage: {get_memory_usage():.2f} GB")
    
    # Compile evaluation results
    eval_summary = None
    if evaluate_quality and evaluation_results:
        eval_summary = compile_evaluation_summary(evaluation_results)
        print_evaluation_summary(eval_summary)
    
    return compression_stats, eval_summary


def process_batch(batch: List[Tuple[str, nn.Module]], model: nn.Module, 
                 compress_ratio: float, is_approximate: bool, device: torch.device,
                 layer_pbar: tqdm, evaluator: Optional[Any], 
                 evaluation_sample_rate: float, evaluation_results: Optional[List]) -> Dict[str, Any]:
    """Process a batch of layers."""
    batch_stats = {
        'batch_size': len(batch),
        'batch_original_params': 0,
        'batch_compressed_params': 0,
        'layer_details': []
    }
    
    for layer_name, layer in batch:
        try:
            # Store original layer for evaluation
            original_layer = None
            if evaluator and torch.rand(1).item() < evaluation_sample_rate:
                # Deep copy only for evaluation
                original_layer = type(layer)(layer.in_features, layer.out_features, 
                                           bias=layer.bias is not None).to(layer.weight.device)
                original_layer.weight.data.copy_(layer.weight.data)
                if layer.bias is not None:
                    original_layer.bias.data.copy_(layer.bias.data)
            
            # Compress layer
            original_params = layer.weight.numel() + (layer.bias.numel() if layer.bias is not None else 0)
            compressed_layer, compress_info = compress_layer_optimized(layer, compress_ratio, is_approximate, device)
            
            # Replace layer in model
            set_nested_attr(model, layer_name, compressed_layer)
            
            # Calculate compressed parameters
            compressed_params = (compressed_layer.linear_l.weight.numel() + 
                               compressed_layer.linear_r.weight.numel() +
                               (compressed_layer.linear_l.bias.numel() if compressed_layer.linear_l.bias is not None else 0) +
                               (compressed_layer.linear_r.bias.numel() if compressed_layer.linear_r.bias is not None else 0))
            
            # Update statistics
            batch_stats['batch_original_params'] += original_params
            batch_stats['batch_compressed_params'] += compressed_params
            
            layer_detail = {
                'name': layer_name,
                'original_params': original_params,
                'compressed_params': compressed_params,
                'layer_compression_ratio': original_params / compressed_params,
                'mse': compress_info.get('mse', None),
                'rank': compress_info.get('rank', None)
            }
            batch_stats['layer_details'].append(layer_detail)
            
            # Evaluate quality if requested
            if evaluator and original_layer is not None:
                try:
                    eval_report = evaluator.comprehensive_comparison(
                        original_layer, compressed_layer, layer_name
                    )
                    evaluation_results.append(eval_report)
                except Exception as e:
                    print(f"⚠️  Evaluation failed for {layer_name}: {e}")
                finally:
                    # Cleanup original layer
                    del original_layer
            
            layer_pbar.update(1)
            layer_pbar.set_postfix({
                'Current': layer_name.split('.')[-1],
                'Ratio': f"{layer_detail['layer_compression_ratio']:.1f}x"
            })
            
        except Exception as e:
            print(f"❌ Failed to compress {layer_name}: {e}")
            layer_pbar.update(1)
        
        # Cleanup after each layer
        cleanup_memory()
    
    return batch_stats


def compile_evaluation_summary(evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compile evaluation results into a summary."""
    import numpy as np
    
    summary = {
        'num_evaluated_layers': len(evaluation_results),
        'weight_metrics': {},
        'forward_metrics': {},
        'singular_value_metrics': {}
    }
    
    # Extract metrics
    weight_metrics = [r['weight_metrics'] for r in evaluation_results 
                     if 'weight_metrics' in r and 'error' not in r['weight_metrics']]
    forward_metrics = [r['forward_metrics'] for r in evaluation_results 
                      if 'forward_metrics' in r and 'error' not in r['forward_metrics']]
    sv_metrics = [r['singular_value_metrics'] for r in evaluation_results 
                 if 'singular_value_metrics' in r and 'error' not in r['singular_value_metrics']]
    
    # Weight metrics summary
    if weight_metrics:
        summary['weight_metrics'] = {
            'mean_cosine_similarity': np.mean([m['cosine_similarity'] for m in weight_metrics]),
            'std_cosine_similarity': np.std([m['cosine_similarity'] for m in weight_metrics]),
            'mean_frobenius_error': np.mean([m['relative_frobenius_error'] for m in weight_metrics]),
            'std_frobenius_error': np.std([m['relative_frobenius_error'] for m in weight_metrics])
        }
    
    # Forward pass metrics summary
    if forward_metrics:
        summary['forward_metrics'] = {
            'mean_output_similarity': np.mean([m['forward_cosine_mean'] for m in forward_metrics]),
            'std_output_similarity': np.std([m['forward_cosine_mean'] for m in forward_metrics]),
            'mean_mse': np.mean([m['forward_mse_mean'] for m in forward_metrics]),
            'std_mse': np.std([m['forward_mse_mean'] for m in forward_metrics])
        }
    
    # Singular value metrics summary
    if sv_metrics:
        summary['singular_value_metrics'] = {
            'mean_energy_preservation': np.mean([m['energy_preservation'] for m in sv_metrics]),
            'std_energy_preservation': np.std([m['energy_preservation'] for m in sv_metrics]),
            'mean_rank_ratio': np.mean([m['rank_ratio'] for m in sv_metrics]),
            'std_rank_ratio': np.std([m['rank_ratio'] for m in sv_metrics])
        }
    
    return summary


def print_evaluation_summary(eval_summary: Dict[str, Any]):
    """Print a summary of evaluation results."""
    print(f"\n🔍 COMPRESSION QUALITY EVALUATION:")
    print(f"📊 Evaluated {eval_summary['num_evaluated_layers']} layers")
    
    if eval_summary['weight_metrics']:
        wm = eval_summary['weight_metrics']
        print(f"📊 Weight Cosine Similarity: {wm['mean_cosine_similarity']:.4f} ± {wm['std_cosine_similarity']:.4f}")
        print(f"📊 Frobenius Error: {wm['mean_frobenius_error']:.4f} ± {wm['std_frobenius_error']:.4f}")
    
    if eval_summary['forward_metrics']:
        fm = eval_summary['forward_metrics']
        print(f"📊 Output Similarity: {fm['mean_output_similarity']:.4f} ± {fm['std_output_similarity']:.4f}")
    
    if eval_summary['singular_value_metrics']:
        svm = eval_summary['singular_value_metrics']
        print(f"📊 Energy Preservation: {svm['mean_energy_preservation']:.4f} ± {svm['std_energy_preservation']:.4f}")


def save_checkpoint(model: nn.Module, compression_stats: Dict[str, Any], 
                   evaluation_results: Optional[List], checkpoint_path: str):
    """Save model and progress checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'compression_stats': compression_stats,
        'evaluation_results': evaluation_results
    }
    torch.save(checkpoint, checkpoint_path)


def main():
    """Main function to test PELA compression on OLMoOCR."""
    # Enhanced argument parsing would go here
    model_path = "allenai/olmOCR-7B-0225-preview"
    
    print("🚀 Starting PELA compression of OLMoOCR model...")
    print("🔍 Quality evaluation enabled")
    
    try:
        # Load model
        model = load_model_safe(model_path)
        
        # Compress with evaluation
        compression_stats, eval_summary = compress_model_with_pela(
            model, 
            compress_ratio=3.0,
            is_approximate=True,
            evaluate_quality=True,
            evaluation_sample_rate=0.2  # Evaluate 20% of layers
        )
        
        # Save compressed model
        output_path = "olmocr_compressed_pela.pt"
        torch.save(model, output_path)
        print(f"💾 Compressed model saved to {output_path}")
        
        # Save detailed results
        results = {
            'compression_stats': compression_stats,
            'evaluation_summary': eval_summary
        }
        
        with open("compression_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("📊 Detailed results saved to compression_results.json")
        
    except Exception as e:
        print(f"❌ Compression failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 