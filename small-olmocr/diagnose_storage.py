#!/usr/bin/env python3
"""
Diagnose storage efficiency issues in PELA compressed models
"""

import torch
import sys
from collections import defaultdict

def analyze_model_storage(model_path):
    """Analyze how efficiently a model is stored"""
    print(f"\n🔍 ANALYZING: {model_path}")
    print("=" * 60)
    
    try:
        # Load model without weights_only for meta analysis
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        # Handle different storage formats
        if hasattr(checkpoint, 'state_dict'):
            # Full model object
            print(f"📋 Storage format: Full model object ({type(checkpoint).__name__})")
            model_state = checkpoint.state_dict()
        elif isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Checkpoint with model key
            print(f"📋 Storage format: Checkpoint dict with model key")
            if hasattr(checkpoint['model'], 'state_dict'):
                model_state = checkpoint['model'].state_dict()
            else:
                model_state = checkpoint['model']
        elif isinstance(checkpoint, dict):
            # Direct state dict
            print(f"📋 Storage format: Direct state dict")
            model_state = checkpoint
        else:
            print(f"📋 Storage format: Unknown ({type(checkpoint).__name__})")
            return
            
        # Analyze data types and sizes
        dtype_counts = defaultdict(int)
        dtype_sizes = defaultdict(int)
        total_params = 0
        total_size_bytes = 0
        
        print(f"📋 Total layers/tensors: {len(model_state)}")
        
        for name, tensor in model_state.items():
            if isinstance(tensor, torch.Tensor):
                dtype_counts[str(tensor.dtype)] += 1
                param_count = tensor.numel()
                size_bytes = tensor.numel() * tensor.element_size()
                
                dtype_sizes[str(tensor.dtype)] += size_bytes
                total_params += param_count
                total_size_bytes += size_bytes
                
                # Show largest tensors
                if param_count > 100_000_000:  # > 100M params
                    print(f"   🔍 Large tensor: {name}")
                    print(f"      Shape: {list(tensor.shape)}, Dtype: {tensor.dtype}")
                    print(f"      Params: {param_count:,}, Size: {size_bytes / (1024**3):.2f} GB")
        
        print(f"\n📊 STORAGE ANALYSIS:")
        print(f"   Total parameters: {total_params:,}")
        print(f"   Total size: {total_size_bytes / (1024**3):.2f} GB")
        print(f"   Avg bytes per param: {total_size_bytes / total_params:.2f}")
        
        print(f"\n📋 DATA TYPE BREAKDOWN:")
        for dtype, count in sorted(dtype_counts.items()):
            size_gb = dtype_sizes[dtype] / (1024**3)
            print(f"   {dtype}: {count} tensors, {size_gb:.2f} GB")
            
        # Expected sizes for different precisions
        print(f"\n💡 THEORETICAL SIZES:")
        print(f"   If all float32: {total_params * 4 / (1024**3):.2f} GB")
        print(f"   If all float16: {total_params * 2 / (1024**3):.2f} GB")
        print(f"   If all bfloat16: {total_params * 2 / (1024**3):.2f} GB")
        
        # Check for storage inefficiencies
        file_size_gb = 30.23 if "1.5x" in model_path else 28.98  # From our earlier analysis
        theoretical_size = total_size_bytes / (1024**3)
        overhead = file_size_gb - theoretical_size
        
        print(f"\n🔍 STORAGE EFFICIENCY:")
        print(f"   File size on disk: {file_size_gb:.2f} GB")
        print(f"   Theoretical tensor size: {theoretical_size:.2f} GB")
        print(f"   Storage overhead: {overhead:.2f} GB")
        
        if overhead > 1:
            print(f"\n🚨 MAJOR STORAGE INEFFICIENCY!")
            print(f"   💡 Likely cause: Saving full model object instead of state_dict")
            print(f"   🎯 Fix: Save only model.state_dict() to reduce overhead")
        
        # Efficiency analysis
        if total_size_bytes / total_params > 3:
            print(f"\n🚨 DATA TYPE INEFFICIENCY!")
            print(f"   Your model uses {total_size_bytes / total_params:.1f} bytes per parameter")
            print(f"   Expected for float16: 2 bytes per parameter")
            print(f"   🎯 Potential savings: {(total_size_bytes - total_params * 2) / (1024**3):.2f} GB")
            
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        import traceback
        traceback.print_exc()

def suggest_fixes():
    """Suggest specific fixes for the storage issues"""
    print(f"\n🔧 SPECIFIC FIXES FOR YOUR ISSUE")
    print("=" * 60)
    
    print("🎯 PRIMARY ISSUE: Saving full model object (with overhead)")
    print("   FIX: Save only the state_dict")
    print()
    print("   # Instead of:")
    print("   torch.save(model, 'model.pt')  # Saves full object")
    print()
    print("   # Do this:")
    print("   torch.save(model.state_dict(), 'model.pt')  # Saves only weights")
    
    print(f"\n🎯 SECONDARY ISSUE: Using float32 precision")
    print("   FIX: Convert to float16 before saving")
    print()
    print("   # Optimize precision:")
    print("   model = model.half()  # Convert to float16")
    print("   torch.save(model.state_dict(), 'model_fp16.pt')")
    
    print(f"\n🎯 COMBINED OPTIMIZATION:")
    print("   # This should reduce your 30GB model to ~8GB:")
    print("   model = model.half()")
    print("   torch.save(model.state_dict(), 'optimized_model.pt')")

if __name__ == "__main__":
    models_to_check = [
        "./models/olmocr_compressed_1.5x.pt",
        "./models/olmocr_compressed_1.8x.pt"
    ]
    
    for model_path in models_to_check:
        try:
            analyze_model_storage(model_path)
        except Exception as e:
            print(f"❌ Failed to analyze {model_path}: {e}")
    
    suggest_fixes() 