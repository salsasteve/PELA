#!/usr/bin/env python3
"""
Optimize storage of PELA compressed models
This will show the TRUE compression results by fixing storage inefficiencies
"""

import torch
import os
from pathlib import Path

def get_file_size_gb(filepath):
    """Get file size in GB"""
    if os.path.exists(filepath):
        return os.path.getsize(filepath) / (1024**3)
    return 0

def optimize_model(input_path, output_path):
    """Optimize a model's storage by converting to fp16 and saving state_dict only"""
    print(f"\n🔧 OPTIMIZING: {input_path}")
    print("=" * 60)
    
    # Get original size
    original_size = get_file_size_gb(input_path)
    print(f"📁 Original size: {original_size:.2f} GB")
    
    try:
        # Load the full model
        print("📥 Loading model...")
        model = torch.load(input_path, map_location='cpu', weights_only=False)
        
        # Convert to float16 for efficiency
        print("🔄 Converting to float16...")
        model = model.half()
        
        # Save only the state_dict
        print("💾 Saving optimized state_dict...")
        torch.save(model.state_dict(), output_path)
        
        # Check new size
        new_size = get_file_size_gb(output_path)
        reduction = (original_size - new_size) / original_size * 100
        
        print(f"✅ Optimization complete!")
        print(f"📁 New size: {new_size:.2f} GB")
        print(f"📊 Size reduction: {reduction:.1f}%")
        print(f"💾 Space saved: {original_size - new_size:.2f} GB")
        
        return new_size
        
    except Exception as e:
        print(f"❌ Error optimizing {input_path}: {e}")
        return None

def demonstrate_true_compression():
    """Show the true compression results after fixing storage issues"""
    print("🎯 PELA COMPRESSION - TRUE RESULTS")
    print("=" * 60)
    
    # Optimize the 1.5x compressed model
    input_model = "./models/olmocr_compressed_1.5x.pt"
    optimized_model = "./models/olmocr_compressed_1.5x_optimized.pt"
    
    if os.path.exists(input_model):
        optimized_size = optimize_model(input_model, optimized_model)
        
        # Estimate original model size in fp16 for fair comparison
        # From debug: original has 8.29B params, in fp16 = ~16.6GB
        original_params = 8.29e9
        estimated_original_fp16 = original_params * 2 / (1024**3)  # 2 bytes per param
        
        if optimized_size:
            print(f"\n🏆 FINAL COMPRESSION RESULTS:")
            print(f"   Original (estimated fp16): {estimated_original_fp16:.1f} GB")
            print(f"   PELA Compressed (fp16): {optimized_size:.1f} GB") 
            print(f"   Compression ratio: {estimated_original_fp16 / optimized_size:.1f}x")
            print(f"   Space saved: {estimated_original_fp16 - optimized_size:.1f} GB")
            
            # Parameter comparison
            compressed_params = 8.11e9  # From debug output
            param_reduction = (original_params - compressed_params) / original_params * 100
            print(f"\n📊 Parameter reduction: {param_reduction:.1f}%")
            print(f"   Original: {original_params/1e9:.2f}B parameters")
            print(f"   Compressed: {compressed_params/1e9:.2f}B parameters")
    else:
        print(f"❌ Model not found: {input_model}")

def create_loading_example():
    """Create example code for loading the optimized model"""
    print(f"\n📋 HOW TO LOAD THE OPTIMIZED MODEL:")
    print("=" * 60)
    print("```python")
    print("import torch")
    print("from transformers import Qwen2VLForConditionalGeneration")
    print("")
    print("# Load the model architecture")
    print('model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/OLMoOCR-7B-0225-preview")')
    print("")
    print("# Load the optimized PELA weights")
    print('state_dict = torch.load("olmocr_compressed_1.5x_optimized.pt", map_location="cpu")')
    print("model.load_state_dict(state_dict)")
    print("")
    print("# Model is now ready to use with PELA compression!")
    print("```")

if __name__ == "__main__":
    demonstrate_true_compression()
    create_loading_example() 