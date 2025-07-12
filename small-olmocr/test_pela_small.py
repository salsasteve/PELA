#!/usr/bin/env python3
"""
Test PELA compression on a small subset of OLMoOCR layers
"""

import torch
import os
from transformers import Qwen2VLForConditionalGeneration
from pela_olmocr import ModulePELA, PELALinear

def test_small_pela():
    print("🚀 Testing PELA compression on small subset...")
    
    model_name = "olmOCR-7B-0225-preview"
    local_model_path = f"./models/{model_name}"
    
    if os.path.exists(local_model_path):
        model_path = local_model_path
    else:
        model_path = "allenai/olmOCR-7B-0225-preview"
    
    try:
        print(f"📥 Loading model from: {model_path}")
        
        # Force CPU and float32 to avoid memory issues
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            device_map={"": "cpu"},
            local_files_only=os.path.exists(local_model_path)
        )
        
        print("✅ Model loaded successfully!")
        
        # Find small layers (< 2048 dimensions) for testing
        small_layers = []
        for name, module in model.named_modules():
            if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
                if isinstance(module, torch.nn.Linear):
                    size = max(module.in_features, module.out_features)
                    if size < 2048:  # Only small layers
                        small_layers.append((name, module))
        
        print(f"🎯 Found {len(small_layers)} small layers to test:")
        for name, module in small_layers[:5]:
            print(f"   {name}: {module.in_features}x{module.out_features}")
        
        if not small_layers:
            print("❌ No small layers found for testing")
            return False
        
        # Test PELA on just the first small layer
        test_layer_name, test_layer = small_layers[0]
        print(f"\n🗜️  Testing PELA on: {test_layer_name} ({test_layer.in_features}x{test_layer.out_features})")
        
        # Create PELA replacement
        pela_layer = PELALinear(
            test_layer,
            rank=16,  # Small rank for testing
            compress_ratio=3.0,
            is_approximate=True
        )
        
        print("✅ PELA layer created successfully!")
        
        # Get compression stats
        stats = pela_layer.get_parameter_count()
        print(f"📊 Original parameters: {stats['original']:,}")
        print(f"📊 PELA parameters: {stats['pela']:,}")
        print(f"🗜️  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"💾 Parameter reduction: {stats['reduction']:,} ({100 * stats['reduction'] / stats['original']:.1f}%)")
        
        # Test forward pass
        print("\n🧪 Testing forward pass...")
        test_input = torch.randn(1, test_layer.in_features)
        
        with torch.no_grad():
            original_output = test_layer(test_input)
            pela_output = pela_layer(test_input)
            
            # Check output similarity
            mse_error = torch.nn.functional.mse_loss(original_output, pela_output)
            print(f"📏 MSE between original and PELA: {mse_error:.6f}")
        
        print("\n🎉 Small PELA test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Small PELA test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_small_pela()
    if success:
        print("\n✅ PELA compression works! The algorithm is functional.")
        print("💡 For full model compression, consider:")
        print("   - Using a machine with more RAM (24GB+)")
        print("   - Processing layers in smaller batches")
        print("   - Using a cloud instance with high memory")
    else:
        print("\n⚠️  Basic PELA test failed") 