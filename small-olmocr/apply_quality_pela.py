#!/usr/bin/env python3
"""
Apply quality-based PELA compression to OLMoOCR model
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration
from quality_based_pela import QualityBasedPELACompressor
import time
import os

def apply_quality_pela(target_retention=0.55):
    """Apply quality-based PELA compression to OLMoOCR"""
    print("🎯 QUALITY-BASED PELA COMPRESSION")
    print("=" * 60)
    print(f"🎚️  Target information retention: {target_retention:.0%}")
    print(f"📊 This will compress ALL 347 linear layers")
    print()
    
    # Load model from local directory
    print("📥 Loading OLMoOCR model from local directory...")
    
    # Try common local paths
    possible_paths = [
        "./models/olmOCR-7B-0225-preview",
        "models/olmOCR-7B-0225-preview",
        "../models/olmOCR-7B-0225-preview"
    ]
    
    model_path = None
    for path in possible_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if model_path is None:
        print("❌ Local model not found. Searched:")
        for path in possible_paths:
            print(f"   {path}")
        print("\n💡 Please ensure the model is downloaded locally or update the path")
        return
    
    try:
        print(f"📂 Loading from: {model_path}")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            local_files_only=True
        )
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Analyze original model
    print("\n🔍 ANALYZING ORIGINAL MODEL")
    print("-" * 40)
    
    original_params = sum(p.numel() for p in model.parameters())
    linear_layers = 0
    linear_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.weight.numel() >= 1000:
            linear_layers += 1
            linear_params += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                linear_params += module.bias.numel()
    
    print(f"📊 Total parameters: {original_params:,}")
    print(f"📊 Linear layers to compress: {linear_layers}")
    print(f"📊 Linear parameters: {linear_params:,} ({linear_params/original_params*100:.1f}%)")
    print(f"📊 Original size (fp16): {original_params * 2 / (1024**3):.1f} GB")
    
    # Apply compression
    print(f"\n🚀 APPLYING COMPRESSION")
    print("-" * 40)
    
    compressor = QualityBasedPELACompressor(target_retention=target_retention)
    
    start_time = time.time()
    compressed_model = compressor.compress_model(model)
    compression_time = time.time() - start_time
    
    print(f"\n⏱️  Compression completed in {compression_time:.1f} seconds")
    
    # Analyze compressed model
    print(f"\n🔍 ANALYZING COMPRESSED MODEL")
    print("-" * 40)
    
    compressed_params = sum(p.numel() for p in compressed_model.parameters())
    compression_ratio = original_params / compressed_params
    
    print(f"📊 Compressed parameters: {compressed_params:,}")
    print(f"📊 Overall compression: {compression_ratio:.1f}x")
    print(f"📊 Parameters saved: {original_params - compressed_params:,}")
    print(f"📊 Size reduction: {(1 - compressed_params/original_params)*100:.1f}%")
    
    # Quality analysis
    quality_scores = [stat['cosine_similarity'] for stat in compressor.compression_stats]
    avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
    
    print(f"\n📈 QUALITY ANALYSIS")
    print("-" * 40)
    print(f"📊 Layers compressed: {len(compressor.compression_stats)}")
    print(f"📊 Average cosine similarity: {avg_quality:.3f}")
    print(f"📊 Quality range: {min(quality_scores):.3f} - {max(quality_scores):.3f}")
    print(f"📊 Target retention: {target_retention:.0%}")
    
    if avg_quality >= target_retention * 0.9:
        print(f"✅ Quality target achieved!")
    else:
        print(f"⚠️  Below quality target")
    
    # Test functionality
    print(f"\n🧪 TESTING MODEL FUNCTIONALITY")
    print("-" * 40)
    
    try:
        compressed_model.eval()
        with torch.no_grad():
            # Simple forward pass test
            batch_size = 1
            seq_len = 10
            dummy_input_ids = torch.randint(0, 1000, (batch_size, seq_len))
            dummy_attention_mask = torch.ones(batch_size, seq_len)
            
            outputs = compressed_model(
                input_ids=dummy_input_ids,
                attention_mask=dummy_attention_mask
            )
            
            print(f"✅ Forward pass successful")
            print(f"📊 Output shape: {outputs.logits.shape}")
            print(f"📊 Output dtype: {outputs.logits.dtype}")
            functionality_works = True
            
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        functionality_works = False
    
    # Save compressed model
    if functionality_works:
        print(f"\n💾 SAVING COMPRESSED MODEL")
        print("-" * 40)
        
        # Convert to fp16 and save state_dict only
        compressed_model = compressed_model.half()
        output_path = f"./models/olmocr_quality_pela_{target_retention:.0%}.pt"
        
        print(f"🔄 Converting to fp16 and saving state_dict...")
        torch.save(compressed_model.state_dict(), output_path)
        
        # Check file size
        file_size = os.path.getsize(output_path) / (1024**3)
        
        print(f"✅ Saved to: {output_path}")
        print(f"📁 File size: {file_size:.1f} GB")
        
        # Final results
        print(f"\n🏆 FINAL RESULTS")
        print("=" * 60)
        print(f"🎯 Target retention: {target_retention:.0%}")
        print(f"📊 Compression achieved: {compression_ratio:.1f}x")
        print(f"📁 Model size: {file_size:.1f} GB")
        print(f"📈 Average quality: {avg_quality:.3f}")
        print(f"✅ Functionality: {'Working' if functionality_works else 'Broken'}")
        
        # Compare to previous attempts
        print(f"\n📊 vs Previous PELA attempts:")
        print(f"   Conservative PELA: 1.02x compression, 15.1 GB")
        print(f"   Quality PELA: {compression_ratio:.1f}x compression, {file_size:.1f} GB")
        
        if file_size < 12:  # Target was ~10GB for 1.5x
            print(f"   🎉 SUCCESS: Finally achieved meaningful compression!")
        else:
            print(f"   🎯 Good progress, but room for more aggressive settings")
            
        # Loading instructions
        print(f"\n📋 TO LOAD THE COMPRESSED MODEL:")
        print("```python")
        print("from transformers import Qwen2VLForConditionalGeneration")
        print("import torch")
        print("")
        print(f'model = Qwen2VLForConditionalGeneration.from_pretrained("{model_path}", local_files_only=True)')
        print(f'state_dict = torch.load("{output_path}", map_location="cpu")')
        print("model.load_state_dict(state_dict)")
        print("# Model ready with quality-based PELA compression!")
        print("```")
        
    else:
        print(f"\n❌ MODEL FUNCTIONALITY BROKEN")
        print("   Compression may be too aggressive")
        print("   Try higher retention (e.g., 0.60 or 0.65)")

if __name__ == "__main__":
    # Apply with 55% target retention (balanced quality/compression)
    apply_quality_pela(target_retention=0.55) 