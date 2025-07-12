#!/usr/bin/env python3
"""
Test PELA compression on real OLMoOCR model
"""

import torch
import os
import psutil
import gc
from tqdm import tqdm
from transformers import AutoModelForCausalLM, Qwen2VLForConditionalGeneration
from pela_olmocr import compress_olmocr_with_pela
from layer_comparison import LayerComparator

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / 1024**3  # Convert to GB

def test_pela_compression():
    print("🚀 Testing PELA compression on real OLMoOCR model...")
    
    # Check if model exists locally first
    model_name = "olmOCR-7B-0225-preview"
    local_model_path = f"./models/{model_name}"
    
    if os.path.exists(local_model_path):
        print(f"📁 Loading model from local path: {local_model_path}")
        model_path = local_model_path
    else:
        print(f"🌐 Local model not found at {local_model_path}")
        print("💡 Please download the model manually using:")
        print("   git lfs install")
        print(f"   git clone https://huggingface.co/allenai/{model_name} ./models/{model_name}")
        print("\nAlternatively, download directly from HuggingFace Hub")
        model_path = "allenai/olmOCR-7B-0225-preview"  # Fallback to online
    
    try:
        # Load the model - olmOCR-7B-0225-preview is based on Qwen2-VL
        print(f"📥 Loading model from: {model_path}")
        
        # Check if GPU is available
        if torch.cuda.is_available():
            print(f"🚀 Using GPU: {torch.cuda.get_device_name(0)}")
            # Check available GPU memory
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU Memory: {gpu_memory:.1f} GB")
            
            # For large models (8B+ params), be more conservative with device mapping
            # But allow some GPU usage for SVD acceleration
            if gpu_memory < 12:  # Less than 12GB GPU memory
                print("⚠️  Limited GPU memory detected, using CPU with GPU SVD acceleration")
                device_map = {"": "cpu"}  # Model on CPU
                torch_dtype = torch.float32
            else:
                print("🚀 Sufficient GPU memory, using GPU acceleration")
                device_map = "auto"  # Let transformers handle GPU allocation
                torch_dtype = torch.float16  # Use float16 for GPU efficiency
        else:
            print("⚠️  GPU not available, using CPU")
            device_map = "cpu"
            torch_dtype = torch.float32
        
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=os.path.exists(local_model_path)  # Only use local files if they exist
        )
        
        print("✅ Model loaded successfully!")
        print(f"📊 Original model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"🎯 Model device: {next(model.parameters()).device}")
        print(f"🔢 Model dtype: {next(model.parameters()).dtype}")
        print(f"💾 Memory usage: {get_memory_usage():.1f} GB")
        
        # Quick diagnostic - check what layers would be compressed
        print("🔍 Analyzing model structure...")
        from pela_olmocr import ModulePELA
        pela_manager = ModulePELA(compress_ratio=1.2)  # Was 1.8
        
        target_layers = []
        for name, module in model.named_modules():
            if pela_manager._should_replace_module(name, module):
                target_layers.append((name, module.in_features, module.out_features))
        
        print(f"🎯 Found {len(target_layers)} layers to compress:")
        for i, (name, in_feat, out_feat) in enumerate(target_layers[:5]):  # Show first 5
            print(f"   {i+1}. {name}: {in_feat}x{out_feat}")
        if len(target_layers) > 5:
            print(f"   ... and {len(target_layers)-5} more layers")
        
        if len(target_layers) == 0:
            print("❌ No target layers found! Check target_modules configuration.")
            return False
        
        # Limit to first 20 layers for quality testing
        target_layers = target_layers[:20]
        print(f"🎯 LIMITED to first {len(target_layers)} layers for quality testing")
        
        # Check memory before compression
        memory_before = get_memory_usage()
        print(f"💾 Memory before compression: {memory_before:.1f} GB")
        
        # Force garbage collection
        gc.collect()
        
        # Apply PELA compression in batches
        print("🗜️  Applying PELA compression in batches...")
        
        # Smart batch sizing - smaller batches for large layers
        def get_batch_size(layers_info):
            max_size = max(max(in_feat, out_feat) for _, in_feat, out_feat in layers_info)
            if max_size > 4096:
                return 1  # Process large layers one at a time
            elif max_size > 2048:
                return 2  # Smaller batches for medium layers
            else:
                return 5  # Regular batch size for small layers
        
        # Group layers by size for optimal batching
        batches = []
        current_batch = []
        
        for layer_info in target_layers:
            current_batch.append(layer_info)
            batch_size = get_batch_size(current_batch)
            
            if len(current_batch) >= batch_size:
                batches.append(current_batch)
                current_batch = []
        
        if current_batch:  # Add remaining layers
            batches.append(current_batch)
        
        print(f"📦 Processing {len(target_layers)} layers in {len(batches)} smart-sized batches")
        
        # Checkpoint file for resuming if needed
        checkpoint_file = "pela_compression_checkpoint.json"
        
        try:
            from pela_olmocr import PELALinear
            import json
            
            # Load checkpoint if exists
            processed_layers = set()
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    processed_layers = set(checkpoint.get('processed_layers', []))
                    print(f"📋 Resuming from checkpoint: {len(processed_layers)} layers already processed")
            
            # Work directly on the model (no deepcopy to save memory)
            print("⚡ Processing layers in-place to save memory...")
            compressed_model = model  # Work directly on original model
            
            compression_stats = []
            total_original_params = 0
            total_compressed_params = 0
            
            # Create progress bars
            batch_pbar = tqdm(range(len(batches)), desc="🔄 Batches", unit="batch")
            
            for batch_idx in batch_pbar:
                batch_layers = batches[batch_idx]
                
                batch_pbar.set_description(f"🔄 Batch {batch_idx + 1}/{len(batches)}")
                print(f"\n💾 Memory before batch {batch_idx + 1}: {get_memory_usage():.1f} GB")
                
                # Process layers in this batch with progress
                layer_pbar = tqdm(batch_layers, desc="🗜️  Layers", leave=False, unit="layer")
                
                for layer_idx, (name, in_feat, out_feat) in enumerate(layer_pbar):
                    global_idx = target_layers.index((name, in_feat, out_feat)) # Re-calculate global_idx
                    layer_pbar.set_description(f"🗜️  {name} ({in_feat}x{out_feat})")
                    
                    # Skip if already processed
                    if name in processed_layers:
                        layer_pbar.set_postfix({'Status': 'SKIPPED'})
                        continue
                    
                    try:
                        # Get the actual module
                        original_module = compressed_model.get_submodule(name)
                        
                        # Replace the fixed rank with proportional scaling:
                        rank = int(min(in_feat, out_feat) * 0.5)  # 50% of smaller dimension
                        rank = min(rank, 2048)  # Cap at 2048 for memory safety
                        rank = max(rank, 256)   # Minimum 256
                        print(f"🔧 Proportional rank for {name} ({in_feat}x{out_feat}): {rank}")
                        
                        # Create PELA replacement
                        pela_module = PELALinear(
                            original_module,
                            rank=rank,
                            compress_ratio=1.2,  # Was 1.8
                            is_approximate=True
                        )
                        
                        # Replace the module
                        parent_name = '.'.join(name.split('.')[:-1])
                        child_name = name.split('.')[-1]
                        
                        if parent_name:
                            parent_module = compressed_model.get_submodule(parent_name)
                            # Explicitly delete the old module first
                            old_module = getattr(parent_module, child_name)
                            del old_module
                            setattr(parent_module, child_name, pela_module)
                        else:
                            # Explicitly delete the old module first
                            old_module = getattr(compressed_model, child_name)
                            del old_module
                            setattr(compressed_model, child_name, pela_module)
                        
                        # Track stats
                        stats = pela_module.get_parameter_count()
                        compression_stats.append((name, stats))
                        total_original_params += stats['original']
                        total_compressed_params += stats['pela']
                        
                        # Add to processed list and save checkpoint
                        processed_layers.add(name)
                        
                        # Save checkpoint every 10 layers
                        if len(processed_layers) % 10 == 0:
                            checkpoint_data = {
                                'processed_layers': list(processed_layers),
                                'compression_stats': [(n, s) for n, s in compression_stats],
                                'total_original_params': total_original_params,
                                'total_compressed_params': total_compressed_params
                            }
                            with open(checkpoint_file, 'w') as f:
                                json.dump(checkpoint_data, f)
                        
                        # Aggressive memory cleanup after each layer
                        del original_module  # Delete reference to original module
                        gc.collect()  # Force garbage collection after each layer
                        
                        # Update progress with compression info
                        layer_pbar.set_postfix({
                            'Rank': rank,
                            'Comp': f"{stats['compression_ratio']:.1f}x",
                            'Mem': f"{get_memory_usage():.1f}GB"
                        })
                        
                        # Monitor compression ratio per layer
                        actual_ratio = stats['compression_ratio']
                        if actual_ratio > 2.0:
                            print(f"⚠️  WARNING: {name} compression too aggressive: {actual_ratio:.1f}x")
                        
                    except Exception as layer_error:
                        print(f"❌ Failed to compress layer {name}: {layer_error}")
                        continue
                
                # Memory cleanup after each batch
                gc.collect()
                print(f"💾 Memory after batch {batch_idx + 1}: {get_memory_usage():.1f} GB")
                
                # Update overall progress
                batch_pbar.set_postfix({
                    'Memory': f"{get_memory_usage():.1f}GB",
                    'Layers': f"{len(compression_stats)}/{len(target_layers)}"
                })
            
            batch_pbar.close()
            print("\n✅ PELA compression successful!")
            print(f"💾 Final memory usage: {get_memory_usage():.1f} GB")
            
            # Clean up checkpoint file on successful completion
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
                print("🗑️  Checkpoint file cleaned up")
            
        except Exception as compression_error:
            print(f"❌ PELA compression failed during batch processing: {compression_error}")
            print(f"💾 Memory when failed: {get_memory_usage():.1f} GB")
            print(f"💾 Checkpoint saved at: {checkpoint_file}")
            print("💡 You can resume by running the script again")
            import traceback
            traceback.print_exc()
            return False
        
        compressed_params = sum(p.numel() for p in compressed_model.parameters())
        original_params = sum(p.numel() for p in model.parameters())
        compression_ratio = original_params / compressed_params
        memory_reduction = (1 - compressed_params / original_params) * 100
        
        print(f"\n📊 COMPRESSION RESULTS:")
        print(f"📊 Original parameters: {original_params:,}")
        print(f"📊 Compressed parameters: {compressed_params:,}")
        print(f"🗜️  Overall compression ratio: {compression_ratio:.2f}x")
        print(f"💾 Memory reduction: {memory_reduction:.1f}%")
        
        print(f"\n🎯 PELA LAYER STATS:")
        print(f"📊 PELA-compressed layers: {len(compression_stats)}")
        print(f"📊 PELA original parameters: {total_original_params:,}")
        print(f"📊 PELA compressed parameters: {total_compressed_params:,}")
        if total_original_params > 0:
            pela_compression_ratio = total_original_params / total_compressed_params
            pela_reduction = (1 - total_compressed_params / total_original_params) * 100
            print(f"🗜️  PELA-only compression ratio: {pela_compression_ratio:.2f}x")
            print(f"💾 PELA parameter reduction: {pela_reduction:.1f}%")
        
        # Show top compressed layers
        print(f"\n🏆 TOP COMPRESSED LAYERS:")
        sorted_stats = sorted(compression_stats, key=lambda x: x[1]['compression_ratio'], reverse=True)
        for i, (name, stats) in enumerate(sorted_stats[:5]):
            print(f"   {i+1}. {name}: {stats['compression_ratio']:.1f}x compression")
        
        # Save the compressed model with actual compression ratio in filename
        if total_original_params > 0:
            pela_compression_ratio = total_original_params / total_compressed_params
            output_path = f"olmocr_compressed_{pela_compression_ratio:.1f}x.pt"
        else:
            output_path = "olmocr_compressed_unknown.pt"
        
        print(f"\n💾 Saving compressed model to {output_path}...")
        try:
            torch.save(compressed_model, output_path)
            print(f"✅ Compressed model saved successfully!")
            print(f"📁 File location: {output_path}")
            print(f"📏 You can now run quality evaluation with:")
            print(f"   python demo_comparison.py --compressed {output_path}")
        except Exception as save_error:
            print(f"❌ Failed to save model: {save_error}")
            print(f"💡 Model is still in memory - you can access it via 'compressed_model'")
        
        return True
        
    except Exception as e:
        print(f"❌ PELA compression failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_pela_compression()
    if success:
        print("\n🎉 Real OLMoOCR PELA compression test PASSED!")
    else:
        print("\n⚠️  Real OLMoOCR PELA compression test FAILED") 