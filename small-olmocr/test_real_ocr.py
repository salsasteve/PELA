#!/usr/bin/env python3
"""
Test compressed model on real OCR tasks
"""

import torch
from PIL import Image
import requests
from io import BytesIO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
import time

def test_ocr_model(model_path, test_images=None):
    """Test OCR model on real images."""
    
    print(f"🔍 Loading model: {model_path}")
    
    # Load model and processor
    if model_path.endswith('.pt'):
        # Compressed model
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        processor = AutoProcessor.from_pretrained("allenai/olmOCR-7B-0225-preview")
        print("✅ Loaded compressed model")
    else:
        # Original model
        model = Qwen2VLForConditionalGeneration.from_pretrained(model_path)
        processor = AutoProcessor.from_pretrained(model_path)
        print("✅ Loaded original model")
    
    model.eval()
    
    # Test images (you can replace these with your own)
    if test_images is None:
        test_images = [
            "https://www.learningcontainer.com/wp-content/uploads/2020/06/sample-ocr-image.png",
            "https://tesseract-ocr.github.io/docs/img/eurotext.png"
        ]
    
    results = []
    
    for i, img_url in enumerate(test_images):
        print(f"\n📄 Testing image {i+1}/{len(test_images)}")
        
        try:
            # Load image
            if img_url.startswith('http'):
                response = requests.get(img_url)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(img_url)
            
            print(f"📷 Image size: {image.size}")
            
            # Prepare inputs
            prompt = "Extract all text from this image:"
            inputs = processor(text=prompt, images=image, return_tensors="pt")
            
            # Time the inference
            start_time = time.time()
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,
                    do_sample=False
                )
            
            inference_time = time.time() - start_time
            
            # Decode output
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)
            extracted_text = generated_text.split(prompt)[-1].strip()
            
            result = {
                'image': img_url,
                'extracted_text': extracted_text,
                'inference_time': inference_time,
                'success': True
            }
            
            print(f"⏱️  Inference time: {inference_time:.2f}s")
            print(f"📝 Extracted text: {extracted_text[:100]}...")
            
        except Exception as e:
            result = {
                'image': img_url,
                'error': str(e),
                'inference_time': None,
                'success': False
            }
            print(f"❌ Error: {e}")
        
        results.append(result)
    
    return results

def compare_models():
    """Compare original vs compressed model performance."""
    
    print("🔍 COMPARING ORIGINAL VS COMPRESSED MODEL")
    print("=" * 60)
    
    # Test original model
    print("\n📊 Testing ORIGINAL model...")
    try:
        original_results = test_ocr_model("models/olmOCR-7B-0225-preview")
        original_success = True
    except Exception as e:
        print(f"❌ Original model failed: {e}")
        original_results = []
        original_success = False
    
    # Test compressed model
    print("\n📊 Testing COMPRESSED model...")
    try:
        compressed_results = test_ocr_model("olmocr_compressed_1.5x.pt")
        compressed_success = True
    except Exception as e:
        print(f"❌ Compressed model failed: {e}")
        compressed_results = []
        compressed_success = False
    
    # Compare results
    print(f"\n📋 COMPARISON RESULTS:")
    print(f"Original model success: {original_success}")
    print(f"Compressed model success: {compressed_success}")
    
    if original_success and compressed_success:
        # Compare inference times
        orig_times = [r['inference_time'] for r in original_results if r['success']]
        comp_times = [r['inference_time'] for r in compressed_results if r['success']]
        
        if orig_times and comp_times:
            avg_orig_time = sum(orig_times) / len(orig_times)
            avg_comp_time = sum(comp_times) / len(comp_times)
            speedup = avg_orig_time / avg_comp_time
            
            print(f"⏱️  Average original time: {avg_orig_time:.2f}s")
            print(f"⏱️  Average compressed time: {avg_comp_time:.2f}s")
            print(f"🚀 Speedup: {speedup:.2f}x")
        
        # Compare text quality (simple character count comparison)
        for i, (orig, comp) in enumerate(zip(original_results, compressed_results)):
            if orig['success'] and comp['success']:
                orig_len = len(orig['extracted_text'])
                comp_len = len(comp['extracted_text'])
                similarity = min(orig_len, comp_len) / max(orig_len, comp_len) if max(orig_len, comp_len) > 0 else 0
                
                print(f"📝 Image {i+1} text similarity: {similarity:.2f}")
    
    return original_results, compressed_results

if __name__ == "__main__":
    # Test compressed model only
    print("🧪 Testing compressed model on real OCR tasks...")
    results = test_ocr_model("olmocr_compressed_1.5x.pt")
    
    # Print summary
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"\n📊 SUMMARY:")
    print(f"✅ Successful: {successful}/{total}")
    
    if successful > 0:
        avg_time = sum(r['inference_time'] for r in results if r['success']) / successful
        print(f"⏱️  Average inference time: {avg_time:.2f}s")
        print("🎉 Model appears to be working!")
    else:
        print("❌ Model failed all tests - compression destroyed functionality") 