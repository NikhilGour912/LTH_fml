import torch
import torch.nn as nn
import torchvision.models as vgg16
import time
import numpy as np
import os
import gc

BATCH_SIZE = 1 
NUM_BATCHES = 1000 
BASELINE_PATH = "baseline_vgg16_m4.pth"
SPARSITY_TARGET = 73.79 


if not torch.backends.mps.is_available():
    device = torch.device("cpu")
else:
    device = torch.device("mps")
print(f"Using device: {device}")


def get_model():
    model = vgg16.vgg16(weights=None, num_classes=10).to(device)
    return model

def generate_mask_oneshot(model, prune_percent):
    print(f"Generating {prune_percent}% sparsity mask...")
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            all_weights.append(param.data.abs().view(-1))
    all_weights = torch.cat(all_weights)
    
    num_to_prune = int(len(all_weights) * prune_percent / 100.0)
    threshold = torch.kthvalue(all_weights, num_to_prune).values
    
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            mask[name] = (param.data.abs() > threshold).float()
    return mask

def apply_mask(model, mask):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name].to(device)

def benchmark(model, input_tensor, name="Model"):
    model.eval()
    
    print(f"Warming up {name}...")
    with torch.no_grad():
        for _ in range(50):
            _ = model(input_tensor)
            if device.type == 'mps':
                torch.mps.synchronize()
    
    print(f"Benchmarking {name}...")
    times = []
    with torch.no_grad():
        for _ in range(NUM_BATCHES):
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
                
            start = time.time()
            
            _ = model(input_tensor)
            
            if device.type == 'mps':
                torch.mps.synchronize()
            elif device.type == 'cuda':
                torch.cuda.synchronize()
                
            end = time.time()
            times.append((end - start) * 1000) 

    avg_time = np.mean(times)
    std_dev = np.std(times)
    fps = 1000 / avg_time * BATCH_SIZE
    
    print(f"--- RESULTS: {name} ---")
    print(f"Avg Latency: {avg_time:.4f} ms per batch (±{std_dev:.4f})")
    print(f"Throughput:  {fps:.2f} FPS")
    print("-" * 30)
    return avg_time


if __name__ == "__main__":
    
    if not os.path.exists(BASELINE_PATH):
        print("Error: Baseline model not found.")
        exit()
        
   
    dummy_input = torch.randn(BATCH_SIZE, 3, 32, 32).to(device)

    dense_model = get_model()
    dense_model.load_state_dict(torch.load(BASELINE_PATH, map_location=device))
    
    dense_latency = benchmark(dense_model, dummy_input, name="Dense VGG-16")
    
   
    mask = generate_mask_oneshot(dense_model, SPARSITY_TARGET)
    apply_mask(dense_model, mask) 
    
    sparse_latency = benchmark(dense_model, dummy_input, name=f"Sparse VGG-16 ({SPARSITY_TARGET}%)")
    
    
    print("\n=== FINAL BENCHMARK ANALYSIS ===")
    print(f"Device:         {device}")
    print(f"Sparsity Level: {SPARSITY_TARGET}%")
    print(f"Dense Latency:  {dense_latency:.4f} ms")
    print(f"Sparse Latency: {sparse_latency:.4f} ms")
    
    speedup = (dense_latency - sparse_latency) / dense_latency * 100
    print(f"Observed Speedup: {speedup:.2f}%")
    
    if abs(speedup) < 5:
        print("\n[CONCLUSION]: No significant speedup detected.")
        print("REASON: We used 'Unstructured Pruning' (setting individual weights to 0).")
        print("Standard GPUs (like Mac M4/NVIDIA) are optimized for DENSE matrix multiplication.")
        print("They still calculate '0 * x', consuming the same compute cycles.")
        print("To achieve speedup, we would need 'Structured Pruning' (removing entire channels).")
    else:
        print("\n[CONCLUSION]: Unexpected speedup detected (likely due to noise or caching).")