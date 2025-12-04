"""
Script to check if GPU or CPU is being used during graph building.

This script monitors:
1. Which device sentence-transformers is using
2. Which device PyTorch is using (if available)
3. Real-time CPU and GPU usage during processing
"""

import sys
import time
import psutil
from pathlib import Path

print("=" * 70)
print("DEVICE USAGE CHECKER - Graph Building")
print("=" * 70)
print()

# Check if GPU monitoring is available
try:
    import GPUtil
    gpu_available = True
    print("✓ GPUtil available - GPU monitoring enabled")
except ImportError:
    gpu_available = False
    print("✗ GPUtil not available - Install with: pip install gputil")

print()

# Check PyTorch
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
    else:
        print("⚠️  PyTorch is using CPU only")
    print()
except ImportError:
    print("PyTorch not installed")
    print()

# Check sentence-transformers
try:
    from sentence_transformers import SentenceTransformer

    print("Testing sentence-transformers device:")
    print("-" * 70)

    # Load a small model to test
    model_name = "all-MiniLM-L6-v2"
    print(f"Loading model: {model_name}")

    start_time = time.time()
    model = SentenceTransformer(model_name)
    load_time = time.time() - start_time

    # Check which device the model is using
    device = model.device
    print(f"Model device: {device}")
    print(f"Load time: {load_time:.2f} seconds")

    # Test encoding
    print("\nTesting encoding speed...")
    test_texts = ["This is a test sentence."] * 100

    start_time = time.time()
    embeddings = model.encode(test_texts, show_progress_bar=False)
    encode_time = time.time() - start_time

    print(f"Encoded {len(test_texts)} sentences in {encode_time:.2f} seconds")
    print(f"Speed: {len(test_texts)/encode_time:.1f} sentences/second")
    print()

except ImportError:
    print("sentence-transformers not installed")
    print()

# Check spaCy
try:
    import spacy
    print("Testing spaCy:")
    print("-" * 70)

    nlp = spacy.load("en_core_web_sm")

    # Check if spaCy is using GPU
    if spacy.prefer_gpu():
        print("✓ spaCy is using GPU")
    else:
        print("✗ spaCy is using CPU")

    print()
except ImportError:
    print("spaCy not installed")
    print()

# Real-time monitoring function
def monitor_usage(duration_seconds=10):
    """Monitor CPU and GPU usage in real-time."""
    print("=" * 70)
    print(f"MONITORING SYSTEM USAGE FOR {duration_seconds} SECONDS")
    print("=" * 70)
    print()

    start_time = time.time()
    samples = []

    while time.time() - start_time < duration_seconds:
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=0.5)
        memory = psutil.virtual_memory()

        sample = {
            'time': time.time() - start_time,
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3)
        }

        # GPU usage (if available)
        if gpu_available:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # First GPU
                    sample['gpu_load'] = gpu.load * 100
                    sample['gpu_memory_used'] = gpu.memoryUsed
                    sample['gpu_memory_total'] = gpu.memoryTotal
            except:
                pass

        samples.append(sample)

        # Print current status
        status = f"[{sample['time']:.1f}s] CPU: {sample['cpu_percent']:.1f}% | RAM: {sample['memory_percent']:.1f}%"
        if 'gpu_load' in sample:
            status += f" | GPU: {sample['gpu_load']:.1f}% | GPU RAM: {sample['gpu_memory_used']:.0f}/{sample['gpu_memory_total']:.0f} MB"
        print(status)

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if samples:
        avg_cpu = sum(s['cpu_percent'] for s in samples) / len(samples)
        max_cpu = max(s['cpu_percent'] for s in samples)
        avg_memory = sum(s['memory_percent'] for s in samples) / len(samples)

        print(f"CPU usage - Average: {avg_cpu:.1f}% | Max: {max_cpu:.1f}%")
        print(f"RAM usage - Average: {avg_memory:.1f}%")

        if 'gpu_load' in samples[0]:
            avg_gpu = sum(s['gpu_load'] for s in samples) / len(samples)
            max_gpu = max(s['gpu_load'] for s in samples)
            print(f"GPU usage - Average: {avg_gpu:.1f}% | Max: {max_gpu:.1f}%")

            if avg_gpu < 5:
                print("\n⚠️  WARNING: GPU usage is very low! Your code is likely running on CPU.")
            else:
                print("\n✓ GPU is being actively used!")
        else:
            print("\n⚠️  GPU monitoring not available")

print()
print("=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print()

# Provide recommendations
recommendations = []

try:
    import torch
    if not torch.cuda.is_available():
        recommendations.append(
            "• Install CUDA-enabled PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )
except:
    pass

try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if str(model.device) == 'cpu':
        recommendations.append(
            "• sentence-transformers is using CPU. To use GPU, ensure PyTorch with CUDA is installed."
        )
except:
    pass

if not gpu_available:
    recommendations.append(
        "• Install GPUtil for GPU monitoring: pip install gputil"
    )

if recommendations:
    print("To enable GPU acceleration:")
    for rec in recommendations:
        print(rec)
else:
    print("✓ All GPU libraries are properly configured!")

print()
print("=" * 70)
print("MANUAL TESTING")
print("=" * 70)
print()
print("To check device usage while building your graph:")
print("1. Open a NEW terminal/command prompt")
print("2. Run: python check_device_usage.py --monitor")
print("3. In ANOTHER terminal, run: python fact_check_kg/build_graph.py")
print("4. Watch the first terminal for CPU/GPU usage")
print()

# Optional: Run monitoring if --monitor flag is provided
if len(sys.argv) > 1 and sys.argv[1] == '--monitor':
    try:
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        monitor_usage(duration)
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
