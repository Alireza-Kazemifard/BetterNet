# --- START OF FILE cal_flops.py ---
import time
import tensorflow as tf
import numpy as np
from model import BetterNet
from utils import load_trained_model

def get_flops(model):
    total_params = model.count_params()
    flops = total_params * 2.0  
    return flops

def measure_inference_time(model, input_shape=(1, 224, 224, 3), iterations=100):
    dummy_input = tf.random.normal(input_shape)
    for _ in range(10):
        _ = model(dummy_input)

    print(f"⏱ Measuring inference time over {iterations} iterations...")
    times = []
    for _ in range(iterations):
        start = time.time()
        _ = model(dummy_input, training=False)
        end = time.time()
        times.append(end - start)
    
    avg_time = np.mean(times)
    fps = 1.0 / avg_time
    return avg_time, fps

if __name__ == "__main__":
    model = BetterNet(input_shape=(224, 224, 3))
    
    total_params = model.count_params()
    try:
        from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
        from tensorflow.python.profiler.model_analyzer import profile
        print("Attempting precise FLOPs calculation...")
    except:
        pass

    print("-" * 40)
    print(f"Model: BetterNet")
    print(f"Total Parameters: {total_params / 1e6:.2f} M (Paper reports ~11.5M)")
    
    avg_time, fps = measure_inference_time(model)
    print(f"Average Inference Time: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print("-" * 40)
