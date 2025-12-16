
import tensorflow as tf
from tensorflow.python.profiler.model_analyzer import profile
from tensorflow.python.profiler.option_builder import ProfileOptionBuilder
from model import BetterNet

def get_flops(model):
    try:
        forward_pass = tf.function(
            model.call,
            input_signature=[tf.TensorSpec(shape=(1,) + model.input_shape[1:], dtype=tf.float32)]
        )
        graph = forward_pass.get_concrete_function().graph
        options = ProfileOptionBuilder.float_operation()
        graph_info = profile(graph, options=options)
        return graph_info.total_float_ops
    except Exception as e:
        print(f"Warning: Could not calculate FLOPs due to TF version mismatch: {e}")
        return 0

if __name__ == "__main__":
    input_shape = (256, 256, 3)
    model = BetterNet(input_shape=input_shape)
    
    total_params = model.count_params()
    flops = get_flops(model)
    
    print("-" * 30)
    print(f"Model: BetterNet")
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"GFLOPs: {flops / 1e9:.4f} G")
    print("-" * 30)
