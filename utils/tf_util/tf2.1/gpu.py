import tensorflow as tf


def gpus_type():
    """detect types of each GPU based on TensorFlow 2
    NOTE: this solution is subject to `CUDA_VISIBLE_DEVICES`.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        gpu_types = {i: gpu.name for i, gpu in enumerate(gpus)}
    else:
        gpu_types = {}

    print("GPU types:", gpu_types)
    return gpu_types
