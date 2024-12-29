import tensorflow as tf
from tensorflow.python.client import device_lib


def gpus_type():
    """detect types of each GPU based on TensorFlow 1.x
    NOTE: this solution is subject to `CUDA_VISIBLE_DEVICES`.
    """
    devices = device_lib.list_local_devices()
    gpus = [device for device in devices if device.device_type == 'GPU']
    if gpus:
        gpu_types = {i: gpu.physical_device_desc for i, gpu in enumerate(gpus)}
    else:
        gpu_types = {}

    print("GPU types:", gpu_types)
    return gpu_types
