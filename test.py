import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
gpu_devices = tf.config.list_physical_devices('GPU')
if gpu_devices:
    print("GPU is available:", gpu_devices)
    # Optional: Run a small computation to see if it uses GPU
    try:
        with tf.device('/GPU:0'): # Or the name shown in gpu_devices
            a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
            c = tf.matmul(a, b)
        print("Matrix multiplication on GPU successful:", c.numpy())
    except RuntimeError as e:
        print("Error during GPU test computation:", e)
else:
    print("GPU not available, TensorFlow will use CPU.")
exit()