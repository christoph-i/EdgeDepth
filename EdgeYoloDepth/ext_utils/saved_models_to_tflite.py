import tensorflow as tf

saved_model_dirs = [
    r"...",
    r"..."
]
tflite_model_paths = [
    r"...",
    r"..."
]


for idx, saved_model_dir in enumerate(saved_model_dirs):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Save the model.
    with open(tflite_model_paths[idx], 'wb') as f:
        f.write(tflite_model)
