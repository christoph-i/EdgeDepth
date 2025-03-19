import torch
import torch.nn as nn
import onnx
import os
import tensorflow as tf
from onnx2tf import convert

from disnet import DisNet, DisNetSingleClass, DisNetClassIds, DisNetSingleClassVehicleBottom, DisNetClassIdsInclXY

def convert_pth_to_tflite(pth_path, out_dir, model):
    # Load the PyTorch model
    checkpoint = torch.load(pth_path)
    model.load_state_dict(checkpoint['weights'])
    model.eval()

    # Dummy input for the model
    dummy_input = torch.randn(1, 6)

    # Export the model to ONNX format
    onnx_path = os.path.join(out_dir, "model.onnx")
    torch.onnx.export(model, dummy_input, onnx_path, input_names=['input'], output_names=['output'])

    # Convert ONNX model to TensorFlow format using onnx2tf
    tf_path = os.path.join(out_dir, "model_tf")
    convert(
        input_onnx_file_path=onnx_path,
        output_folder_path=tf_path,
        output_signaturedefs=True,
        output_h5=False,
    )

    # Convert the TensorFlow model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(os.path.join(out_dir, "model.tflite"), "wb") as f:
        f.write(tflite_model)

# Provide the paths to the .pth file and the output .tflite file
pth_path = r".../.../best.pth"
out_dir = r".../.../..."

# Convert the model


class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)






args = Args()

# Hardcoded necessary params


args.lr = 1e-4 # default = 1e-4
args.device = "cpu"
args.optimizer = "adam"  # choices=("adam", "sgd")
args.max_patience = 100 # default 20
args.test_only = False  # Test only without training
args.accumulation_steps = 4 # default = 4 (number of mini batches accumulated before gradient updates)
args.epochs = 1500 # default = 1000 (as suggested in paper)
args.batch_size = 2 # default = 2
args.weight_decay = 1e-5 # default = 1e-5
args.n_workers = 0 # default = 4
args.scheduler = "cosine" # default = "cosine" | choices=("cosine", "plateau", "none")

args.wandb = False
args.long_range = False
args.use_debug_dataset = False


# single class mode
args.single_class_active_class = "vehicle"

# hardcoded optional params

args.checkpoint = None # Path to checkpoint to load - if None train from scratch
args.resume = False
args.exp_name = None


model = DisNetClassIdsInclXY(args=args)

convert_pth_to_tflite(pth_path, out_dir, model)
