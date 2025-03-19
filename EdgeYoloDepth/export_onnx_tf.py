import os
import json
import yaml
import torch
import argparse
import onnx
import numpy as np
import onnx
from onnx_tf.backend import prepare
from loguru import logger
import tensorflow as tf 


from edgeyolo import EdgeYOLO



def get_args():
    parser = argparse.ArgumentParser("EdgeYOLO Export Parser")

    # basic
    parser.add_argument("--weights", type=str, default="/.../best.pth")
    parser.add_argument("--input-size", type=int, nargs="+", default=[384, 384])
    parser.add_argument("-b", '--batch', type=int, default=1, help='max batch size in detect')
    parser.add_argument("--depth-mode", type=bool, default=True, help='loaded model contains building blocks of depth model (esp. DepthHead)')

    # onnx
    parser.add_argument("--no-simplify", action="store_true", help="do not simplify models(not recommend)")
    parser.add_argument("--opset", type=int, default=11, help="onnx opset")
    
    return parser.parse_args()



def edgeyolo_pth_to_onnx(pth_weights_path: str, out_file_base_path: str, batchsize: int, input_size: tuple, opset: int, no_simplify=True, depth_mode=False):
    exp = EdgeYOLO(weights=pth_weights_path, depth_mode=depth_mode)
    model = exp.model

    model.fuse()
    model.eval()
    
    x = np.ones([batchsize, 3, *input_size], dtype=np.float32)
    x = torch.from_numpy(x)  # .cuda()

    model(x)  # warm and init

    input_names = ["input_0"]
    output_names = ["output_0"]
    
    onnx_file = out_file_base_path + ".onnx"
    torch.onnx.export(model,
                        x,
                        onnx_file,
                        verbose=False,
                        opset_version=opset,
                        input_names=input_names,
                        output_names=output_names,
                        dynamic_axes=None)
    onnx_model = onnx.load(onnx_file) 
    onnx.checker.check_model(onnx_model)  
    if not no_simplify:
        try:
            import onnxsim
            logger.info('\nstart to simplify ONNX...')
            onnx_model, check = onnxsim.simplify(onnx_model)
            assert check, 'assert check failed'
        except Exception as e:
            logger.error(f'Simplifier failure: {e}')

    onnx.save(onnx_model, onnx_file)
    logger.info(f'ONNX export success, saved as {onnx_file}')
    data_save = {
        "names": exp.class_names,
        "img_size": input_size,
        "batch_size": batchsize,
        "pixel_range": exp.ckpt.get("pixel_range") or 255,  # input image pixel value range: 0-1 or 0-255
        "obj_conf_enabled": True,  # Edge-YOLO use cls conf and obj conf
        "input_name": "input_0",
        "output_name": "output_0",
        "dtype": "float"
    }
    with open(out_file_base_path + ".yaml", "w") as yamlf:
        yaml.dump(data_save, yamlf)

    with open(out_file_base_path + ".json", "w") as jsonf:
        json.dump(data_save, jsonf)



@logger.catch
@torch.no_grad()
def main():

    args = get_args()

    if not os.path.isfile(args.weights):
        raise FileNotFoundError(f"No weights file found at given path {args.weights}. Current working dir is {os.getcwd()}")
    
    if isinstance(args.input_size, int):
        args.input_size = [args.input_size] * 2
    if len(args.input_size) == 1:
        args.input_size *= 2
        
    
    # export into same dir as inputs weights 
    export_dir = os.path.dirname(args.weights)
    weights_base_filename = os.path.basename(args.weights).split('.')[0]
    output_file_path_no_extension = os.path.join(export_dir, weights_base_filename + 
                             f"_{args.input_size[0]}x{args.input_size[1]}_" +
                             f"batch{args.batch}")
    
    # ONNX    
    edgeyolo_pth_to_onnx(args.weights, output_file_path_no_extension, args.batch, args.input_size, args.opset, args.no_simplify, depth_mode=args.depth_mode)
    onnx_file_path = output_file_path_no_extension + ".onnx"
    
    # ONNX -> TF saved model 
    onnx_model = onnx.load(onnx_file_path)
    tf_rep = prepare(onnx_model)
    tf_saved_model_path = output_file_path_no_extension + "_saved_model"
    tf_rep.export_graph(tf_saved_model_path)
    
    # TF saved model -> TFlite model
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_saved_model_path)
    tflite_model = converter.convert()
    tflite_model_file_path = output_file_path_no_extension + ".tflite"
    with open(tflite_model_file_path, 'wb') as f:
        f.write(tflite_model)
    

        
    
if __name__ == "__main__":
    main()