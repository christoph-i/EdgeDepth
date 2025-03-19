import cv2

from edgeyolo_tflite_depth_ir import EdgeyoloTfliteDepthIr
from edgeyolo_tflite_ir import EdgeyoloTfliteIr
from edgeyolo_pytorch_depth_ir import EdgeyoloPytorchDepthIr
from edgeyolo_pytorch_ir import EdgeyoloPytorchIr
from Evaluation_Framework.visualisation.render_DetectionImages import show_single_detection_images


raw_img_dir = r"..."



model_path_pth_depth = r".../best.pth"

model_path_tflite_depth = r".../best_384x384_batch1.tflite"


edgeyolo_pth_depth_ir = EdgeyoloPytorchDepthIr(model_path_tflite_depth, "PTH-DEPTH")

edgeyolo_tflite_ir_depth = EdgeyoloTfliteDepthIr(model_path_tflite_depth, "TFLITE-DEPTH")




det_imgs = edgeyolo_tflite_ir_depth.run_inference_for_dir_of_images(raw_img_dir, max_imgs=5, shuffle_images=True, image_file_types=(".jpg"))
show_single_detection_images(det_imgs, raw_img_dir, image_display_size=(1440, 810), conf_th=0.25, show_image_filename=True)
