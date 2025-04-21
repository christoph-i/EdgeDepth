

## Quickstart example

### Prepare data 
1. Download the kitti 3D OD dataset and convert the images to 1:1 aspect ratio. Place the images into the EdgeDepth/Dataset/kitti/images_1x1 dir. (If you just want to quickly verify the environment works 100 example images from kitti are already provided in the dir). Labels for all images are provided with this repo. 
2. Do only if you are NOT using the provided Docker env to train or use  custom dataset: Modify paths in the data yaml file at params/dataset/kitti_train_val_norm.yaml to match you local path.


### Train model 
1. Adjust paths in the train_model.sh script to match your dir 
    - Replace /PATH/TO/TRAIN/CONFIG with path to your train dir, e.g. "/home/EdgeDepth/EdgeYoloDepth/kitti_train_example" (The mounted train dir must always include a train_params.yaml file)
    - Replace /PATH/TO/EdgeDepth with your path, e.g. "/home/EdgeDepth"
2. Run the train script with './train_model.sh' (you might need to make it execuable first with 'chmod +x train_model.sh'). Resulting models and output will be saved to the kitti_train_example dir. 


## Convert models 
To convert models to ONNX and TensorFlow Saved Model or TFLite format the train docker container can be reused. 

Attach your shell to the container (e.g. 'docker exec -it YOU_CONTAINER bash'). 
Activate the venv (e.g. 'source /install/venv/bin/activate')
Run the conversion script with respective args, e.g. 'python export_onnx_tf.py --weights /training/best.pth'

ONNX, Saved Model and TFLite model files will automatically be saved to the same dir as the provided .pth file. 


## Full train documentation



## Evaluate 

### Standard OD evaluation 


### Depth evaluation


## TODO

* [ ] Merge readme with original readme
- [ ] Verfiy integrations like TensorRT with the depth variant
- [ ] Add capabilty to handle images which are not 1:1 dynamically instead of needing to convert images manually before 