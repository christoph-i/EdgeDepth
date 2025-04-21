USER_NAME=$(whoami)

# Build command needs to run only if the image does not exist yet or needs an update! Just comment the following line out if the image exists already!
# docker build -t edgeyolo_train_$USER_NAME -f Dockerfile.Train .

export UID=$(id -u)
export GID=$(id -g)
# make sure the mappings to /training is correct and that /training contains the "train_params.yaml" config file expected by Dockerfile.Train!
# -v /PATH/TO/MOUNT:/mnt/EdgeDepth -> genreal mount -> make sure it contains the train data, EdgeYoloDepth Repo etc. If necessary adjust paths in the config file according to this mount 
docker run -u $UID:$GID -d \
    -v /PATH/TO/TRAIN/CONFIG:/training \
    -v /PATH/TO/EdgeDepth:/mnt/EdgeDepth \
    -- rm --shm-size 16G --name edgeyolo_train_01_$USER_NAME edgeyolo_train_$USER_NAME

