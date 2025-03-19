USER_NAME=$(whoami)

docker build -t edgeyolo_devel_$USER_NAME -f Dockerfile.Devel .

export UID=$(id -u)
export GID=$(id -g)
docker run -u $UID:$GID -d \
    -v /PATH/TO/TRAIN/CONFIG:/training \
    -v /PATH/TO/MOUNT:/mnt/shared_dir \
    --shm-size 16G --name edgeyolo_devel_$USER_NAME -it --entrypoint bash edgeyolo_devel_$USER_NAME