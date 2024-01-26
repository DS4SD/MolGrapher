image_name=molgrapher
version=latest

docker run --ipc=host \
           -u nobody \
           --shm-size=4gb \
           $image_name:$version