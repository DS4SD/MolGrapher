sudo systemctl start docker

image_name=molgrapher
version=latest

DOCKER_BUILDKIT=1 docker build --ssh default \
                               --progress=plain \
                               -t $image_name:$version \
                               -f Dockerfile .