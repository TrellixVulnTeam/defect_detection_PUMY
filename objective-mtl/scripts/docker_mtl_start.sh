docker run -it --runtime=nvidia \
           --rm -P --shm-size=120g \
           --ulimit memlock=-1 \
           --ulimit stack=67108864 \
           --name objective-mtl \
           --net host -v {source_path}:{dest_path} \
           {docker_name}:{tag} bash
