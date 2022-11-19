# Machine Learning Experiment Environment Boilerplate
Boilerplate for creating independent container environments for MachineLearning / DeepLearning training



# start container 
Each framework has its own docker-compose file.

## pytorch
```yaml
version: "3.8"
services:
  lab:
    build:
      context: .
      dockerfile: docker/pytorch/Dockerfile
      args:
        TAG: 1.12.0-cuda11.3-cudnn8-runtime
        USERNAME: ${USERNAME}
        USER_UID: ${UID}
        USER_GID: ${GID}
    command: tail -f /dev/null
    volumes:
      - ./:/experiment

    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [ gpu ]
```



```bash
UID=${UID} GID=${GID} USERNAME=yslee docker-compose -f ./docker-compose.torch.yaml  up -d --build
```


## TensorFlow

## jax/flax
TODO ...

## huggingface
TODO ...