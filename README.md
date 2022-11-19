# Machine Learning Experiment Environment Boilerplate
Boilerplate for creating independent container environments for MachineLearning / DeepLearning training


# Start Container 
Each framework has its own docker-compose file.

# pytorch

## docker-compsoe 
```yaml
version: "3.8"
services:
  lab:
    build:
      context: .
      dockerfile: docker/pytorch/Dockerfile
      args:
        TAG: 1.12.1-cuda11.3-cudnn8-runtime
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

Docker build args TAG candidate check [pytorch docker hub](https://hub.docker.com/r/pytorch/pytorch/tags) default TAG is ["1.12.1-cuda11.3-cudnn8-runtime"](https://hub.docker.com/layers/pytorch/pytorch/1.12.1-cuda11.3-cudnn8-runtime/images/sha256-0bc0971dc8ae319af610d493aced87df46255c9508a8b9e9bc365f11a56e7b75?context=explore)


```bash
UID=${UID} GID=${GID} USERNAME={USER USER NAME} docker-compose -f ./docker-compose.torch.yaml  up -d --build
```

## Container Test 
```bash
# Attetch container shell
docker-compose -f docker-compose.torch.yaml  exec lab bash
# inside container 
cd samples/pytorch
pip install tensorboard
python main.py
[0]:  2.2146: 100%|████████████████████| 59/59 [00:03<00:00, 16.25it/s]
[1]:  2.0021: 100%|████████████████████| 59/59 [00:02<00:00, 22.74it/s]
[2]:  1.6061: 100%|████████████████████| 59/59 [00:02<00:00, 22.04it/s]
...
```

# TensorFlow

```yaml
version: "3.8"
services:
  lab:
    build:
      context: ./
      dockerfile: docker/tensorflow/Dockerfile
      args:
        TAG: 2.11.0-gpu
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

Docker build args TAG candidate check [tensorflow docker hub](https://hub.docker.com/r/tensorflow/tensorflow/tags) default TAG is ["2.11.0-gpu"](https://hub.docker.com/layers/tensorflow/tensorflow/2.11.0-gpu/images/sha256-67f1a7b35fd52bdda071c0cd311655be7477f2bc1b6f27e014b9a57231bd55b3?context=explore)

```bash
UID=${UID} GID=${GID} USERNAME={USER USER NAME} docker-compose -f ./docker-compose.tf.yaml  up -d --build
```

## Container Test 
```bash
# Attetch container shell
docker-compose -f docker-compose.torch.yaml  exec lab bash
# inside container 
cd samples/pytorch
pip install tensorboard
python main.py
59/59 [==============================] - 3s 8ms/step - loss: 0.8085 - sparse_categorical_accuracy: 0.7895 - val_loss: 0.3501 - val_sparse_categorical_accuracy: 0.9045
Epoch 2/10
59/59 [==============================] - 0s 5ms/step - loss: 0.3075 - sparse_categorical_accuracy: 0.9150 - val_loss: 0.2621 - val_sparse_categorical_accuracy: 0.9242
...
```





# jax/flax
TODO ...

# huggingface
TODO ...