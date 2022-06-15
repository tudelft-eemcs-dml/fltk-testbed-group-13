# FedKNOW

For the original repository docs, check: https://github.com/LINC-BIT/FedKNOW/blob/main/README.md

# Running experiments

For 5 clients FedWEIT algorithm with 6 layer CNN model:

1. cd into the FedKNOW directory
2. Run the following command 
```
py -m single.main_WEIT --alg=WEIT --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=5 --round 20 --shard_per_user=5 --frac=1.0 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=200  --local_ep=2  --gpu=0
```

# Quick test

To verify whether everything is okay before doing the actual experiments:

1. cd into the FedKNOW directory
2. Run the following command

```
py -m single.main_WEIT --alg=WEIT --dataset=cifar100 --num_classes=100 --model=6layer_CNN --num_users=5 --round 2 --shard_per_user=5 --frac=1.0 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=20  --local_ep=2  --gpu=0
```

# Reproducing results for FedWEIT with LeNet

1. cd into the FedKNOW directory
2. Run the following command

```
py -m single.main_WEIT --alg=WEIT --dataset=cifar100 --num_classes=100 --model=LeNet --num_users=5 --round 20 --shard_per_user=5 --frac=1.0 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=200  --local_ep=2  --gpu=0
```

# Reproducing results for FedProx with LeNet

1. cd into the FedKNOW directory
2. Run the following command

```
py -m single.main_FedProx --alg=prox --dataset=cifar100 --num_classes=100 --model=LeNet --num_users=5 --round 20 --shard_per_user=5 --frac=1.0 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=200  --local_ep=2  --gpu=0 --lamb 1
```

# Running on a multi cluster with docker

To run a flower client:
1. cd into the FedKNOW directory
2. Run the following command to build a docker image (Linux)

```
./FedKNOW/build_image.sh
```
3. For Windows, you can build the image using:
```
docker build -t flower_client:latest . -f FedKNOW/docker/Dockerfile
```
4. Then you need to start the Flower server and the client as mentioned below.
## For a simpler and relatively quicker run:

To create the server (from root dir)
```
py -m FedKNOW.multi.server --num_users 5 --frac 1.0 --ip 127.0.0.1:8000 --epochs 20
```
Then open 5 terminals and create a client as shown below (--ip arg will change when running on a real cluster). Set the client id appropriately.
```
docker run flower_client --alg=WEIT --dataset=cifar100 --num_classes=100 --model=LeNet --num_users=5 --round 2 --shard_per_user=5 --frac=1.0 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=20  --local_ep=2  --gpu=0 --client_id <SOME_ID> --ip host.docker.internal:8000
```
## For reproducing the paper's experimental settings with Overlapped CIFAR-100

To create the server
```
py -m multi.server --num_users 5 --frac 1.0 --ip 127.0.0.1:8000 --epochs 200
```
To spawn the clients
```
docker run flower_client --alg=WEIT --dataset=cifar100 --num_classes=100 --model=LeNet --num_users=5 --round 20 --shard_per_user=5 --frac=1.0 --local_bs=40 --optim=Adam --lr=0.001 --lr_decay=1e-4 --task=10 --epoch=200  --local_ep=2  --gpu=0 --client_id <SOME_ID> --ip host.docker.internal:8000
```

To do a true distributed setup, the clients can be deployed to google cloud run as a job
To spawn a job on google cloud run  
```
gcloud beta run jobs create flowerclient"$i" --image=gcr.io/{project-name}/flower_client --region=us-central1   \
  --args="--alg=WEIT" \
  --args="--dataset=cifar100" \
  --args="--num_classes=100" \
  --args="--model=LeNet" \
  --args="--num_users=5" \
  --args="--round=2" \
  --args="--frac=1.0" \
  --args="--local_bs=40" \
  --args="--optim=Adam" \
  --args="--lr=0.001" \
  --args="--lr_decay=1e-4" \
  --args="--task=10" \
  --args="--epoch=20" \
  --args="--local_ep=2" \
  --args="--gpu=0" \
  --args="--client_id=$i" \
  --args="--ip={server-ip}" \
  --memory 4096Mi \
  --task-timeout 50m
```

To run the job on google cloud run 
```
gcloud beta run jobs execute flowerclient"$i" --region=us-central1
```

You can also deploy multiple clients using the preexisting script:
```
./scripts/cloudrun.sh {number_of_clients_to_run} {compression_algo}
```
