# FedKNOW

For the original repository docs, check: https://github.com/LINC-BIT/FedKNOW/blob/main/README.md

# Running experiments

For 5 clients FedWEIT algorithm:

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


