CLIENTS=$1
COMPRESSION=$2
IND=$((CLIENTS-1))
for i in $(seq 0 $IND);
do
  gcloud beta run jobs create flowerclient"$COMPRESSION""$i" --image=gcr.io/festive-freedom-351515/flower_client_experimental_gzip --region=europe-west4   \
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
  --args="--ip=34.90.96.210:8000" \
  --memory 4096Mi \
  --task-timeout 50m
  gcloud beta run jobs execute flowerclient"$COMPRESSION""$i" --region=europe-west4
done
