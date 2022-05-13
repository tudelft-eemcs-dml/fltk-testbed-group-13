# Federated Continual Learning with Weighted Inter-client Transfer

## Environmental Setup

Please install packages from `requirements_fedweit.txt` after creating your own environment with `python 3.8.x` or `python 3.9.x`.

```bash
$ pip install --upgrade pip
$ pip install -r requirements_fedweit.txt
```

## Data Generation
Please see `config.py` to set your custom path for both `datasets` and `output files`.
```python
args.task_path = '/path/to/task/'  # for dataset
args.output_path = '/path/to/outputs/' # for logs, weights, etc.
```
Run below script to generate datasets
```bash
$ cd scripts
$ sh gen-data.sh
```
or you may run the following comamnd line directly:

```bash
python3 ../main.py --work-type gen_data --task non_iid_50 --seed 777 
```
It automatically downloads `8 heterogeneous datasets`, including `CIFAR-10`, `CIFAR-100`, `MNIST`, `Fashion-MNIST`, `Not-MNIST`, `TrafficSigns`, `Facescrub`, and `SVHN`, and finally processes to generate `non_iid_50` dataset.

## Run Experiments
To reproduce experiments, please execute `train-non-iid-50.sh` file in the `scripts` folder, or you may run the following comamnd line directly:

```bash
python3 ../main.py --gpu 0,1,2,3,4 \
		--work-type train \
		--model fedweit \
		--task non_iid_50 \
	 	--gpu-mem-multiplier 9 \
		--num-rounds 20 \
		--num-epochs 1 \
		--batch-size 100 \
		--seed 777 
```
Please replace arguments as you wish, and for the other options (i.e. hyper-parameters, etc.), please refer to `config.py` file at the project root folder.

> Note: while training, all participating clients are logically swiched across the physical gpus given by `--gpu` options (5 gpus in the above example). 

## Results
All clients and server create their own log files in `\path\to\output\logs\`, which include evaluation results, such as local & global performance and communication costs, and the experimental setups, such as learning rate, batch-size, etc. The log files will be updated for every comunication rounds. 

