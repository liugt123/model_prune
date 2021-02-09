# model_pruning
A pytorch toolkit for structured neural network pruning and layer dependency maintaining.

**need set prune prob based on project**

## Example: ResNet18 on Cifar10

* download cifar10
```
Downloading https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz to ./data/cifar-10-python.tar.gz
Extracting ./data/cifar-10-python.tar.gz to ./data
```

* Train the model
```shell
python prune_resnet18_cifar10.py --mode train
```

* Pruning and fintuning
```shell
python prune_resnet18_cifar10.py --mode prune --round 1 --total_epochs 30 --step_size 20
python prune_resnet18_cifar10.py --mode prune --round 2 --total_epochs 30 --step_size 20
python prune_resnet18_cifar10.py --mode prune --round 3 --total_epochs 30 --step_size 20
python prune_resnet18_cifar10.py --mode prune --round 4 --total_epochs 30 --step_size 20
python prune_resnet18_cifar10.py --mode prune --round 5 --total_epochs 30 --step_size 20
python prune_resnet18_cifar10.py --mode prune --round 6 --total_epochs 30 --step_size 20
...
```

* Test
```
prune_resnet18_cifar10.py --mode test --round N
```

* TensorRT speed test
```shell 
prune_resnet18_cifar10.py --mode tensorrt --round N
```

* result

|  Model                       | Parameter size | Acc    |FP32 FPS|
|  :------:                    | :------------: | :----: | :----: |
| resnet18_cifar10             | 11.2M          | 92.54% | 1235   |
| resnet18_cifar10_prun_round1 | 4.5M           | 92.50% | 1285   |
| resnet18_cifar10_prun_round2 | 1.9M           | 92.37% | 1575   |
| resnet18_cifar10_prun_round3 | 0.8M           | 91.76% | 1728   |
| resnet18_cifar10_prun_round4 | 0.4M           | 91.33% | 1989   |
| resnet18_cifar10_prun_round5 | 0.2M           | 90.42% | 2435   |
| resnet18_cifar10_prun_round6 | 0.1M           | 89.10% | 2441   |
| resnet50_cifar10             | 23.5M          | 91.91% | 602    |
| resnet50_cifar10_prun_round1 | 8.8M           | 91.96% | 556    |
| resnet50_cifar10_prun_round2 | 4.1M           | 92.10% | 852    |
| resnet50_cifar10_prun_round3 | 2.2M           | 92.32% | 942    |
| resnet50_cifar10_prun_round4 | 1.3M           | 91.33% | 1121   |
| resnet50_cifar10_prun_round5 | 0.7M           | 87.43% | 1209   |
