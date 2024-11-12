## Dataset

Download the [CIFAR10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), unzip it to the data file, then run this command:
```
python ./src/parse_cifar10_to_png.py
```


## Training

To train the model(s) in the paper, run this command:

```train
python ./src/train.py --model resnet56
```

>📋  the command has one argument "model", you can choose one from "q44", "q56", "q110", "resnet44", "resnet56", and "resnet110".

## Evaluation

To evaluate my model, run:

```eval
python ./src/eval.py --model resnet110 --check_point ./pre_train/checkpoint_resnet110.pkl
```

>📋  The command has two arguments. For "model," you can choose one from "q44", "q56", "q110", "resnet44", "resnet56", and "resnet110". Then, for "check_point," you need to provide the path of the trained check_point.

## Pre-trained Models

You can find the pre-trained models(i.e. check_points) in the pre_train file:

>📋  There are two check_points in the pre_train file, one for resnet110, and the other for quotient network110. For more pre-trained models, after the paper openreview, I will post them on the web.

## Results

Our model achieves the following performance on CIFAR10:

### Image Classification on CIFAR10

| quotient model      | Accuracy         | ResNet    | Accuracy         |
|---------------------|------------------|-----------|------------------|
| quotient network44  | 92.78%&#177;0.25 | ResNet44  | 92.61%&#177;0.33 |
| quotient network56  | 93.10%&#177;0.15 | ResNet56  | 92.84%&#177;0.18 |
| quotient network110 | 93.44%&#177;0.17 | ResNet110 | 93.02%&#177;0.33 |

### Visualization

To visualize the first three quotient features and residual features, run this command

```eval
python ./src/visual.py --num 10 --layer 110 --check_point_quo ./pre_train/checkpoint_quonet110.pkl --check_point_res ./pre_train/checkpoint_resnet110.pkl
```
>📋 This command has four arguments. For "num," you need to choose which picture in all 10000-picture test sets to input the network. For "layer," you need to choose the number of layers, which is 44, 56, or 110. For "check_point_quo" and "check_point_res," you need to provide the path of checkpoints of the quotient network and the resnet, respectively.