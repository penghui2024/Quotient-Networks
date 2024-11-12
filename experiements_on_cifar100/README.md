## Dataset

Download the [CIFAR100 dataset](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz), unzip it to the data file, then run this command:
```
python ./src/parse_cifar100_to_png.py
```

## Training

To train the model(s) in the paper, run this command:

```train
python ./src/train.py --model q44
```

>📋  the command has one argument "model", you can choose one from "q44", "q56", "q110", "resnet44", "resnet56", and "resnet110".

## Evaluation

To evaluate my model, run:

```eval
python ./src/eval.py --model q56 --check_point ./pre_train/checkpoint_quonet56.pkl
```

>📋  The command has two arguments. For "model," you can choose one from "q44", "q56", "q110", "resnet44", "resnet56", and "resnet110". Then, for "check_point," you need to provide the path of the trained check_point.

## Pre-trained Models
You can find the pre-trained models(i.e. check_points) in the pre_train file:


>📋 There are one check_point in the pre_train file, for quotient network56. For more pre-trained models, after the paper openreview, I will post them on the web.

## Results

Our model achieves the following performance on CIFAR100:

### Image Classification on CIFAR100

| quotient model      | Accuracy         | ResNet    | Accuracy         |
|---------------------|------------------|-----------|------------------|
| quotient network44  | 73.25%&#177;0.27 | ResNet44  | 72.66%&#177;1.24 |
| quotient network56  | 73.53%&#177;0.18 | ResNet56  | 73.07%&#177;0.24 |
| quotient network110 | 73.00%&#177;0.55 | ResNet110 | 72.34%&#177;0.95 |
