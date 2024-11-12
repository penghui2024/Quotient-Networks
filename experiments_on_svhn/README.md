## Dataset

to download the SVHN [train dataset](http://ufldl.stanford.edu/housenumbers/train_32x32.mat) and [test dataset](http://ufldl.stanford.edu/housenumbers/test_32x32.mat), put them to the data file, then run this command:
```
python ./src/parse_svhn_to_png.py
```

## Training

To train the model(s) in the paper, run this command:

```train
python ./src/train.py --model q56
```

>📋  the command has one argument "model", you can choose one from "q44", "q56", "q110", "resnet44", "resnet56", and "resnet110".

## Evaluation

To evaluate my model, run:

```eval
python ./src/eval.py --model resnet56 --check_point ./pre_train/checkpoint_resnet56.pkl
```

>📋  The command has two arguments. For "model," you can choose one from "q44", "q56", "q110", "resnet44", "resnet56", and "resnet110". Then, for "check_point," you need to provide the path of the trained check_point.

## Pre-trained Models

You can find the pre-trained models(i.e. check_points) in the pre_train file:

>📋  There are two check_points in the pre_train file, one for resnet110, and the other for quotient network110. For more pre-trained models, after the paper openreview, I will post them on the web.

## Results

Our model achieves the following performance on SVHN:

### Image Classification on SVHN

| quotient model      | Accuracy         | quotient model | Accuracy         |
|---------------------|------------------|----------------|------------------|
| quotient network44  | 96.17%&#177;0.12 | ResNet44       | 95.98%&#177;0.04 |
| quotient network56  | 96.20%&#177;0.11 | ResNet56       | 95.96%&#177;0.06 |
| quotient network110 | 96.12%&#177;0.05 | ResNet110      | 96.03%&#177;0.01 |
