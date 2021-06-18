# MIL-RNN nature-medicine-2019
This repository provides training and testing scripts for the article *Campanella et al. 2019*.
## Directory organisation
+ MIL-Train : Training MIL
+ MIL-Test : Inference MIL
+ RNN_train : Train RNN
+ RNN_Test : Test RNN
+ CheckSet.py: Open an check all images in {Train,Val}.json
+ CheckQuick.py: Open an check all images in {Train,Val}.json
+ Untitled.ipynb : Split Typical/Atypical **To rename**
+ Untitled1.ipynb : Experience data augmentation **To rename**
+ Res*.ipynb : Jupyter notebook model summarizing the results
+ {Train*,Val*}.json: Files need for torch DataLoader
## Dataset
{
  'Slides':['TNE1095' , 'TNE1411', ...],

  'Tiles' : [[TNE1095/Tiles1095_x_y.jpg, ...]
            [TNE1411/TNE1411_x_y.jpg, ...] ],

  'Targets': [0, 0, 1 ... ]
}
## Main modification from the original folder
- MIL:
  - DataLoader
  - Data augmentation:
    + Removed the normizalition
    + Add color augmentation
  - Output modification:
    + Save learning rate
    + Calculation of Training error, FPR, FNR
  - Training with learning rate schedule
  - Possibility to load a graph
- RNN:
  - Add Attention layer

## Experiment 1: Typical/Atypical
**Try to locate atypical area.**
### Dataset:
- ~1M Tiles 512x512px Vahadane color normalization Typical/Atypical (see *Untitled.ipynb*).
### Experience:
#### A - ResNet34 - Batch size 32
- Epoch 25
- Huge instability
- From ImageNet
#### B - Resnet34 - Batch size 64
- Epoch 30
- Huge instability
- From exp A
- + RNN

## Experiment 2: Tumoral/No Tumoral
See xlsx file to get a summary
#### Results organisation:
+ ModelName:
  - Checkpoint_best.pth
  - convergence.csv
  - prediction.csv // From MIL_test.py
  - probability.csv // From MIL_test.py
  - *.ipynb // results summary
  - BestTiles:
    - SampleName_{0,1}
      - Normal
        - Tiles_{}.jpg
      - Tumour
        - TIles_{}.jpg
+ ResMapVal
+ ResMapTrain
