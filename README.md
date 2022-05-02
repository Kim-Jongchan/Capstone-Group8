# Molecule-Graph-Prediction

## Abstract
In this project, GNN has been introduced with self-attention mechanism to solve drug candidate screening problems. Specifically, the model is trained to predict whether a drug molecule can inhibit HIV replication or not. As a result, this model has showed the same performance with a smaller number of parameters. 

## Dataset
The dataset consists of each molecule information and whether the molecule inhibits HIV replication or not. (Binary Classification problem) The name of the dataset is ogbg-molhiv.
[More info](https://ogb.stanford.edu/docs/graphprop/#ogbg-mol)

## Model
Main Architecture consists of two parts : Transformer and GNN

![Transformer](https://github.com/Kim-Jongchan/Molecule-Graph-Prediction/blob/main/resources/Graphormer.jpg)
- [Main source](https://github.com/TencentYoutuResearch/HIG-GraphClassification)


![GNN](https://github.com/Kim-Jongchan/Molecule-Graph-Prediction/blob/main/resources/GNN.jpg)

## Experiment
Transformer only (Baseline) vs Transformer + GNN

![](https://github.com/Kim-Jongchan/Molecule-Graph-Prediction/blob/main/resources/GNN%20vs%20Graphormer.jpg)

## Result


|   | Transformer Only | Transformer + GNN |
| ------------- | ------------- | ------------- |
| Test ROCAUC  | 0.824  | 0.824  |
| Valid ROCAUC  | 0.837  | 0.839  |
| # of Parameters  | 532,418  | 456,074  |
| Epochs  | 20  | 75  |

