| Rank | Model | Type | Kaggle-61 top-1 | New signers top-1 / top-5 | Combined top-1 | Combined top-5 | Precision | Recall | Macro F1 | INCLUDE part top-1 | Params | ms per clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | LSTM | keypoints | 98.4 | 24.6 / 43.7 |  |  |  |  |  |  | 0.72 M | 1.6 |
| 2 | GRU | keypoints | 99.5 | 21.0 / 46.7 |  |  |  |  |  |  | 0.57 M | 1.5 |
| 3 | Transformer | keypoints | 100.0 | 15.0 / 37.1 |  |  |  |  |  |  | 1.47 M | 1.1 |
| 4 | BiLSTM + attention | keypoints | 99.5 | 15.0 / 34.7 |  |  |  |  |  |  | 1.67 M | 3.1 |
| 5 | Random Forest | classical | 99.5 | 12.0 / 37.1 |  |  |  |  |  |  |  | 14.6 |
| 6 | BiGRU | keypoints | 99.5 | 11.4 / 40.1 |  |  |  |  |  |  | 1.30 M | 3.0 |
| 7 | 1-D CNN (TCN) | keypoints | 100.0 | 6.6 / 33.5 |  |  |  |  |  |  | 0.86 M | 0.9 |
| 8 | ResNet-18 + BiLSTM (ImageNet) | pretrained, pixels | 99.5 | 2.4 / 10.8 |  |  |  |  |  |  | 0.71 M | 86.2 |
| 9 | R3D-18 3-D CNN (Kinetics) | pretrained, pixels | 98.4 | 0.6 / 11.4 |  |  |  |  |  |  | 0.03 M | 87.0 |
| 10 | SVM | classical | 96.7 | 0.0 / 0.0 |  |  |  |  |  |  |  | 0.5 |
