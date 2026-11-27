| Rank | Model | Type | Kaggle-61 top-1 | New signers top-1 / top-5 | Combined top-1 | Combined top-5 | Precision | Recall | Macro F1 | INCLUDE part top-1 | Params | ms per clip |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | Transformer | keypoints | 100.0 | 41.0 / 71.1 | 95.8 | 98.8 | 95.7 | 95.5 | 95.1 | 94.5 | 1.52 M | 0.8 |
| 2 | 1-D CNN (TCN) | keypoints | 100.0 | 28.9 / 57.8 | 95.8 | 98.8 | 96.6 | 95.7 | 95.4 | 94.5 | 0.91 M | 0.7 |
| 3 | GRU | keypoints | 99.5 | 32.5 / 61.4 | 94.4 | 98.4 | 94.9 | 94.6 | 93.9 | 92.9 | 0.62 M | 1.4 |
| 4 | BiLSTM + attention | keypoints | 99.5 | 31.3 / 50.6 | 94.8 | 98.3 | 94.7 | 94.4 | 93.8 | 93.3 | 1.77 M | 3.0 |
| 5 | BiGRU | keypoints | 99.5 | 32.5 / 59.0 | 94.3 | 98.1 | 94.1 | 94.0 | 93.3 | 92.6 | 1.40 M | 2.9 |
| 6 | ResNet-18 + BiLSTM (ImageNet) | pretrained, pixels | 99.5 | 2.4 / 10.8* | 90.4 | 96.8 | 91.4 | 91.0 | 90.0 | 87.8 | 0.77 M | 235.9 |
| 7 | LSTM | keypoints | 98.4 | 30.1 / 47.0 | 90.3 | 97.9 | 89.4 | 89.2 | 88.1 | 87.5 | 0.77 M | 1.6 |
| 8 | Random Forest | classical | 99.5 | 25.3 / 65.1 | 85.4 | 96.1 | 85.2 | 85.3 | 83.5 | 81.7 |  | 13.2 |
| 9 | R3D-18 3-D CNN (Kinetics) | pretrained, pixels | 98.4 | 0.6 / 11.4* | 80.0 | 93.9 | 82.7 | 81.2 | 79.9 | 75.3 | 0.16 M | 218.5 |
| 10 | SVM | classical | 96.7 | 14.5 / 49.4 | 66.9 | 82.0 | 69.7 | 66.5 | 65.5 | 59.9 |  | 3.7 |

New signers: trained on Kaggle-61 (one signer) with the aspect fix, scored on the held-out half of the INCLUDE clips of the 9 shared words (83 clips, 61-way choice, chance 1.6%). Ranked by combined macro F1; models within 1 point of the best are tied and ranked by new-signer top-1. * pixel models: Kaggle-61 run, all 167 clips.
