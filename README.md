# PreSTNet: Pre-trained Spatio-Temporal Network for Traffic Forecasting, Information Fusion.

![image](https://github.com/WoodSugar/PreSTNet/blob/main/img/model.png)

Source Code of the [PreSTNet](https://www.sciencedirect.com/science/article/pii/S1566253524000198).

This work has been accepted in Information Fusion, and the all of the source codes are being updated. 

If you find the paper or this repository is helpful, please cite it with the following format:

```
@article{fang-prestnet,
  title = {{{PreSTNet}}: {{Pre-trained}} Spatio-Temporal Network for Traffic Forecasting},
  author = {Fang, Shen and Ji, Wei and Xiang, Shiming and Hua, Wei},
  date = {2024},
  journaltitle = {Information Fusion},
  volume = {106},
  pages = {102241},
  issn = {1566-2535}
}
```

## We have provided the following information for reproducibility
1. Hyper-parameter settings in pre-training and fine-tuning phases
2. Pre-trained backbone on three evaluation datasets (SubwayBJ, TaxiBJ, and PeMS03)
3. Pre-processed data of the open-source PeMS03 dataset (saved on Google Drive)
4. Detailed model structure under the framework of PyTorch, [EasyTorch](https://github.com/cnstark/easytorch), and [BasicTS](https://github.com/zezhishao/BasicTS).

## More details is on the way
1. Training log file
2. Visualization result

## We are sorry that partial data is not provided due to commercial copyright agreement
1. Original and pre-processed data of SubwayBJ dataset
2. Original and pre-processed data of TaxiBJ dataset

## Acknowledgement
We appreciate the EasyTorch and BasicTS toolboxes to support this work.
