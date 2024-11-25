# Explicit High-level Semantic Network for Domain Generalization in Hyperspectral Image Classification

This is the official code of EHSnet. 
Paper web page: [Explicit High-level Semantic Network for Domain Generalization in Hyperspectral Image Classification](https://ieeexplore.ieee.org/document/10750220).

## Citation
```

@ARTICLE{10750220,
  author={Wang, Xusheng and Dong, Shoubin and Zheng, Xiaorou and Lu, Runuo and Jia, Jianxin},
  journal={IEEE Transactions on Geoscience and Remote Sensing}, 
  title={Explicit High-level Semantic Network for Domain Generalization in Hyperspectral Image Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TGRS.2024.3495765}}
```

## Dataset
You can download the Pavia and Houston dataset on [here](https://github.com/YuxiangZhang-BIT/Data-CSHSI). 
As for the XS dataset, please contact the authors below:
* [The effect of artificial intelligence evolving on hyperspectral imagery with different signal-to-noise ratio, spectral and spatial resolutions](https://www.sciencedirect.com/science/article/pii/S0034425724003092). 
* [Design, Performance, and Applications of AMMIS: A Novel Airborne Multi-Modular Imaging Spectrometer for High-Resolution Earth Observations](https://www.sciencedirect.com/science/article/pii/S0034425724003092). 

The dataset directory should look like this:
```

datasets
├── Houston
│   ├── Houston13.mat
│   ├── Houston13_7gt.mat
│   ├── Houston18.mat
│   └── Houston18_7gt.mat
├── Pavia
│   ├── paviaC.mat
│   ├── paviaC_7gt.mat
│   ├── paviaU.mat
│   └── paviaU_7gt.mat
└── XS
    ├── XS_0.mat
    ├── XS_gt_0.mat
    ├── XS_1.mat
    └── XS_gt_1.mat
```
## Requirement
* CUDA Version: 11.3
* PyTorch version: 1.11.0
* Python version: 3.8.10
* You can download the the CLIP pre-training weight ViT-B-32\.pt [here](https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt).
## Usage
* For Houston dataset, you can run the `train.py` with `python train.py --dataset Houston --alpha 0.1 --beta 1e+0 --re_ratio 5`
* For Pavia dataset, you can run the `train.py` with `python train.py --dataset Pavia --alpha 0.7 --beta 1e+0 --re_ratio 1`
* For XS dataset, you can run the `train.py` with `python train.py --dataset XS --alpha 0.3 --beta 1e-1 --re_ratio 1 --training_sample_ratio 0.1 --num_epoch 100`

## Acknowledgment
 Our code is based on the method of [LDGnet](https://github.com/YuxiangZhang-BIT/IEEE_TGRS_LDGnet.git). Thanks for their work.

