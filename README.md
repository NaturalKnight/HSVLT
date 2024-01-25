# HSVLT
Code for ACM MM 2023 paper 'HSVLT: Hierarchical Scale-Aware Vision-Language Transformer for Multi-Label Image Classification'


## Data Preparation
1. Download dataset and organize them as follow:
```
|datasets
|---- MSCOCO
|---- NUS-WIDE
|---- VOC2007
```
2. Preprocess using following commands:
```bash
python scripts/mscoco.py
python scripts/nuswide.py
python scripts/voc2007.py
python embedding.py --data [mscoco, nuswide, voc2007]
```

## Requirements
```
torch >= 1.9.0
torchvision >= 0.10.0
```

## Reference

- [TSFormer](https://github.com/jasonseu/TSFormer)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
