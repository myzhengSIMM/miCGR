# miCGR: Interpretable deep neural network for predicting both site-level and gene-level functional targets of microRNA

## System requirements
- The codes have been test on Linux platform of Ubuntu 20.04 LTS.
- The codes were implemented in python==3.8 and pytorch==1.9.1.
- The miCGR model was on GPU-mode, rewrite device = torch.device('cuda:0') with device = torch.device('cpu') would turn it into cpu-mode.
- Instructions

```python
git clone https://e.coding.net/colinwxl/micgr/MICGR.git
cd MICGR

python predict_balanced.py
python predict_imbalanced.py

```

### For case studies
```
python predict.py xx/to_predict.txt gpu_num offset-9-mer-m7
```

## License
This code is licensed under the Apache 2.0 License.
