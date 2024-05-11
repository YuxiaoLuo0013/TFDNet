TFDNet: Time-Frequency Enhanced Decomposed Network for Long-term Time Series Forecasting [【paper】](https://arxiv.org/pdf/2308.13386)




## Get Started

1. Install Python 3.9, PyTorch 1.12.0.

2. Download data. You can obtain all the five  benchmarks from
ETT https://github.com/zhouhaoyi/ETDataset
Electricity https://archive.ics.uci.edu/ml/datasets/ElectricityLoadDiagrams20112014
Traffic http://pems.dot.ca.gov
Weather https://www.bgc-jena.mpg.de/wetter/
Illness https://gis.cdc.gov/grasp/fluview/fluportaldashboard.html

3. Train the model. We provide the experiment scripts of all benchmarks under the folder `./scripts`. You can reproduce the experiment results by:

```bash
bash ./scripts/ETTm1.sh
bash ./scripts/ETTm2.sh
bash ./scripts/ETTh1.sh
bash ./scripts/ETTh1.sh
bash ./scripts/ECL.sh
bash ./scripts/traffic.sh
bash ./scripts/weather.sh
bash ./scripts/ILI.sh
```
4. You can also run run_longExp.py.
```

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Autoformer

https://github.com/zhouhaoyi/Informer2020

https://github.com/zhouhaoyi/ETDataset

https://github.com/laiguokun/multivariate-time-series-data

https://github.com/cure-lab/LTSFLinear

