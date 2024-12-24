
# Install Packages

Download data following this structure:
```bash

```

# Usage

# BankMarketing dataset
LightGBM method:

`python src/bankMarketing/1-lightgbm.py`

RRL method:

```bash
python src/bankMarketing/2-rrl.py
cd rrl
python experiment-bankMarketing.py -d bankMarketing (hyperparameters)
```

# BreastCancer dataset
LightGBM method:

`python src/breastCancer/1-lightgbm.py`

RRL method:

```bash
python src/breastCancer/2-rrl.py
cd rrl
python experiment-breastCancer.py -d breastCancer (hyperparameters)
```




# 说明

## 总体说明

- 大部分数据集主要评估指标为Macro F1-score
- breastCancer数据集关注指标Recall指标（假负例代价高）

## RRL方法

- 修改了GitHub中RRL代码仓库的数个错误
- 对于BostonHousing数据集，修改rrl代码(修改内容见commit)，使用默认的5-fold cross validation
- 对于breastCancer数据集，不使用k-fold cross validation，直接按指定的数据集划分
- 对于bankMarketing数据集，使用默认的5-fold cross validation
- 按照GitHub仓库README的Tuning Suggestions进行调参
- 不同数据集、不同超参数的实验结果在`rrl/log_folder`文件夹下
- `rrl/log_folder`文件夹下的文件名形如events.out.xxx的文件为tensorboard数据文件，使用tensorboard画图
- `rrl/log_folder`文件夹下的`rrl.txt`为输出规则，在报告中加上
- 形如`rrl/log_folder/breastCancer/breastCancer_e401_bs32_lr0.002_lrdr0.75_lrde200_wd0.0001_ki0_rc0_useNOTFalse_saveBestFalse_useNLAFFalse_estimatedGradFalse_useSkipFalse_alpha0.999_beta8_gamma1_temp1.0_L1@16`的文件夹名称记录了超参数，超参数的具体含义见RRL GitHub仓库README

## LightGBM方法
- 对lgb.LGBMClassifier进行调参
- 不同数据集、不同超参数的实验结果在`results/(数据集名称)`文件夹下