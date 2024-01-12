---
date: '2024-01-11'
title: "🎁What is TabNet?| 📈TabNet & ensemble EN"
author: JDW
type: book
weight: 20
output: 
  rmarkdown::html_document()
editor_options: 
  markdown: 
    wrap: 255
---





<center>

 **Original Notebook** : <https://www.kaggle.com/code/stechparme/what-is-tabnet-tabnet-ensemble-en/notebook#predict-with-TabNet+Catboost+LGBM> 
 
</center>


# Introduction 

&nbsp; This is the trail notebook of pytorch_tabnet. 

> <b> In this notebook </b> <br> 1. try TabNet model prediction <br> 2. ensemble TabNet and CatBoost and LGBM (skip this process) 


# About TabNet 

 - deep tabular data learning model 
 - this model can improve accuracy by unsupervised learning 
 - the performance of this model is equal to or greater than gbdt model 
 

# import and read data 


```python
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import ConfusionMatrixDisplay 
from pytorch_tabnet.pretraining import TabNetPretrainer 
from pytorch_tabnet.tab_model import TabNetClassifier 

import torch 
# from catboost import CatBoostClassifier
# from lightgbm import LGBMClassifier 
```


```python
train = pd.read_csv("./data/train.csv") 
test  = pd.read_csv("./data/test.csv") 
sample = pd.read_csv("./data/sample_submission.csv") 

df_all = pd.concat([train, test], sort = True) \
         .reset_index(drop = True)
```



```python
print(df_all.info()) 
#> <class 'pandas.core.frame.DataFrame'>
#> RangeIndex: 275057 entries, 0 to 275056
#> Data columns (total 14 columns):
#>  #   Column           Non-Null Count   Dtype  
#> ---  ------           --------------   -----  
#>  0   Age              275057 non-null  float64
#>  1   Balance          275057 non-null  float64
#>  2   CreditScore      275057 non-null  int64  
#>  3   CustomerId       275057 non-null  int64  
#>  4   EstimatedSalary  275057 non-null  float64
#>  5   Exited           165034 non-null  float64
#>  6   Gender           275057 non-null  object 
#>  7   Geography        275057 non-null  object 
#>  8   HasCrCard        275057 non-null  float64
#>  9   IsActiveMember   275057 non-null  float64
#>  10  NumOfProducts    275057 non-null  int64  
#>  11  Surname          275057 non-null  object 
#>  12  Tenure           275057 non-null  int64  
#>  13  id               275057 non-null  int64  
#> dtypes: float64(6), int64(5), object(3)
#> memory usage: 29.4+ MB
#> None
df_all.head(2)
#>     Age  Balance  CreditScore  ...         Surname  Tenure  id
#> 0  33.0      0.0          668  ...  Okwudilichukwu       3   0
#> 1  33.0      0.0          627  ...   Okwudiliolisa       1   1
#> 
#> [2 rows x 14 columns]
```


```python
df_all.describe() 
#>                  Age        Balance  ...         Tenure             id
#> count  275057.000000  275057.000000  ...  275057.000000  275057.000000
#> mean       38.124415   55420.296450  ...       5.010867  137528.000000
#> std         8.864927   62805.933171  ...       2.806173   79402.260834
#> min        18.000000       0.000000  ...       0.000000       0.000000
#> 25%        32.000000       0.000000  ...       3.000000   68764.000000
#> 50%        37.000000       0.000000  ...       5.000000  137528.000000
#> 75%        42.000000  120037.960000  ...       7.000000  206292.000000
#> max        92.000000  250898.090000  ...      10.000000  275056.000000
#> 
#> [8 rows x 11 columns]
```


# create features 


```python
# Age group 
df_all['Age_label'] = 0 
df_all.loc[(df_all['Age'] < 20), 'Age_label'] = 1 
df_all.loc[(df_all['Age'] >= 20) & (df_all['Age'] < 30), 'Age_label'] = 2 
df_all.loc[(df_all['Age'] >= 30) & (df_all['Age'] < 40), 'Age_label'] = 3 
df_all.loc[(df_all['Age'] >= 40), 'Age_label'] = 4 
```




```python
# Balance group 
df_all['Balance_label'] = 0 
df_all.loc[(df_all['Balance'] >= 50000) & (df_all['Balance'] < 100000), 'Balance_label'] = 1 
df_all.loc[(df_all['Balance'] >= 100000) & (df_all['Balance'] < 150000), 'Balance_label'] = 2 
df_all.loc[(df_all['Balance'] >= 150000) & (df_all['Balance'] < 200000), 'Balance_label'] = 3
df_all.loc[(df_all['Balance'] >= 200000), 'Balance_label'] = 4 
```


```python
# CreditScore group 
df_all['CreditScore_label'] = 0 
df_all.loc[(df_all['CreditScore'] >= 500) & (df_all['CreditScore'] < 600), 'CreditScore_label'] = 1 
df_all.loc[(df_all['CreditScore'] >= 600) & (df_all['CreditScore'] < 700), 'CreditScore_label'] = 2 
df_all.loc[(df_all['CreditScore'] >= 700), 'CreditScore_label'] = 3 
```


```python
# EstimatedSalary group 
df_all['EstimatedSalary_label'] = 0 
df_all.loc[(df_all['EstimatedSalary'] >= 50000) & (df_all['EstimatedSalary'] < 100000), 'EstimatedSalary_label'] = 1 
df_all.loc[(df_all['EstimatedSalary'] >= 100000) & (df_all['EstimatedSalary'] < 150000), 'EstimatedSalary_label'] = 2
df_all.loc[(df_all['EstimatedSalary'] >= 150000), 'EstimatedSalary_label'] = 3 
```


```python
# feature convert 
df_all['Surname'], uniques = pd.factorize(df_all['Surname']) 
df_all = pd.get_dummies(df_all, columns = ['Gender', 'Geography', 'HasCrCard', 'IsActiveMember', 'NumOfProducts']) 
```


```python
print(df_all.shape) 
#> (275057, 26)
print(df_all.info()) 
#> <class 'pandas.core.frame.DataFrame'>
#> RangeIndex: 275057 entries, 0 to 275056
#> Data columns (total 26 columns):
#>  #   Column                 Non-Null Count   Dtype  
#> ---  ------                 --------------   -----  
#>  0   Age                    275057 non-null  float64
#>  1   Balance                275057 non-null  float64
#>  2   CreditScore            275057 non-null  int64  
#>  3   CustomerId             275057 non-null  int64  
#>  4   EstimatedSalary        275057 non-null  float64
#>  5   Exited                 165034 non-null  float64
#>  6   Surname                275057 non-null  int64  
#>  7   Tenure                 275057 non-null  int64  
#>  8   id                     275057 non-null  int64  
#>  9   Age_label              275057 non-null  int64  
#>  10  Balance_label          275057 non-null  int64  
#>  11  CreditScore_label      275057 non-null  int64  
#>  12  EstimatedSalary_label  275057 non-null  int64  
#>  13  Gender_Female          275057 non-null  bool   
#>  14  Gender_Male            275057 non-null  bool   
#>  15  Geography_France       275057 non-null  bool   
#>  16  Geography_Germany      275057 non-null  bool   
#>  17  Geography_Spain        275057 non-null  bool   
#>  18  HasCrCard_0.0          275057 non-null  bool   
#>  19  HasCrCard_1.0          275057 non-null  bool   
#>  20  IsActiveMember_0.0     275057 non-null  bool   
#>  21  IsActiveMember_1.0     275057 non-null  bool   
#>  22  NumOfProducts_1        275057 non-null  bool   
#>  23  NumOfProducts_2        275057 non-null  bool   
#>  24  NumOfProducts_3        275057 non-null  bool   
#>  25  NumOfProducts_4        275057 non-null  bool   
#> dtypes: bool(13), float64(4), int64(9)
#> memory usage: 30.7 MB
#> None
print(df_all.head())
#>     Age    Balance  ...  NumOfProducts_3  NumOfProducts_4
#> 0  33.0       0.00  ...            False            False
#> 1  33.0       0.00  ...            False            False
#> 2  40.0       0.00  ...            False            False
#> 3  34.0  148882.54  ...            False            False
#> 4  33.0       0.00  ...            False            False
#> 
#> [5 rows x 26 columns]
```


# split data for train and test 


```python
# split data into train and test 
df_all.drop(['id'], inplace = True, axis = 1) 
df_train = df_all.loc[df_all['Exited'].notnull()] 
df_test  = df_all.loc[df_all['Exited'].isnull()] 
df_test.drop(['Exited'], inplace = True, axis = 1)
#> <string>:1: SettingWithCopyWarning: 
#> A value is trying to be set on a copy of a slice from a DataFrame
#> 
#> See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
```



```python
# convert data for using TabNet 
X = df_train.drop(['Exited'], axis = 1) \
    .values \
    .astype(float) 
y = df_train['Exited'] \
    .values \
    .astype(int)
df_test = df_test \
          .values \
          .astype(int)

# split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1, stratify = y)
```


# model train 


```python
# pretrain with unspervised model 
unsupervised_model = TabNetPretrainer(optimizer_fn = torch.optim.Adam, 
                                      optimizer_params = dict(lr = 1e-3), 
                                      device_name = "cuda", 
                                      mask_type = "entmax") 
#> /home/rstudio/.local/share/r-miniconda/envs/r-reticulate/lib/python3.10/site-packages/pytorch_tabnet/abstract_model.py:82: UserWarning: Device used : cuda
#>   warnings.warn(f"Device used : {self.device}")

unsupervised_model.fit(X_train = X_train, 
                       eval_set = [X_test], 
                       batch_size = 64, 
                       max_epochs = 100, 
                       patience = 10, 
                       pretraining_ratio = 0.8)
#> epoch 0  | loss: 2050.35993| val_0_unsup_loss_numpy: 2011.8619384765625|  0:00:30s
#> epoch 1  | loss: 2042.20971| val_0_unsup_loss_numpy: 2010.4073486328125|  0:01:01s
#> epoch 2  | loss: 2044.25991| val_0_unsup_loss_numpy: 2010.07275390625|  0:01:31s
#> epoch 3  | loss: 2041.11629| val_0_unsup_loss_numpy: 2009.5718994140625|  0:02:01s
#> epoch 4  | loss: 2044.52132| val_0_unsup_loss_numpy: 2009.6256103515625|  0:02:31s
#> epoch 5  | loss: 2039.13427| val_0_unsup_loss_numpy: 2009.303955078125|  0:03:09s
#> epoch 6  | loss: 2046.92803| val_0_unsup_loss_numpy: 2009.201171875|  0:03:43s
#> epoch 7  | loss: 2039.25961| val_0_unsup_loss_numpy: 2008.898193359375|  0:04:14s
#> epoch 8  | loss: 2042.7654| val_0_unsup_loss_numpy: 2010.17333984375|  0:04:44s
#> epoch 9  | loss: 2042.94586| val_0_unsup_loss_numpy: 2009.300537109375|  0:05:15s
#> epoch 10 | loss: 2043.9666| val_0_unsup_loss_numpy: 2009.748046875|  0:05:44s
#> epoch 11 | loss: 2043.60059| val_0_unsup_loss_numpy: 2009.5777587890625|  0:06:14s
#> epoch 12 | loss: 2040.07033| val_0_unsup_loss_numpy: 2008.737548828125|  0:06:44s
#> epoch 13 | loss: 2044.81842| val_0_unsup_loss_numpy: 2008.8499755859375|  0:07:13s
#> epoch 14 | loss: 2040.84951| val_0_unsup_loss_numpy: 2008.99951171875|  0:07:43s
#> epoch 15 | loss: 2045.44028| val_0_unsup_loss_numpy: 2008.0086669921875|  0:08:16s
#> epoch 16 | loss: 2045.66059| val_0_unsup_loss_numpy: 2009.361572265625|  0:08:46s
#> epoch 17 | loss: 2041.97707| val_0_unsup_loss_numpy: 2008.5216064453125|  0:09:18s
#> epoch 18 | loss: 2035.32246| val_0_unsup_loss_numpy: 2009.1259765625|  0:09:51s
#> epoch 19 | loss: 2041.81241| val_0_unsup_loss_numpy: 2009.13525390625|  0:10:22s
#> epoch 20 | loss: 2033.32754| val_0_unsup_loss_numpy: 2008.73291015625|  0:10:53s
#> epoch 21 | loss: 2041.27862| val_0_unsup_loss_numpy: 2008.46484375|  0:11:21s
#> epoch 22 | loss: 2039.38183| val_0_unsup_loss_numpy: 2009.5911865234375|  0:11:51s
#> epoch 23 | loss: 2039.49226| val_0_unsup_loss_numpy: 2006.888427734375|  0:12:20s
#> epoch 24 | loss: 2044.64617| val_0_unsup_loss_numpy: 2008.709716796875|  0:12:51s
#> epoch 25 | loss: 2038.77137| val_0_unsup_loss_numpy: 2010.7569580078125|  0:13:23s
#> epoch 26 | loss: 2034.80459| val_0_unsup_loss_numpy: 2008.857421875|  0:13:53s
#> epoch 27 | loss: 2038.67015| val_0_unsup_loss_numpy: 2008.715576171875|  0:14:21s
#> epoch 28 | loss: 2034.56014| val_0_unsup_loss_numpy: 2007.0107421875|  0:14:50s
#> epoch 29 | loss: 2044.26342| val_0_unsup_loss_numpy: 2007.596435546875|  0:15:21s
#> epoch 30 | loss: 2036.45683| val_0_unsup_loss_numpy: 2007.323974609375|  0:15:52s
#> epoch 31 | loss: 2040.55116| val_0_unsup_loss_numpy: 2007.0291748046875|  0:16:22s
#> epoch 32 | loss: 2032.43736| val_0_unsup_loss_numpy: 2007.6968994140625|  0:16:53s
#> epoch 33 | loss: 2033.66415| val_0_unsup_loss_numpy: 2007.7215576171875|  0:17:21s
#> 
#> Early stopping occurred at epoch 33 with best_epoch = 23 and best_val_0_unsup_loss_numpy = 2006.888427734375
#> 
#> /home/rstudio/.local/share/r-miniconda/envs/r-reticulate/lib/python3.10/site-packages/pytorch_tabnet/callbacks.py:172: UserWarning: Best weights from best epoch are automatically used!
#>   warnings.warn(wrn_msg)
```



```python
# train model 
model = TabNetClassifier(optimizer_fn = torch.optim.Adam, 
                         optimizer_params = dict(lr = 1e-3), 
                         device_name = "cuda", 
                         verbose = 1) 
#> /home/rstudio/.local/share/r-miniconda/envs/r-reticulate/lib/python3.10/site-packages/pytorch_tabnet/abstract_model.py:82: UserWarning: Device used : cuda
#>   warnings.warn(f"Device used : {self.device}")

model.fit(X_train, y_train, 
          eval_set = [(X_train, y_train), (X_test, y_test)], 
          eval_metric = ['accuracy', 'accuracy'], 
          eval_name = ['train', 'valid'], 
          batch_size = 64, 
          max_epochs = 100, 
          patience = 10, 
          from_unsupervised = unsupervised_model) 
#> epoch 0  | loss: 0.42906 | train_accuracy: 0.85574 | valid_accuracy: 0.85179 |  0:00:33s
#> epoch 1  | loss: 0.34494 | train_accuracy: 0.85845 | valid_accuracy: 0.85315 |  0:01:08s
#> epoch 2  | loss: 0.33713 | train_accuracy: 0.86165 | valid_accuracy: 0.85676 |  0:01:45s
#> epoch 3  | loss: 0.33414 | train_accuracy: 0.86343 | valid_accuracy: 0.85888 |  0:02:20s
#> epoch 4  | loss: 0.33183 | train_accuracy: 0.86333 | valid_accuracy: 0.85939 |  0:02:54s
#> epoch 5  | loss: 0.33078 | train_accuracy: 0.86269 | valid_accuracy: 0.85842 |  0:03:29s
#> epoch 6  | loss: 0.32883 | train_accuracy: 0.86407 | valid_accuracy: 0.8587  |  0:04:08s
#> epoch 7  | loss: 0.32811 | train_accuracy: 0.865   | valid_accuracy: 0.86012 |  0:04:45s
#> epoch 8  | loss: 0.32789 | train_accuracy: 0.86429 | valid_accuracy: 0.85964 |  0:05:21s
#> epoch 9  | loss: 0.32723 | train_accuracy: 0.865   | valid_accuracy: 0.86094 |  0:05:56s
#> epoch 10 | loss: 0.32707 | train_accuracy: 0.8652  | valid_accuracy: 0.86188 |  0:06:31s
#> epoch 11 | loss: 0.32597 | train_accuracy: 0.86601 | valid_accuracy: 0.86157 |  0:07:05s
#> epoch 12 | loss: 0.32566 | train_accuracy: 0.86537 | valid_accuracy: 0.86112 |  0:07:39s
#> epoch 13 | loss: 0.32567 | train_accuracy: 0.86355 | valid_accuracy: 0.85867 |  0:08:13s
#> epoch 14 | loss: 0.32541 | train_accuracy: 0.86487 | valid_accuracy: 0.86067 |  0:08:47s
#> epoch 15 | loss: 0.32441 | train_accuracy: 0.86504 | valid_accuracy: 0.86094 |  0:09:21s
#> epoch 16 | loss: 0.32435 | train_accuracy: 0.86509 | valid_accuracy: 0.86045 |  0:09:54s
#> epoch 17 | loss: 0.32388 | train_accuracy: 0.86396 | valid_accuracy: 0.85909 |  0:10:28s
#> epoch 18 | loss: 0.32387 | train_accuracy: 0.8656  | valid_accuracy: 0.86179 |  0:11:02s
#> epoch 19 | loss: 0.32399 | train_accuracy: 0.866   | valid_accuracy: 0.86064 |  0:11:39s
#> epoch 20 | loss: 0.32397 | train_accuracy: 0.86457 | valid_accuracy: 0.86064 |  0:12:14s
#> 
#> Early stopping occurred at epoch 20 with best_epoch = 10 and best_valid_accuracy = 0.86188
#> 
#> /home/rstudio/.local/share/r-miniconda/envs/r-reticulate/lib/python3.10/site-packages/pytorch_tabnet/abstract_model.py:118: UserWarning: Pretraining: mask_type changed from sparsemax to entmax
#>   warnings.warn(wrn_msg)
#> /home/rstudio/.local/share/r-miniconda/envs/r-reticulate/lib/python3.10/site-packages/pytorch_tabnet/abstract_model.py:248: UserWarning: Loading weights from unsupervised pretraining
#>   warnings.warn("Loading weights from unsupervised pretraining")
#> /home/rstudio/.local/share/r-miniconda/envs/r-reticulate/lib/python3.10/site-packages/pytorch_tabnet/callbacks.py:172: UserWarning: Best weights from best epoch are automatically used!
#>   warnings.warn(wrn_msg)
```


# view feature interpretation and predict probability 


```python
# mask(local interpretability) 
explain_matrix, masks = model.explain(df_test) 
fig, ax = plt.subplots(1, 3, figsize = (10, 7)) 

for i in range(3): 
  ax[i].imshow(masks[i][:25]) 
  ax[i].set_title(f"mask {i}") 
plt.show()
```

<img src="/courses/rolling-in-the-kaggle/Binary_Classification_with_a_Bank_Churn_Dataset/What_is_TabNet_files/figure-html/unnamed-chunk-17-1.png" width="960" />


```python
pred_tabnet = model.predict_proba(df_test)[:, 1] 
output_tabnet = pd.DataFrame({'id': sample['id'], 'Exited' : pred_tabnet}) 
output_tabnet.to_csv('./data/submission_tabnet.csv', index = False) 

output_tabnet.head() 
#>        id    Exited
#> 0  165034  0.018964
#> 1  165035  0.793878
#> 2  165036  0.019698
#> 3  165037  0.218071
#> 4  165038  0.357962
```

# predict with TabNet + Catboost + LGBM 

skip


























