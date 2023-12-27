---
date: '2023-12-02'
title: spaceship_titanic🛸
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

 **Original Notebook** : <https://www.kaggle.com/code/parthavjoshi/spaceship-titanic> 
 
</center>


```python
import os 
import numpy as np 
import pandas as pd 
import gc

import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import Dataset, DataLoader 

from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler 
from sklearn.compose import ColumnTransformer 
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import StratifiedKFold, cross_val_score 
from sklearn.metrics import accuracy_score, roc_curve,auc, confusion_matrix,precision_recall_curve,precision_recall_curve

import tqdm

import warnings 
warnings.simplefilter('ignore') 
```

# Load data


```python
train = pd.read_csv(os.path.join(os.getcwd(), "data", "train.csv"))
train.head(2)
#>   PassengerId HomePlanet CryoSleep  ... VRDeck             Name  Transported
#> 0     0001_01     Europa     False  ...    0.0  Maham Ofracculy        False
#> 1     0002_01      Earth     False  ...   44.0     Juanna Vines         True
#> 
#> [2 rows x 14 columns]
train.shape
#> (8693, 14)

test  = pd.read_csv(os.path.join(os.getcwd(), "data", "test.csv"))
test.head(2) 
#>   PassengerId HomePlanet CryoSleep  ...     Spa VRDeck             Name
#> 0     0013_01      Earth      True  ...     0.0    0.0  Nelly Carsoning
#> 1     0018_01      Earth     False  ...  2823.0    0.0   Lerome Peckers
#> 
#> [2 rows x 13 columns]
test.shape
#> (4277, 13)
```

# Exploring the data


```python
train.describe() 
#>                Age   RoomService  ...           Spa        VRDeck
#> count  8514.000000   8512.000000  ...   8510.000000   8505.000000
#> mean     28.827930    224.687617  ...    311.138778    304.854791
#> std      14.489021    666.717663  ...   1136.705535   1145.717189
#> min       0.000000      0.000000  ...      0.000000      0.000000
#> 25%      19.000000      0.000000  ...      0.000000      0.000000
#> 50%      27.000000      0.000000  ...      0.000000      0.000000
#> 75%      38.000000     47.000000  ...     59.000000     46.000000
#> max      79.000000  14327.000000  ...  22408.000000  24133.000000
#> 
#> [8 rows x 6 columns]

train.dtypes 
#> PassengerId      object
#> HomePlanet       object
#> CryoSleep        object
#> Cabin            object
#> Destination      object
#> Age             float64
#> VIP              object
#> RoomService     float64
#> FoodCourt       float64
#> ShoppingMall    float64
#> Spa             float64
#> VRDeck          float64
#> Name             object
#> Transported        bool
#> dtype: object

train.nunique() 
#> PassengerId     8693
#> HomePlanet         3
#> CryoSleep          2
#> Cabin           6560
#> Destination        3
#> Age               80
#> VIP                2
#> RoomService     1273
#> FoodCourt       1507
#> ShoppingMall    1115
#> Spa             1327
#> VRDeck          1306
#> Name            8473
#> Transported        2
#> dtype: int64

train.isna().sum()
#> PassengerId       0
#> HomePlanet      201
#> CryoSleep       217
#> Cabin           199
#> Destination     182
#> Age             179
#> VIP             203
#> RoomService     181
#> FoodCourt       183
#> ShoppingMall    208
#> Spa             183
#> VRDeck          188
#> Name            200
#> Transported       0
#> dtype: int64

test.isna().sum()
#> PassengerId       0
#> HomePlanet       87
#> CryoSleep        93
#> Cabin           100
#> Destination      92
#> Age              91
#> VIP              93
#> RoomService      82
#> FoodCourt       106
#> ShoppingMall     98
#> Spa             101
#> VRDeck           80
#> Name             94
#> dtype: int64
```

# Data Preprocessing 

```python
train['is_train'] = True 
test['is_train'] = False

data = pd.concat([train, test]) 
data.head() 
#>   PassengerId HomePlanet CryoSleep  ...               Name Transported  is_train
#> 0     0001_01     Europa     False  ...    Maham Ofracculy       False      True
#> 1     0002_01      Earth     False  ...       Juanna Vines        True      True
#> 2     0003_01     Europa     False  ...      Altark Susent       False      True
#> 3     0003_02     Europa     False  ...       Solam Susent       False      True
#> 4     0004_01      Earth     False  ...  Willy Santantines        True      True
#> 
#> [5 rows x 15 columns]
```


```python
def fill_missing_value(data, cols): 
  for c in cols: 
    data[c].fillna(data[c].median(skipna = True), inplace = True) 

fill_missing_value(data, ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']) 

data['HomePlanet'].fillna('Z', inplace = True) 
```


```python
def label_encode(data, col): 
  data[col] = data[col].astype(str) 
  data[col] = LabelEncoder().fit_transform(data[col]) 
  return data[col] 

data['HomePlanet']  = label_encode(data, 'HomePlanet') 
data['CryoSleep']   = label_encode(data, 'CryoSleep') 
data['VIP']         = label_encode(data, 'VIP') 
data['Destination'] = label_encode(data, 'Destination')
```


```python
mask = data['is_train'] == True 
mask
#> 0        True
#> 1        True
#> 2        True
#> 3        True
#> 4        True
#>         ...  
#> 4272    False
#> 4273    False
#> 4274    False
#> 4275    False
#> 4276    False
#> Name: is_train, Length: 12970, dtype: bool
```


```python
train = data[mask] 
train.shape 
#> (8693, 15)
```


```python
test = data[~mask] 
test.shape 
#> (4277, 15)
```


```python
train_data = train.drop(['is_train'], axis = 1) 
test_data  = test.drop(['is_train'], axis = 1) 
```


```python
train_data.isna().sum() 
#> PassengerId       0
#> HomePlanet        0
#> CryoSleep         0
#> Cabin           199
#> Destination       0
#> Age               0
#> VIP               0
#> RoomService       0
#> FoodCourt         0
#> ShoppingMall      0
#> Spa               0
#> VRDeck            0
#> Name            200
#> Transported       0
#> dtype: int64

test_data.isna().sum() 
#> PassengerId        0
#> HomePlanet         0
#> CryoSleep          0
#> Cabin            100
#> Destination        0
#> Age                0
#> VIP                0
#> RoomService        0
#> FoodCourt          0
#> ShoppingMall       0
#> Spa                0
#> VRDeck             0
#> Name              94
#> Transported     4277
#> dtype: int64
```


# Model Training 


```python
train_data = train_data.dropna() 
train_data.drop(['PassengerId', 'Cabin', 'Name'], axis = 1, inplace = True) 
test_data.drop(['PassengerId', 'Cabin', 'Name'], axis = 1, inplace = True) 
train_data['Transported'] = train_data['Transported'].map({True : 1, False : 0}) 
```


```python
class Config: 
  lr = 1e-4 
  nb_epochs = 100 
  train_bs = 64 
  valid_bs = 64 
  train_split = 0.8 
  k_folds = 5 
  device = "cuda" 
  train_loss_fn = nn.BCEWithLogitsLoss() 
  valid_loss_fn = nn.BCEWithLogitsLoss() 
  feature_names = [
    "HomePlanet", "CryoSleep", "Destination", "Age", "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"
  ] 
  target_name = "Transported"
```



```python
class SpaceshipTitanicModel(nn.Module): 
  def __init__(self, input_size = None, output_size = None): 
    super().__init__() 
    self.input_size = 10 if not input_size else input_size 
    self.output_size = 1 if not output_size else output_size 
    
    # model architecture 
    self.fc1 = nn.Linear(self.input_size, 1024) 
    self.fc2 = nn.Linear(1024, 768) 
    self.fc3 = nn.Linear(768, 128) 
    self.fc4 = nn.Linear(128, self.output_size) 
    self.relu = nn.ReLU() 
    self.sig = nn.Sigmoid() 
    
  def forward(self, x): 
    out = self.fc1(x) 
    out = self.relu(out) 
    out = self.fc2(out) 
    out = self.relu(out) 
    out = self.fc3(out) 
    out = self.relu(out) 
    out = self.fc4(out) 
    out = self.sig(out) 
    
    return out 
  
  
def binary_acc(y_pred, y_test): 
  y_pred = torch.round(torch.sigmoid(y_pred)) 
  
  correct = (y_pred == y_test).sum().float() 
  acc = correct / y_test.shape[0] 
  acc = torch.round(acc * 100) 
  
  return acc 
```


```python
class SpaceshipTitanicData(Dataset): 
  def __init__(self, features, target, is_test = False): 
    self.features = features 
    self.target = target 
    self.is_test = is_test 
    
  def __getitem__(self, idx): 
    data = self.features.values[idx] 
    if self.is_test: 
      return torch.tensor(data, dtype = torch.float32) 
    else: 
      target = self.target.values[idx] 
      return torch.tensor(data, dtype = torch.float32), torch.tensor(target, dtype = torch.float32) 
      
  def __len__(self): 
    return len(self.features) 
```


```python
def train_model(model, train_loader, optimizer, loss_fn, device): 
  """
  Training Function 
  """
  print("Training............") 
  # breakpoint()
  with HiddenPrints():
    model.train() 
  global y 
  global z 
  running_loss = 0 
  all_targets = [] 
  all_preds = []
  
  with HiddenPrints(): 
    prog_bar = tqdm.tqdm(train_loader, total = len(train_loader), disable=True) 
  for x, y in prog_bar: 
    x = x.to(device, torch.float32) 
    y = y.to(device, torch.float32) 
    
    z = model(x) 
    train_loss = loss_fn(z, y) 
    acc = binary_acc(z, y) 
    train_loss.backward() 
    
    optimizer.step() 
    optimizer.zero_grad() 
    
    running_loss = running_loss + train_loss 
    prog_bar.set_description(f"train loss : {train_loss.item():.2f}") 
    
    all_targets.append(y.detach().cpu().numpy()) 
    all_preds.append(z.detach().cpu().numpy()) 
    
  return all_targets, all_preds 

def valid_fn(model, tvalid_loader, loss_fn, device): 
  """
  Validation function 
  """
  with HiddenPrints(): 
    model.eval() 
  running_loss = 0 
  all_targets = []
  all_preds = [] 
  with HiddenPrints(): 
    prog_bar = tqdm.tqdm(valid_loader, total = len(valid_loader), disable=True) 
  for x, y in prog_bar: 
    x = x.to(device, torch.float32) 
    y = y.to(device, torch.float32) 
    z = model(x) 
    
    valid_loss = loss_fn(z, y) 
    acc = binary_acc(z, y) 
    running_loss = running_loss + valid_loss 
    prog_bar.set_description(f"Validation loss{valid_loss.item():.2f}") 
    all_targets.append(y.detach().cpu().numpy()) 
    all_preds.append(z.detach().cpu().numpy()) 
    
  print(f"Validation Loss: {running_loss:.4f}") 
  print(f"Acc : {acc:.3f}") 
  return all_targets, all_preds
```




```python
if __name__ == "__main__": 
  data = train_data.sample(frac = 1).reset_index(drop = True) 
  
  kfold = StratifiedKFold(n_splits = Config.k_folds, shuffle = True) 
  for fold, (train_ids, valid_ids) in enumerate(kfold.split(data.drop(['Transported'], axis = 1), data['Transported'])): 
    print(f"FOLD {fold}") 
    print("-"*20) 
    
    train_ = data.iloc[train_ids] 
    valid_ = data.iloc[valid_ids] 
    
    train_dataset = SpaceshipTitanicData(
      features = train_.drop(['Transported'], axis = 1), target = train_[['Transported']]
    ) 
    
    valid_dataset = SpaceshipTitanicData( 
      features = valid_.drop(['Transported'], axis = 1), target = valid_[['Transported']]   
    )
    
    train_loader = DataLoader(train_dataset, batch_size = 32, shuffle = True) 
    valid_loader = DataLoader(valid_dataset, batch_size = 32, shuffle = True) 
    
    # model = SpaceshipTitanicModel(None, None) 
    with HiddenPrints(): 
      model = SpaceshipTitanicModel(10, output_size = 1) 
      model.to(Config.device) 
    
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr = Config.lr) 
    
    print("[INFO]: Starting training! \n") 
    for epoch in range(1, Config.nb_epochs + 2): 
      print(f"{'='*20} Epoch: {epoch}/{Config.nb_epochs + 1} {'='*20}") 
      # breakpoint()
      _, _ = train_model(model, train_loader, optimizer, Config.train_loss_fn, device = Config.device) 
      val_targets, val_preds = valid_fn(model, valid_loader, Config.valid_loss_fn, device = Config.device) 
      
      filepath = f"./model/fold_{fold}_model.pth" 
      torch.save(model.state_dict(), filepath)
```

```
#> FOLD 0
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/101 ====================
#> Training............
#> Validation Loss: 32.9721
#> Acc : 64.000
#> ==================== Epoch: 2/101 ====================
#> Training............
#> Validation Loss: 32.9451
#> Acc : 68.000
#> ==================== Epoch: 3/101 ====================
#> Training............
#> Validation Loss: 32.7616
#> Acc : 61.000
#> ==================== Epoch: 4/101 ====================
#> Training............
#> Validation Loss: 33.0184
#> Acc : 46.000
#> ==================== Epoch: 5/101 ====================
#> Training............
#> Validation Loss: 32.9175
#> Acc : 61.000
#> ==================== Epoch: 6/101 ====================
#> Training............
#> Validation Loss: 32.9008
#> Acc : 75.000
#> ==================== Epoch: 7/101 ====================
#> Training............
#> Validation Loss: 32.7833
#> Acc : 68.000
#> ==================== Epoch: 8/101 ====================
#> Training............
#> Validation Loss: 32.6718
#> Acc : 64.000
#> ==================== Epoch: 9/101 ====================
#> Training............
#> Validation Loss: 32.7567
#> Acc : 68.000
#> ==================== Epoch: 10/101 ====================
#> Training............
#> Validation Loss: 32.8100
#> Acc : 61.000
#> ==================== Epoch: 11/101 ====================
#> Training............
#> Validation Loss: 32.6540
#> Acc : 86.000
#> ==================== Epoch: 12/101 ====================
#> Training............
#> Validation Loss: 32.6463
#> Acc : 68.000
#> ==================== Epoch: 13/101 ====================
#> Training............
#> Validation Loss: 32.6278
#> Acc : 79.000
#> ==================== Epoch: 14/101 ====================
#> Training............
#> Validation Loss: 32.6276
#> Acc : 96.000
#> ==================== Epoch: 15/101 ====================
#> Training............
#> Validation Loss: 32.6208
#> Acc : 68.000
#> ==================== Epoch: 16/101 ====================
#> Training............
#> Validation Loss: 32.6880
#> Acc : 61.000
#> ==================== Epoch: 17/101 ====================
#> Training............
#> Validation Loss: 32.6252
#> Acc : 71.000
#> ==================== Epoch: 18/101 ====================
#> Training............
#> Validation Loss: 32.6143
#> Acc : 75.000
#> ==================== Epoch: 19/101 ====================
#> Training............
#> Validation Loss: 32.7284
#> Acc : 86.000
#> ==================== Epoch: 20/101 ====================
#> Training............
#> Validation Loss: 32.6267
#> Acc : 79.000
#> ==================== Epoch: 21/101 ====================
#> Training............
#> Validation Loss: 32.6392
#> Acc : 71.000
#> ==================== Epoch: 22/101 ====================
#> Training............
#> Validation Loss: 32.6944
#> Acc : 71.000
#> ==================== Epoch: 23/101 ====================
#> Training............
#> Validation Loss: 32.6182
#> Acc : 75.000
#> ==================== Epoch: 24/101 ====================
#> Training............
#> Validation Loss: 32.6009
#> Acc : 86.000
#> ==================== Epoch: 25/101 ====================
#> Training............
#> Validation Loss: 32.6116
#> Acc : 71.000
#> ==================== Epoch: 26/101 ====================
#> Training............
#> Validation Loss: 32.5519
#> Acc : 82.000
#> ==================== Epoch: 27/101 ====================
#> Training............
#> Validation Loss: 32.7343
#> Acc : 54.000
#> ==================== Epoch: 28/101 ====================
#> Training............
#> Validation Loss: 32.6637
#> Acc : 71.000
#> ==================== Epoch: 29/101 ====================
#> Training............
#> Validation Loss: 32.6478
#> Acc : 68.000
#> ==================== Epoch: 30/101 ====================
#> Training............
#> Validation Loss: 32.6703
#> Acc : 86.000
#> ==================== Epoch: 31/101 ====================
#> Training............
#> Validation Loss: 32.5538
#> Acc : 68.000
#> ==================== Epoch: 32/101 ====================
#> Training............
#> Validation Loss: 32.5788
#> Acc : 54.000
#> ==================== Epoch: 33/101 ====================
#> Training............
#> Validation Loss: 32.6384
#> Acc : 71.000
#> ==================== Epoch: 34/101 ====================
#> Training............
#> Validation Loss: 32.5269
#> Acc : 71.000
#> ==================== Epoch: 35/101 ====================
#> Training............
#> Validation Loss: 32.6504
#> Acc : 71.000
#> ==================== Epoch: 36/101 ====================
#> Training............
#> Validation Loss: 32.6172
#> Acc : 79.000
#> ==================== Epoch: 37/101 ====================
#> Training............
#> Validation Loss: 32.5784
#> Acc : 61.000
#> ==================== Epoch: 38/101 ====================
#> Training............
#> Validation Loss: 32.6027
#> Acc : 75.000
#> ==================== Epoch: 39/101 ====================
#> Training............
#> Validation Loss: 32.6837
#> Acc : 68.000
#> ==================== Epoch: 40/101 ====================
#> Training............
#> Validation Loss: 32.6208
#> Acc : 68.000
#> ==================== Epoch: 41/101 ====================
#> Training............
#> Validation Loss: 32.6247
#> Acc : 64.000
#> ==================== Epoch: 42/101 ====================
#> Training............
#> Validation Loss: 32.5238
#> Acc : 75.000
#> ==================== Epoch: 43/101 ====================
#> Training............
#> Validation Loss: 32.4938
#> Acc : 75.000
#> ==================== Epoch: 44/101 ====================
#> Training............
#> Validation Loss: 32.5215
#> Acc : 61.000
#> ==================== Epoch: 45/101 ====================
#> Training............
#> Validation Loss: 32.6543
#> Acc : 64.000
#> ==================== Epoch: 46/101 ====================
#> Training............
#> Validation Loss: 32.5570
#> Acc : 82.000
#> ==================== Epoch: 47/101 ====================
#> Training............
#> Validation Loss: 32.5015
#> Acc : 36.000
#> ==================== Epoch: 48/101 ====================
#> Training............
#> Validation Loss: 32.6139
#> Acc : 79.000
#> ==================== Epoch: 49/101 ====================
#> Training............
#> Validation Loss: 32.5770
#> Acc : 79.000
#> ==================== Epoch: 50/101 ====================
#> Training............
#> Validation Loss: 32.4856
#> Acc : 79.000
#> ==================== Epoch: 51/101 ====================
#> Training............
#> Validation Loss: 32.6654
#> Acc : 64.000
#> ==================== Epoch: 52/101 ====================
#> Training............
#> Validation Loss: 32.5877
#> Acc : 71.000
#> ==================== Epoch: 53/101 ====================
#> Training............
#> Validation Loss: 32.6560
#> Acc : 50.000
#> ==================== Epoch: 54/101 ====================
#> Training............
#> Validation Loss: 32.5226
#> Acc : 75.000
#> ==================== Epoch: 55/101 ====================
#> Training............
#> Validation Loss: 32.5638
#> Acc : 79.000
#> ==================== Epoch: 56/101 ====================
#> Training............
#> Validation Loss: 32.6685
#> Acc : 82.000
#> ==================== Epoch: 57/101 ====================
#> Training............
#> Validation Loss: 32.5034
#> Acc : 71.000
#> ==================== Epoch: 58/101 ====================
#> Training............
#> Validation Loss: 32.4747
#> Acc : 64.000
#> ==================== Epoch: 59/101 ====================
#> Training............
#> Validation Loss: 32.5946
#> Acc : 71.000
#> ==================== Epoch: 60/101 ====================
#> Training............
#> Validation Loss: 32.6357
#> Acc : 75.000
#> ==================== Epoch: 61/101 ====================
#> Training............
#> Validation Loss: 32.5137
#> Acc : 71.000
#> ==================== Epoch: 62/101 ====================
#> Training............
#> Validation Loss: 32.6340
#> Acc : 54.000
#> ==================== Epoch: 63/101 ====================
#> Training............
#> Validation Loss: 32.4892
#> Acc : 64.000
#> ==================== Epoch: 64/101 ====================
#> Training............
#> Validation Loss: 32.5161
#> Acc : 75.000
#> ==================== Epoch: 65/101 ====================
#> Training............
#> Validation Loss: 32.4710
#> Acc : 75.000
#> ==================== Epoch: 66/101 ====================
#> Training............
#> Validation Loss: 32.4833
#> Acc : 71.000
#> ==================== Epoch: 67/101 ====================
#> Training............
#> Validation Loss: 32.4834
#> Acc : 68.000
#> ==================== Epoch: 68/101 ====================
#> Training............
#> Validation Loss: 32.4528
#> Acc : 75.000
#> ==================== Epoch: 69/101 ====================
#> Training............
#> Validation Loss: 32.4887
#> Acc : 64.000
#> ==================== Epoch: 70/101 ====================
#> Training............
#> Validation Loss: 32.4838
#> Acc : 79.000
#> ==================== Epoch: 71/101 ====================
#> Training............
#> Validation Loss: 32.5077
#> Acc : 71.000
#> ==================== Epoch: 72/101 ====================
#> Training............
#> Validation Loss: 32.4879
#> Acc : 82.000
#> ==================== Epoch: 73/101 ====================
#> Training............
#> Validation Loss: 32.4515
#> Acc : 86.000
#> ==================== Epoch: 74/101 ====================
#> Training............
#> Validation Loss: 32.4931
#> Acc : 71.000
#> ==================== Epoch: 75/101 ====================
#> Training............
#> Validation Loss: 32.4663
#> Acc : 68.000
#> ==================== Epoch: 76/101 ====================
#> Training............
#> Validation Loss: 32.4320
#> Acc : 64.000
#> ==================== Epoch: 77/101 ====================
#> Training............
#> Validation Loss: 32.4683
#> Acc : 61.000
#> ==================== Epoch: 78/101 ====================
#> Training............
#> Validation Loss: 32.4915
#> Acc : 68.000
#> ==================== Epoch: 79/101 ====================
#> Training............
#> Validation Loss: 32.4562
#> Acc : 71.000
#> ==================== Epoch: 80/101 ====================
#> Training............
#> Validation Loss: 32.4750
#> Acc : 61.000
#> ==================== Epoch: 81/101 ====================
#> Training............
#> Validation Loss: 32.4483
#> Acc : 68.000
#> ==================== Epoch: 82/101 ====================
#> Training............
#> Validation Loss: 32.4506
#> Acc : 79.000
#> ==================== Epoch: 83/101 ====================
#> Training............
#> Validation Loss: 32.5008
#> Acc : 71.000
#> ==================== Epoch: 84/101 ====================
#> Training............
#> Validation Loss: 32.5901
#> Acc : 86.000
#> ==================== Epoch: 85/101 ====================
#> Training............
#> Validation Loss: 32.4524
#> Acc : 68.000
#> ==================== Epoch: 86/101 ====================
#> Training............
#> Validation Loss: 32.4343
#> Acc : 54.000
#> ==================== Epoch: 87/101 ====================
#> Training............
#> Validation Loss: 32.4879
#> Acc : 79.000
#> ==================== Epoch: 88/101 ====================
#> Training............
#> Validation Loss: 32.6224
#> Acc : 61.000
#> ==================== Epoch: 89/101 ====================
#> Training............
#> Validation Loss: 32.4471
#> Acc : 68.000
#> ==================== Epoch: 90/101 ====================
#> Training............
#> Validation Loss: 32.4686
#> Acc : 75.000
#> ==================== Epoch: 91/101 ====================
#> Training............
#> Validation Loss: 32.5360
#> Acc : 61.000
#> ==================== Epoch: 92/101 ====================
#> Training............
#> Validation Loss: 32.4304
#> Acc : 79.000
#> ==================== Epoch: 93/101 ====================
#> Training............
#> Validation Loss: 32.5232
#> Acc : 71.000
#> ==================== Epoch: 94/101 ====================
#> Training............
#> Validation Loss: 32.4515
#> Acc : 64.000
#> ==================== Epoch: 95/101 ====================
#> Training............
#> Validation Loss: 32.6347
#> Acc : 71.000
#> ==================== Epoch: 96/101 ====================
#> Training............
#> Validation Loss: 32.4506
#> Acc : 71.000
#> ==================== Epoch: 97/101 ====================
#> Training............
#> Validation Loss: 32.4264
#> Acc : 71.000
#> ==================== Epoch: 98/101 ====================
#> Training............
#> Validation Loss: 32.4248
#> Acc : 79.000
#> ==================== Epoch: 99/101 ====================
#> Training............
#> Validation Loss: 32.4205
#> Acc : 68.000
#> ==================== Epoch: 100/101 ====================
#> Training............
#> Validation Loss: 32.6022
#> Acc : 82.000
#> ==================== Epoch: 101/101 ====================
#> Training............
#> Validation Loss: 32.4567
#> Acc : 71.000
#> FOLD 1
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/101 ====================
#> Training............
#> Validation Loss: 32.0741
#> Acc : 74.000
#> ==================== Epoch: 2/101 ====================
#> Training............
#> Validation Loss: 32.0793
#> Acc : 78.000
#> ==================== Epoch: 3/101 ====================
#> Training............
#> Validation Loss: 32.0112
#> Acc : 67.000
#> ==================== Epoch: 4/101 ====================
#> Training............
#> Validation Loss: 31.9913
#> Acc : 70.000
#> ==================== Epoch: 5/101 ====================
#> Training............
#> Validation Loss: 31.9664
#> Acc : 74.000
#> ==================== Epoch: 6/101 ====================
#> Training............
#> Validation Loss: 31.9633
#> Acc : 67.000
#> ==================== Epoch: 7/101 ====================
#> Training............
#> Validation Loss: 32.0019
#> Acc : 59.000
#> ==================== Epoch: 8/101 ====================
#> Training............
#> Validation Loss: 31.9894
#> Acc : 78.000
#> ==================== Epoch: 9/101 ====================
#> Training............
#> Validation Loss: 31.9527
#> Acc : 56.000
#> ==================== Epoch: 10/101 ====================
#> Training............
#> Validation Loss: 42.0798
#> Acc : 52.000
#> ==================== Epoch: 11/101 ====================
#> Training............
#> Validation Loss: 42.0882
#> Acc : 44.000
#> ==================== Epoch: 12/101 ====================
#> Training............
#> Validation Loss: 42.0450
#> Acc : 63.000
#> ==================== Epoch: 13/101 ====================
#> Training............
#> Validation Loss: 33.1219
#> Acc : 70.000
#> ==================== Epoch: 14/101 ====================
#> Training............
#> Validation Loss: 32.6106
#> Acc : 89.000
#> ==================== Epoch: 15/101 ====================
#> Training............
#> Validation Loss: 31.8354
#> Acc : 74.000
#> ==================== Epoch: 16/101 ====================
#> Training............
#> Validation Loss: 32.1190
#> Acc : 81.000
#> ==================== Epoch: 17/101 ====================
#> Training............
#> Validation Loss: 32.2270
#> Acc : 70.000
#> ==================== Epoch: 18/101 ====================
#> Training............
#> Validation Loss: 32.4035
#> Acc : 67.000
#> ==================== Epoch: 19/101 ====================
#> Training............
#> Validation Loss: 31.9205
#> Acc : 59.000
#> ==================== Epoch: 20/101 ====================
#> Training............
#> Validation Loss: 31.7721
#> Acc : 93.000
#> ==================== Epoch: 21/101 ====================
#> Training............
#> Validation Loss: 31.7417
#> Acc : 93.000
#> ==================== Epoch: 22/101 ====================
#> Training............
#> Validation Loss: 32.2490
#> Acc : 81.000
#> ==================== Epoch: 23/101 ====================
#> Training............
#> Validation Loss: 31.7279
#> Acc : 85.000
#> ==================== Epoch: 24/101 ====================
#> Training............
#> Validation Loss: 31.5132
#> Acc : 78.000
#> ==================== Epoch: 25/101 ====================
#> Training............
#> Validation Loss: 31.8979
#> Acc : 74.000
#> ==================== Epoch: 26/101 ====================
#> Training............
#> Validation Loss: 31.9311
#> Acc : 78.000
#> ==================== Epoch: 27/101 ====================
#> Training............
#> Validation Loss: 31.9809
#> Acc : 74.000
#> ==================== Epoch: 28/101 ====================
#> Training............
#> Validation Loss: 31.9282
#> Acc : 78.000
#> ==================== Epoch: 29/101 ====================
#> Training............
#> Validation Loss: 32.0917
#> Acc : 81.000
#> ==================== Epoch: 30/101 ====================
#> Training............
#> Validation Loss: 31.9609
#> Acc : 56.000
#> ==================== Epoch: 31/101 ====================
#> Training............
#> Validation Loss: 31.9452
#> Acc : 93.000
#> ==================== Epoch: 32/101 ====================
#> Training............
#> Validation Loss: 32.0786
#> Acc : 78.000
#> ==================== Epoch: 33/101 ====================
#> Training............
#> Validation Loss: 31.9342
#> Acc : 70.000
#> ==================== Epoch: 34/101 ====================
#> Training............
#> Validation Loss: 31.9449
#> Acc : 81.000
#> ==================== Epoch: 35/101 ====================
#> Training............
#> Validation Loss: 31.9726
#> Acc : 70.000
#> ==================== Epoch: 36/101 ====================
#> Training............
#> Validation Loss: 31.9385
#> Acc : 67.000
#> ==================== Epoch: 37/101 ====================
#> Training............
#> Validation Loss: 31.9732
#> Acc : 81.000
#> ==================== Epoch: 38/101 ====================
#> Training............
#> Validation Loss: 31.9723
#> Acc : 78.000
#> ==================== Epoch: 39/101 ====================
#> Training............
#> Validation Loss: 31.9287
#> Acc : 78.000
#> ==================== Epoch: 40/101 ====================
#> Training............
#> Validation Loss: 31.9447
#> Acc : 78.000
#> ==================== Epoch: 41/101 ====================
#> Training............
#> Validation Loss: 32.0138
#> Acc : 67.000
#> ==================== Epoch: 42/101 ====================
#> Training............
#> Validation Loss: 31.9472
#> Acc : 74.000
#> ==================== Epoch: 43/101 ====================
#> Training............
#> Validation Loss: 32.0060
#> Acc : 81.000
#> ==================== Epoch: 44/101 ====================
#> Training............
#> Validation Loss: 31.9748
#> Acc : 81.000
#> ==================== Epoch: 45/101 ====================
#> Training............
#> Validation Loss: 31.9444
#> Acc : 81.000
#> ==================== Epoch: 46/101 ====================
#> Training............
#> Validation Loss: 32.2896
#> Acc : 78.000
#> ==================== Epoch: 47/101 ====================
#> Training............
#> Validation Loss: 32.3560
#> Acc : 74.000
#> ==================== Epoch: 48/101 ====================
#> Training............
#> Validation Loss: 32.1566
#> Acc : 74.000
#> ==================== Epoch: 49/101 ====================
#> Training............
#> Validation Loss: 31.9675
#> Acc : 63.000
#> ==================== Epoch: 50/101 ====================
#> Training............
#> Validation Loss: 31.9837
#> Acc : 67.000
#> ==================== Epoch: 51/101 ====================
#> Training............
#> Validation Loss: 31.9219
#> Acc : 81.000
#> ==================== Epoch: 52/101 ====================
#> Training............
#> Validation Loss: 31.9377
#> Acc : 81.000
#> ==================== Epoch: 53/101 ====================
#> Training............
#> Validation Loss: 31.9479
#> Acc : 78.000
#> ==================== Epoch: 54/101 ====================
#> Training............
#> Validation Loss: 31.9445
#> Acc : 70.000
#> ==================== Epoch: 55/101 ====================
#> Training............
#> Validation Loss: 32.0322
#> Acc : 78.000
#> ==================== Epoch: 56/101 ====================
#> Training............
#> Validation Loss: 31.9841
#> Acc : 63.000
#> ==================== Epoch: 57/101 ====================
#> Training............
#> Validation Loss: 32.0148
#> Acc : 89.000
#> ==================== Epoch: 58/101 ====================
#> Training............
#> Validation Loss: 32.0192
#> Acc : 78.000
#> ==================== Epoch: 59/101 ====================
#> Training............
#> Validation Loss: 32.0356
#> Acc : 81.000
#> ==================== Epoch: 60/101 ====================
#> Training............
#> Validation Loss: 31.9601
#> Acc : 85.000
#> ==================== Epoch: 61/101 ====================
#> Training............
#> Validation Loss: 31.9739
#> Acc : 78.000
#> ==================== Epoch: 62/101 ====================
#> Training............
#> Validation Loss: 31.9333
#> Acc : 89.000
#> ==================== Epoch: 63/101 ====================
#> Training............
#> Validation Loss: 31.9509
#> Acc : 67.000
#> ==================== Epoch: 64/101 ====================
#> Training............
#> Validation Loss: 31.9794
#> Acc : 74.000
#> ==================== Epoch: 65/101 ====================
#> Training............
#> Validation Loss: 31.9404
#> Acc : 67.000
#> ==================== Epoch: 66/101 ====================
#> Training............
#> Validation Loss: 31.9598
#> Acc : 59.000
#> ==================== Epoch: 67/101 ====================
#> Training............
#> Validation Loss: 31.9947
#> Acc : 59.000
#> ==================== Epoch: 68/101 ====================
#> Training............
#> Validation Loss: 31.9563
#> Acc : 67.000
#> ==================== Epoch: 69/101 ====================
#> Training............
#> Validation Loss: 31.9281
#> Acc : 78.000
#> ==================== Epoch: 70/101 ====================
#> Training............
#> Validation Loss: 32.1664
#> Acc : 81.000
#> ==================== Epoch: 71/101 ====================
#> Training............
#> Validation Loss: 31.9624
#> Acc : 56.000
#> ==================== Epoch: 72/101 ====================
#> Training............
#> Validation Loss: 31.9584
#> Acc : 70.000
#> ==================== Epoch: 73/101 ====================
#> Training............
#> Validation Loss: 31.9461
#> Acc : 81.000
#> ==================== Epoch: 74/101 ====================
#> Training............
#> Validation Loss: 31.9632
#> Acc : 78.000
#> ==================== Epoch: 75/101 ====================
#> Training............
#> Validation Loss: 31.9631
#> Acc : 78.000
#> ==================== Epoch: 76/101 ====================
#> Training............
#> Validation Loss: 31.9898
#> Acc : 70.000
#> ==================== Epoch: 77/101 ====================
#> Training............
#> Validation Loss: 31.9634
#> Acc : 67.000
#> ==================== Epoch: 78/101 ====================
#> Training............
#> Validation Loss: 31.9527
#> Acc : 89.000
#> ==================== Epoch: 79/101 ====================
#> Training............
#> Validation Loss: 32.0380
#> Acc : 67.000
#> ==================== Epoch: 80/101 ====================
#> Training............
#> Validation Loss: 31.9615
#> Acc : 70.000
#> ==================== Epoch: 81/101 ====================
#> Training............
#> Validation Loss: 31.9719
#> Acc : 78.000
#> ==================== Epoch: 82/101 ====================
#> Training............
#> Validation Loss: 31.9661
#> Acc : 70.000
#> ==================== Epoch: 83/101 ====================
#> Training............
#> Validation Loss: 31.9625
#> Acc : 74.000
#> ==================== Epoch: 84/101 ====================
#> Training............
#> Validation Loss: 31.9785
#> Acc : 78.000
#> ==================== Epoch: 85/101 ====================
#> Training............
#> Validation Loss: 31.9491
#> Acc : 70.000
#> ==================== Epoch: 86/101 ====================
#> Training............
#> Validation Loss: 31.9629
#> Acc : 70.000
#> ==================== Epoch: 87/101 ====================
#> Training............
#> Validation Loss: 31.9587
#> Acc : 81.000
#> ==================== Epoch: 88/101 ====================
#> Training............
#> Validation Loss: 32.1117
#> Acc : 70.000
#> ==================== Epoch: 89/101 ====================
#> Training............
#> Validation Loss: 31.9552
#> Acc : 59.000
#> ==================== Epoch: 90/101 ====================
#> Training............
#> Validation Loss: 31.9508
#> Acc : 74.000
#> ==================== Epoch: 91/101 ====================
#> Training............
#> Validation Loss: 31.9446
#> Acc : 70.000
#> ==================== Epoch: 92/101 ====================
#> Training............
#> Validation Loss: 31.9456
#> Acc : 89.000
#> ==================== Epoch: 93/101 ====================
#> Training............
#> Validation Loss: 31.9771
#> Acc : 44.000
#> ==================== Epoch: 94/101 ====================
#> Training............
#> Validation Loss: 31.9220
#> Acc : 74.000
#> ==================== Epoch: 95/101 ====================
#> Training............
#> Validation Loss: 31.9390
#> Acc : 78.000
#> ==================== Epoch: 96/101 ====================
#> Training............
#> Validation Loss: 31.9546
#> Acc : 63.000
#> ==================== Epoch: 97/101 ====================
#> Training............
#> Validation Loss: 31.9418
#> Acc : 74.000
#> ==================== Epoch: 98/101 ====================
#> Training............
#> Validation Loss: 31.9513
#> Acc : 74.000
#> ==================== Epoch: 99/101 ====================
#> Training............
#> Validation Loss: 31.9636
#> Acc : 81.000
#> ==================== Epoch: 100/101 ====================
#> Training............
#> Validation Loss: 31.9668
#> Acc : 70.000
#> ==================== Epoch: 101/101 ====================
#> Training............
#> Validation Loss: 31.9640
#> Acc : 81.000
#> FOLD 2
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/101 ====================
#> Training............
#> Validation Loss: 32.7515
#> Acc : 81.000
#> ==================== Epoch: 2/101 ====================
#> Training............
#> Validation Loss: 32.5457
#> Acc : 74.000
#> ==================== Epoch: 3/101 ====================
#> Training............
#> Validation Loss: 33.1386
#> Acc : 70.000
#> ==================== Epoch: 4/101 ====================
#> Training............
#> Validation Loss: 32.9074
#> Acc : 74.000
#> ==================== Epoch: 5/101 ====================
#> Training............
#> Validation Loss: 32.3112
#> Acc : 59.000
#> ==================== Epoch: 6/101 ====================
#> Training............
#> Validation Loss: 32.2465
#> Acc : 81.000
#> ==================== Epoch: 7/101 ====================
#> Training............
#> Validation Loss: 32.2952
#> Acc : 70.000
#> ==================== Epoch: 8/101 ====================
#> Training............
#> Validation Loss: 32.2554
#> Acc : 59.000
#> ==================== Epoch: 9/101 ====================
#> Training............
#> Validation Loss: 32.2470
#> Acc : 93.000
#> ==================== Epoch: 10/101 ====================
#> Training............
#> Validation Loss: 32.2554
#> Acc : 63.000
#> ==================== Epoch: 11/101 ====================
#> Training............
#> Validation Loss: 32.2492
#> Acc : 78.000
#> ==================== Epoch: 12/101 ====================
#> Training............
#> Validation Loss: 32.2571
#> Acc : 78.000
#> ==================== Epoch: 13/101 ====================
#> Training............
#> Validation Loss: 32.2611
#> Acc : 78.000
#> ==================== Epoch: 14/101 ====================
#> Training............
#> Validation Loss: 32.2393
#> Acc : 63.000
#> ==================== Epoch: 15/101 ====================
#> Training............
#> Validation Loss: 32.2470
#> Acc : 85.000
#> ==================== Epoch: 16/101 ====================
#> Training............
#> Validation Loss: 32.2468
#> Acc : 67.000
#> ==================== Epoch: 17/101 ====================
#> Training............
#> Validation Loss: 32.2287
#> Acc : 67.000
#> ==================== Epoch: 18/101 ====================
#> Training............
#> Validation Loss: 32.2556
#> Acc : 63.000
#> ==================== Epoch: 19/101 ====================
#> Training............
#> Validation Loss: 32.2520
#> Acc : 78.000
#> ==================== Epoch: 20/101 ====================
#> Training............
#> Validation Loss: 32.2724
#> Acc : 78.000
#> ==================== Epoch: 21/101 ====================
#> Training............
#> Validation Loss: 32.9061
#> Acc : 74.000
#> ==================== Epoch: 22/101 ====================
#> Training............
#> Validation Loss: 32.2217
#> Acc : 74.000
#> ==================== Epoch: 23/101 ====================
#> Training............
#> Validation Loss: 32.2754
#> Acc : 63.000
#> ==================== Epoch: 24/101 ====================
#> Training............
#> Validation Loss: 32.2897
#> Acc : 56.000
#> ==================== Epoch: 25/101 ====================
#> Training............
#> Validation Loss: 32.2237
#> Acc : 78.000
#> ==================== Epoch: 26/101 ====================
#> Training............
#> Validation Loss: 32.3220
#> Acc : 70.000
#> ==================== Epoch: 27/101 ====================
#> Training............
#> Validation Loss: 32.2558
#> Acc : 56.000
#> ==================== Epoch: 28/101 ====================
#> Training............
#> Validation Loss: 32.2462
#> Acc : 59.000
#> ==================== Epoch: 29/101 ====================
#> Training............
#> Validation Loss: 32.3054
#> Acc : 70.000
#> ==================== Epoch: 30/101 ====================
#> Training............
#> Validation Loss: 32.2338
#> Acc : 78.000
#> ==================== Epoch: 31/101 ====================
#> Training............
#> Validation Loss: 32.2343
#> Acc : 63.000
#> ==================== Epoch: 32/101 ====================
#> Training............
#> Validation Loss: 32.2325
#> Acc : 74.000
#> ==================== Epoch: 33/101 ====================
#> Training............
#> Validation Loss: 32.2339
#> Acc : 67.000
#> ==================== Epoch: 34/101 ====================
#> Training............
#> Validation Loss: 32.2460
#> Acc : 81.000
#> ==================== Epoch: 35/101 ====================
#> Training............
#> Validation Loss: 32.2630
#> Acc : 74.000
#> ==================== Epoch: 36/101 ====================
#> Training............
#> Validation Loss: 32.2373
#> Acc : 59.000
#> ==================== Epoch: 37/101 ====================
#> Training............
#> Validation Loss: 32.2395
#> Acc : 59.000
#> ==================== Epoch: 38/101 ====================
#> Training............
#> Validation Loss: 32.2701
#> Acc : 74.000
#> ==================== Epoch: 39/101 ====================
#> Training............
#> Validation Loss: 32.3130
#> Acc : 81.000
#> ==================== Epoch: 40/101 ====================
#> Training............
#> Validation Loss: 32.2522
#> Acc : 67.000
#> ==================== Epoch: 41/101 ====================
#> Training............
#> Validation Loss: 32.2397
#> Acc : 63.000
#> ==================== Epoch: 42/101 ====================
#> Training............
#> Validation Loss: 32.3460
#> Acc : 70.000
#> ==================== Epoch: 43/101 ====================
#> Training............
#> Validation Loss: 32.9451
#> Acc : 81.000
#> ==================== Epoch: 44/101 ====================
#> Training............
#> Validation Loss: 32.3240
#> Acc : 70.000
#> ==================== Epoch: 45/101 ====================
#> Training............
#> Validation Loss: 32.4281
#> Acc : 67.000
#> ==================== Epoch: 46/101 ====================
#> Training............
#> Validation Loss: 32.2544
#> Acc : 70.000
#> ==================== Epoch: 47/101 ====================
#> Training............
#> Validation Loss: 32.2397
#> Acc : 81.000
#> ==================== Epoch: 48/101 ====================
#> Training............
#> Validation Loss: 32.2309
#> Acc : 81.000
#> ==================== Epoch: 49/101 ====================
#> Training............
#> Validation Loss: 32.2377
#> Acc : 74.000
#> ==================== Epoch: 50/101 ====================
#> Training............
#> Validation Loss: 32.2374
#> Acc : 74.000
#> ==================== Epoch: 51/101 ====================
#> Training............
#> Validation Loss: 32.2315
#> Acc : 74.000
#> ==================== Epoch: 52/101 ====================
#> Training............
#> Validation Loss: 32.2288
#> Acc : 85.000
#> ==================== Epoch: 53/101 ====================
#> Training............
#> Validation Loss: 32.2446
#> Acc : 67.000
#> ==================== Epoch: 54/101 ====================
#> Training............
#> Validation Loss: 32.2045
#> Acc : 85.000
#> ==================== Epoch: 55/101 ====================
#> Training............
#> Validation Loss: 32.2060
#> Acc : 89.000
#> ==================== Epoch: 56/101 ====================
#> Training............
#> Validation Loss: 32.4252
#> Acc : 74.000
#> ==================== Epoch: 57/101 ====================
#> Training............
#> Validation Loss: 32.2831
#> Acc : 70.000
#> ==================== Epoch: 58/101 ====================
#> Training............
#> Validation Loss: 32.1882
#> Acc : 89.000
#> ==================== Epoch: 59/101 ====================
#> Training............
#> Validation Loss: 32.2554
#> Acc : 67.000
#> ==================== Epoch: 60/101 ====================
#> Training............
#> Validation Loss: 32.1626
#> Acc : 74.000
#> ==================== Epoch: 61/101 ====================
#> Training............
#> Validation Loss: 32.2494
#> Acc : 63.000
#> ==================== Epoch: 62/101 ====================
#> Training............
#> Validation Loss: 32.1637
#> Acc : 78.000
#> ==================== Epoch: 63/101 ====================
#> Training............
#> Validation Loss: 32.3259
#> Acc : 74.000
#> ==================== Epoch: 64/101 ====================
#> Training............
#> Validation Loss: 32.2530
#> Acc : 63.000
#> ==================== Epoch: 65/101 ====================
#> Training............
#> Validation Loss: 32.2392
#> Acc : 70.000
#> ==================== Epoch: 66/101 ====================
#> Training............
#> Validation Loss: 32.2068
#> Acc : 81.000
#> ==================== Epoch: 67/101 ====================
#> Training............
#> Validation Loss: 32.2431
#> Acc : 74.000
#> ==================== Epoch: 68/101 ====================
#> Training............
#> Validation Loss: 32.6418
#> Acc : 89.000
#> ==================== Epoch: 69/101 ====================
#> Training............
#> Validation Loss: 32.2659
#> Acc : 63.000
#> ==================== Epoch: 70/101 ====================
#> Training............
#> Validation Loss: 32.2025
#> Acc : 70.000
#> ==================== Epoch: 71/101 ====================
#> Training............
#> Validation Loss: 32.2328
#> Acc : 78.000
#> ==================== Epoch: 72/101 ====================
#> Training............
#> Validation Loss: 32.1969
#> Acc : 78.000
#> ==================== Epoch: 73/101 ====================
#> Training............
#> Validation Loss: 32.1521
#> Acc : 81.000
#> ==================== Epoch: 74/101 ====================
#> Training............
#> Validation Loss: 32.1717
#> Acc : 67.000
#> ==================== Epoch: 75/101 ====================
#> Training............
#> Validation Loss: 32.2132
#> Acc : 70.000
#> ==================== Epoch: 76/101 ====================
#> Training............
#> Validation Loss: 32.2570
#> Acc : 70.000
#> ==================== Epoch: 77/101 ====================
#> Training............
#> Validation Loss: 32.2470
#> Acc : 78.000
#> ==================== Epoch: 78/101 ====================
#> Training............
#> Validation Loss: 32.2170
#> Acc : 81.000
#> ==================== Epoch: 79/101 ====================
#> Training............
#> Validation Loss: 32.1842
#> Acc : 74.000
#> ==================== Epoch: 80/101 ====================
#> Training............
#> Validation Loss: 32.2305
#> Acc : 81.000
#> ==================== Epoch: 81/101 ====================
#> Training............
#> Validation Loss: 32.2114
#> Acc : 85.000
#> ==================== Epoch: 82/101 ====================
#> Training............
#> Validation Loss: 32.1777
#> Acc : 63.000
#> ==================== Epoch: 83/101 ====================
#> Training............
#> Validation Loss: 32.2724
#> Acc : 78.000
#> ==================== Epoch: 84/101 ====================
#> Training............
#> Validation Loss: 32.1935
#> Acc : 70.000
#> ==================== Epoch: 85/101 ====================
#> Training............
#> Validation Loss: 32.1724
#> Acc : 74.000
#> ==================== Epoch: 86/101 ====================
#> Training............
#> Validation Loss: 32.2101
#> Acc : 85.000
#> ==================== Epoch: 87/101 ====================
#> Training............
#> Validation Loss: 32.2215
#> Acc : 78.000
#> ==================== Epoch: 88/101 ====================
#> Training............
#> Validation Loss: 32.2357
#> Acc : 63.000
#> ==================== Epoch: 89/101 ====================
#> Training............
#> Validation Loss: 32.2769
#> Acc : 81.000
#> ==================== Epoch: 90/101 ====================
#> Training............
#> Validation Loss: 32.2098
#> Acc : 70.000
#> ==================== Epoch: 91/101 ====================
#> Training............
#> Validation Loss: 32.2341
#> Acc : 70.000
#> ==================== Epoch: 92/101 ====================
#> Training............
#> Validation Loss: 32.2384
#> Acc : 74.000
#> ==================== Epoch: 93/101 ====================
#> Training............
#> Validation Loss: 32.1831
#> Acc : 81.000
#> ==================== Epoch: 94/101 ====================
#> Training............
#> Validation Loss: 32.2071
#> Acc : 85.000
#> ==================== Epoch: 95/101 ====================
#> Training............
#> Validation Loss: 32.2244
#> Acc : 74.000
#> ==================== Epoch: 96/101 ====================
#> Training............
#> Validation Loss: 32.2809
#> Acc : 56.000
#> ==================== Epoch: 97/101 ====================
#> Training............
#> Validation Loss: 32.1799
#> Acc : 67.000
#> ==================== Epoch: 98/101 ====================
#> Training............
#> Validation Loss: 32.1987
#> Acc : 74.000
#> ==================== Epoch: 99/101 ====================
#> Training............
#> Validation Loss: 32.2270
#> Acc : 74.000
#> ==================== Epoch: 100/101 ====================
#> Training............
#> Validation Loss: 32.2151
#> Acc : 85.000
#> ==================== Epoch: 101/101 ====================
#> Training............
#> Validation Loss: 32.2073
#> Acc : 81.000
#> FOLD 3
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/101 ====================
#> Training............
#> Validation Loss: 32.4459
#> Acc : 89.000
#> ==================== Epoch: 2/101 ====================
#> Training............
#> Validation Loss: 32.4004
#> Acc : 74.000
#> ==================== Epoch: 3/101 ====================
#> Training............
#> Validation Loss: 32.3176
#> Acc : 78.000
#> ==================== Epoch: 4/101 ====================
#> Training............
#> Validation Loss: 32.2679
#> Acc : 59.000
#> ==================== Epoch: 5/101 ====================
#> Training............
#> Validation Loss: 32.2924
#> Acc : 78.000
#> ==================== Epoch: 6/101 ====================
#> Training............
#> Validation Loss: 32.3137
#> Acc : 70.000
#> ==================== Epoch: 7/101 ====================
#> Training............
#> Validation Loss: 32.2433
#> Acc : 74.000
#> ==================== Epoch: 8/101 ====================
#> Training............
#> Validation Loss: 32.2696
#> Acc : 70.000
#> ==================== Epoch: 9/101 ====================
#> Training............
#> Validation Loss: 32.1701
#> Acc : 78.000
#> ==================== Epoch: 10/101 ====================
#> Training............
#> Validation Loss: 32.1451
#> Acc : 70.000
#> ==================== Epoch: 11/101 ====================
#> Training............
#> Validation Loss: 32.2696
#> Acc : 74.000
#> ==================== Epoch: 12/101 ====================
#> Training............
#> Validation Loss: 32.2576
#> Acc : 78.000
#> ==================== Epoch: 13/101 ====================
#> Training............
#> Validation Loss: 32.2771
#> Acc : 81.000
#> ==================== Epoch: 14/101 ====================
#> Training............
#> Validation Loss: 32.1397
#> Acc : 78.000
#> ==================== Epoch: 15/101 ====================
#> Training............
#> Validation Loss: 32.1111
#> Acc : 63.000
#> ==================== Epoch: 16/101 ====================
#> Training............
#> Validation Loss: 32.2222
#> Acc : 70.000
#> ==================== Epoch: 17/101 ====================
#> Training............
#> Validation Loss: 32.1745
#> Acc : 63.000
#> ==================== Epoch: 18/101 ====================
#> Training............
#> Validation Loss: 32.2462
#> Acc : 74.000
#> ==================== Epoch: 19/101 ====================
#> Training............
#> Validation Loss: 32.1151
#> Acc : 67.000
#> ==================== Epoch: 20/101 ====================
#> Training............
#> Validation Loss: 32.1163
#> Acc : 74.000
#> ==================== Epoch: 21/101 ====================
#> Training............
#> Validation Loss: 32.2448
#> Acc : 81.000
#> ==================== Epoch: 22/101 ====================
#> Training............
#> Validation Loss: 32.2887
#> Acc : 70.000
#> ==================== Epoch: 23/101 ====================
#> Training............
#> Validation Loss: 32.3035
#> Acc : 67.000
#> ==================== Epoch: 24/101 ====================
#> Training............
#> Validation Loss: 32.2374
#> Acc : 81.000
#> ==================== Epoch: 25/101 ====================
#> Training............
#> Validation Loss: 32.4458
#> Acc : 70.000
#> ==================== Epoch: 26/101 ====================
#> Training............
#> Validation Loss: 32.3264
#> Acc : 81.000
#> ==================== Epoch: 27/101 ====================
#> Training............
#> Validation Loss: 32.1449
#> Acc : 78.000
#> ==================== Epoch: 28/101 ====================
#> Training............
#> Validation Loss: 32.2199
#> Acc : 74.000
#> ==================== Epoch: 29/101 ====================
#> Training............
#> Validation Loss: 32.2986
#> Acc : 81.000
#> ==================== Epoch: 30/101 ====================
#> Training............
#> Validation Loss: 32.1877
#> Acc : 74.000
#> ==================== Epoch: 31/101 ====================
#> Training............
#> Validation Loss: 32.3110
#> Acc : 78.000
#> ==================== Epoch: 32/101 ====================
#> Training............
#> Validation Loss: 32.2457
#> Acc : 63.000
#> ==================== Epoch: 33/101 ====================
#> Training............
#> Validation Loss: 32.2906
#> Acc : 81.000
#> ==================== Epoch: 34/101 ====================
#> Training............
#> Validation Loss: 32.2804
#> Acc : 70.000
#> ==================== Epoch: 35/101 ====================
#> Training............
#> Validation Loss: 32.2962
#> Acc : 74.000
#> ==================== Epoch: 36/101 ====================
#> Training............
#> Validation Loss: 32.2793
#> Acc : 85.000
#> ==================== Epoch: 37/101 ====================
#> Training............
#> Validation Loss: 32.3082
#> Acc : 78.000
#> ==================== Epoch: 38/101 ====================
#> Training............
#> Validation Loss: 32.2681
#> Acc : 81.000
#> ==================== Epoch: 39/101 ====================
#> Training............
#> Validation Loss: 32.2872
#> Acc : 70.000
#> ==================== Epoch: 40/101 ====================
#> Training............
#> Validation Loss: 32.2857
#> Acc : 78.000
#> ==================== Epoch: 41/101 ====================
#> Training............
#> Validation Loss: 32.3191
#> Acc : 70.000
#> ==================== Epoch: 42/101 ====================
#> Training............
#> Validation Loss: 32.3137
#> Acc : 67.000
#> ==================== Epoch: 43/101 ====================
#> Training............
#> Validation Loss: 32.3099
#> Acc : 67.000
#> ==================== Epoch: 44/101 ====================
#> Training............
#> Validation Loss: 32.2905
#> Acc : 89.000
#> ==================== Epoch: 45/101 ====================
#> Training............
#> Validation Loss: 32.3119
#> Acc : 56.000
#> ==================== Epoch: 46/101 ====================
#> Training............
#> Validation Loss: 32.3027
#> Acc : 63.000
#> ==================== Epoch: 47/101 ====================
#> Training............
#> Validation Loss: 32.2989
#> Acc : 81.000
#> ==================== Epoch: 48/101 ====================
#> Training............
#> Validation Loss: 32.1966
#> Acc : 63.000
#> ==================== Epoch: 49/101 ====================
#> Training............
#> Validation Loss: 32.2982
#> Acc : 78.000
#> ==================== Epoch: 50/101 ====================
#> Training............
#> Validation Loss: 32.3127
#> Acc : 85.000
#> ==================== Epoch: 51/101 ====================
#> Training............
#> Validation Loss: 32.3110
#> Acc : 74.000
#> ==================== Epoch: 52/101 ====================
#> Training............
#> Validation Loss: 32.3562
#> Acc : 74.000
#> ==================== Epoch: 53/101 ====================
#> Training............
#> Validation Loss: 32.2441
#> Acc : 63.000
#> ==================== Epoch: 54/101 ====================
#> Training............
#> Validation Loss: 32.3322
#> Acc : 59.000
#> ==================== Epoch: 55/101 ====================
#> Training............
#> Validation Loss: 32.3293
#> Acc : 78.000
#> ==================== Epoch: 56/101 ====================
#> Training............
#> Validation Loss: 32.3212
#> Acc : 70.000
#> ==================== Epoch: 57/101 ====================
#> Training............
#> Validation Loss: 32.2967
#> Acc : 85.000
#> ==================== Epoch: 58/101 ====================
#> Training............
#> Validation Loss: 32.3245
#> Acc : 70.000
#> ==================== Epoch: 59/101 ====================
#> Training............
#> Validation Loss: 32.2777
#> Acc : 89.000
#> ==================== Epoch: 60/101 ====================
#> Training............
#> Validation Loss: 32.3346
#> Acc : 63.000
#> ==================== Epoch: 61/101 ====================
#> Training............
#> Validation Loss: 32.2933
#> Acc : 70.000
#> ==================== Epoch: 62/101 ====================
#> Training............
#> Validation Loss: 32.3149
#> Acc : 70.000
#> ==================== Epoch: 63/101 ====================
#> Training............
#> Validation Loss: 32.3489
#> Acc : 70.000
#> ==================== Epoch: 64/101 ====================
#> Training............
#> Validation Loss: 32.3209
#> Acc : 59.000
#> ==================== Epoch: 65/101 ====================
#> Training............
#> Validation Loss: 32.3399
#> Acc : 67.000
#> ==================== Epoch: 66/101 ====================
#> Training............
#> Validation Loss: 32.3396
#> Acc : 81.000
#> ==================== Epoch: 67/101 ====================
#> Training............
#> Validation Loss: 32.3359
#> Acc : 81.000
#> ==================== Epoch: 68/101 ====================
#> Training............
#> Validation Loss: 32.3114
#> Acc : 63.000
#> ==================== Epoch: 69/101 ====================
#> Training............
#> Validation Loss: 32.3262
#> Acc : 67.000
#> ==================== Epoch: 70/101 ====================
#> Training............
#> Validation Loss: 32.3311
#> Acc : 63.000
#> ==================== Epoch: 71/101 ====================
#> Training............
#> Validation Loss: 32.3334
#> Acc : 67.000
#> ==================== Epoch: 72/101 ====================
#> Training............
#> Validation Loss: 32.3105
#> Acc : 74.000
#> ==================== Epoch: 73/101 ====================
#> Training............
#> Validation Loss: 32.2733
#> Acc : 63.000
#> ==================== Epoch: 74/101 ====================
#> Training............
#> Validation Loss: 32.3717
#> Acc : 74.000
#> ==================== Epoch: 75/101 ====================
#> Training............
#> Validation Loss: 32.3453
#> Acc : 70.000
#> ==================== Epoch: 76/101 ====================
#> Training............
#> Validation Loss: 32.3088
#> Acc : 89.000
#> ==================== Epoch: 77/101 ====================
#> Training............
#> Validation Loss: 32.3270
#> Acc : 78.000
#> ==================== Epoch: 78/101 ====================
#> Training............
#> Validation Loss: 32.3451
#> Acc : 78.000
#> ==================== Epoch: 79/101 ====================
#> Training............
#> Validation Loss: 32.3111
#> Acc : 70.000
#> ==================== Epoch: 80/101 ====================
#> Training............
#> Validation Loss: 32.3300
#> Acc : 85.000
#> ==================== Epoch: 81/101 ====================
#> Training............
#> Validation Loss: 32.3218
#> Acc : 74.000
#> ==================== Epoch: 82/101 ====================
#> Training............
#> Validation Loss: 32.4535
#> Acc : 74.000
#> ==================== Epoch: 83/101 ====================
#> Training............
#> Validation Loss: 32.3112
#> Acc : 81.000
#> ==================== Epoch: 84/101 ====================
#> Training............
#> Validation Loss: 32.3338
#> Acc : 70.000
#> ==================== Epoch: 85/101 ====================
#> Training............
#> Validation Loss: 32.3199
#> Acc : 74.000
#> ==================== Epoch: 86/101 ====================
#> Training............
#> Validation Loss: 32.3067
#> Acc : 85.000
#> ==================== Epoch: 87/101 ====================
#> Training............
#> Validation Loss: 32.3575
#> Acc : 63.000
#> ==================== Epoch: 88/101 ====================
#> Training............
#> Validation Loss: 32.3516
#> Acc : 85.000
#> ==================== Epoch: 89/101 ====================
#> Training............
#> Validation Loss: 32.1795
#> Acc : 74.000
#> ==================== Epoch: 90/101 ====================
#> Training............
#> Validation Loss: 32.2965
#> Acc : 67.000
#> ==================== Epoch: 91/101 ====================
#> Training............
#> Validation Loss: 32.3653
#> Acc : 78.000
#> ==================== Epoch: 92/101 ====================
#> Training............
#> Validation Loss: 32.3756
#> Acc : 74.000
#> ==================== Epoch: 93/101 ====================
#> Training............
#> Validation Loss: 32.2672
#> Acc : 78.000
#> ==================== Epoch: 94/101 ====================
#> Training............
#> Validation Loss: 32.1775
#> Acc : 59.000
#> ==================== Epoch: 95/101 ====================
#> Training............
#> Validation Loss: 32.2946
#> Acc : 59.000
#> ==================== Epoch: 96/101 ====================
#> Training............
#> Validation Loss: 32.3335
#> Acc : 67.000
#> ==================== Epoch: 97/101 ====================
#> Training............
#> Validation Loss: 32.3237
#> Acc : 78.000
#> ==================== Epoch: 98/101 ====================
#> Training............
#> Validation Loss: 32.2961
#> Acc : 78.000
#> ==================== Epoch: 99/101 ====================
#> Training............
#> Validation Loss: 32.2880
#> Acc : 67.000
#> ==================== Epoch: 100/101 ====================
#> Training............
#> Validation Loss: 32.1370
#> Acc : 93.000
#> ==================== Epoch: 101/101 ====================
#> Training............
#> Validation Loss: 32.2427
#> Acc : 74.000
#> FOLD 4
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/101 ====================
#> Training............
#> Validation Loss: 33.2708
#> Acc : 63.000
#> ==================== Epoch: 2/101 ====================
#> Training............
#> Validation Loss: 32.3070
#> Acc : 70.000
#> ==================== Epoch: 3/101 ====================
#> Training............
#> Validation Loss: 32.2484
#> Acc : 70.000
#> ==================== Epoch: 4/101 ====================
#> Training............
#> Validation Loss: 32.2537
#> Acc : 74.000
#> ==================== Epoch: 5/101 ====================
#> Training............
#> Validation Loss: 32.2348
#> Acc : 70.000
#> ==================== Epoch: 6/101 ====================
#> Training............
#> Validation Loss: 32.1730
#> Acc : 78.000
#> ==================== Epoch: 7/101 ====================
#> Training............
#> Validation Loss: 32.1084
#> Acc : 63.000
#> ==================== Epoch: 8/101 ====================
#> Training............
#> Validation Loss: 32.1946
#> Acc : 81.000
#> ==================== Epoch: 9/101 ====================
#> Training............
#> Validation Loss: 32.1127
#> Acc : 70.000
#> ==================== Epoch: 10/101 ====================
#> Training............
#> Validation Loss: 32.1157
#> Acc : 67.000
#> ==================== Epoch: 11/101 ====================
#> Training............
#> Validation Loss: 32.1735
#> Acc : 70.000
#> ==================== Epoch: 12/101 ====================
#> Training............
#> Validation Loss: 32.0610
#> Acc : 59.000
#> ==================== Epoch: 13/101 ====================
#> Training............
#> Validation Loss: 32.1058
#> Acc : 85.000
#> ==================== Epoch: 14/101 ====================
#> Training............
#> Validation Loss: 32.2133
#> Acc : 63.000
#> ==================== Epoch: 15/101 ====================
#> Training............
#> Validation Loss: 32.1410
#> Acc : 74.000
#> ==================== Epoch: 16/101 ====================
#> Training............
#> Validation Loss: 32.1577
#> Acc : 70.000
#> ==================== Epoch: 17/101 ====================
#> Training............
#> Validation Loss: 32.1245
#> Acc : 89.000
#> ==================== Epoch: 18/101 ====================
#> Training............
#> Validation Loss: 32.1331
#> Acc : 74.000
#> ==================== Epoch: 19/101 ====================
#> Training............
#> Validation Loss: 32.0699
#> Acc : 74.000
#> ==================== Epoch: 20/101 ====================
#> Training............
#> Validation Loss: 32.1260
#> Acc : 70.000
#> ==================== Epoch: 21/101 ====================
#> Training............
#> Validation Loss: 32.0995
#> Acc : 67.000
#> ==================== Epoch: 22/101 ====================
#> Training............
#> Validation Loss: 32.2034
#> Acc : 85.000
#> ==================== Epoch: 23/101 ====================
#> Training............
#> Validation Loss: 32.1913
#> Acc : 56.000
#> ==================== Epoch: 24/101 ====================
#> Training............
#> Validation Loss: 32.1626
#> Acc : 74.000
#> ==================== Epoch: 25/101 ====================
#> Training............
#> Validation Loss: 32.1170
#> Acc : 93.000
#> ==================== Epoch: 26/101 ====================
#> Training............
#> Validation Loss: 32.1352
#> Acc : 74.000
#> ==================== Epoch: 27/101 ====================
#> Training............
#> Validation Loss: 31.9213
#> Acc : 63.000
#> ==================== Epoch: 28/101 ====================
#> Training............
#> Validation Loss: 31.9532
#> Acc : 70.000
#> ==================== Epoch: 29/101 ====================
#> Training............
#> Validation Loss: 32.1377
#> Acc : 67.000
#> ==================== Epoch: 30/101 ====================
#> Training............
#> Validation Loss: 32.1315
#> Acc : 67.000
#> ==================== Epoch: 31/101 ====================
#> Training............
#> Validation Loss: 32.1613
#> Acc : 74.000
#> ==================== Epoch: 32/101 ====================
#> Training............
#> Validation Loss: 32.1136
#> Acc : 85.000
#> ==================== Epoch: 33/101 ====================
#> Training............
#> Validation Loss: 32.1270
#> Acc : 74.000
#> ==================== Epoch: 34/101 ====================
#> Training............
#> Validation Loss: 32.1646
#> Acc : 56.000
#> ==================== Epoch: 35/101 ====================
#> Training............
#> Validation Loss: 32.1336
#> Acc : 70.000
#> ==================== Epoch: 36/101 ====================
#> Training............
#> Validation Loss: 32.1141
#> Acc : 78.000
#> ==================== Epoch: 37/101 ====================
#> Training............
#> Validation Loss: 32.1370
#> Acc : 85.000
#> ==================== Epoch: 38/101 ====================
#> Training............
#> Validation Loss: 32.1512
#> Acc : 74.000
#> ==================== Epoch: 39/101 ====================
#> Training............
#> Validation Loss: 32.1385
#> Acc : 74.000
#> ==================== Epoch: 40/101 ====================
#> Training............
#> Validation Loss: 32.0590
#> Acc : 81.000
#> ==================== Epoch: 41/101 ====================
#> Training............
#> Validation Loss: 32.1215
#> Acc : 67.000
#> ==================== Epoch: 42/101 ====================
#> Training............
#> Validation Loss: 32.1529
#> Acc : 78.000
#> ==================== Epoch: 43/101 ====================
#> Training............
#> Validation Loss: 32.1561
#> Acc : 89.000
#> ==================== Epoch: 44/101 ====================
#> Training............
#> Validation Loss: 32.1800
#> Acc : 74.000
#> ==================== Epoch: 45/101 ====================
#> Training............
#> Validation Loss: 32.0987
#> Acc : 81.000
#> ==================== Epoch: 46/101 ====================
#> Training............
#> Validation Loss: 32.0948
#> Acc : 96.000
#> ==================== Epoch: 47/101 ====================
#> Training............
#> Validation Loss: 31.9811
#> Acc : 78.000
#> ==================== Epoch: 48/101 ====================
#> Training............
#> Validation Loss: 32.1245
#> Acc : 70.000
#> ==================== Epoch: 49/101 ====================
#> Training............
#> Validation Loss: 32.1620
#> Acc : 63.000
#> ==================== Epoch: 50/101 ====================
#> Training............
#> Validation Loss: 32.1584
#> Acc : 70.000
#> ==================== Epoch: 51/101 ====================
#> Training............
#> Validation Loss: 32.1604
#> Acc : 74.000
#> ==================== Epoch: 52/101 ====================
#> Training............
#> Validation Loss: 32.1582
#> Acc : 74.000
#> ==================== Epoch: 53/101 ====================
#> Training............
#> Validation Loss: 32.1735
#> Acc : 78.000
#> ==================== Epoch: 54/101 ====================
#> Training............
#> Validation Loss: 32.1184
#> Acc : 74.000
#> ==================== Epoch: 55/101 ====================
#> Training............
#> Validation Loss: 32.1632
#> Acc : 63.000
#> ==================== Epoch: 56/101 ====================
#> Training............
#> Validation Loss: 31.9969
#> Acc : 67.000
#> ==================== Epoch: 57/101 ====================
#> Training............
#> Validation Loss: 32.1353
#> Acc : 70.000
#> ==================== Epoch: 58/101 ====================
#> Training............
#> Validation Loss: 31.9599
#> Acc : 70.000
#> ==================== Epoch: 59/101 ====================
#> Training............
#> Validation Loss: 31.9234
#> Acc : 81.000
#> ==================== Epoch: 60/101 ====================
#> Training............
#> Validation Loss: 32.0813
#> Acc : 78.000
#> ==================== Epoch: 61/101 ====================
#> Training............
#> Validation Loss: 32.1591
#> Acc : 56.000
#> ==================== Epoch: 62/101 ====================
#> Training............
#> Validation Loss: 32.1370
#> Acc : 63.000
#> ==================== Epoch: 63/101 ====================
#> Training............
#> Validation Loss: 31.8901
#> Acc : 74.000
#> ==================== Epoch: 64/101 ====================
#> Training............
#> Validation Loss: 31.9577
#> Acc : 70.000
#> ==================== Epoch: 65/101 ====================
#> Training............
#> Validation Loss: 31.8985
#> Acc : 70.000
#> ==================== Epoch: 66/101 ====================
#> Training............
#> Validation Loss: 31.9028
#> Acc : 63.000
#> ==================== Epoch: 67/101 ====================
#> Training............
#> Validation Loss: 31.9257
#> Acc : 56.000
#> ==================== Epoch: 68/101 ====================
#> Training............
#> Validation Loss: 32.1350
#> Acc : 81.000
#> ==================== Epoch: 69/101 ====================
#> Training............
#> Validation Loss: 32.1109
#> Acc : 81.000
#> ==================== Epoch: 70/101 ====================
#> Training............
#> Validation Loss: 31.9052
#> Acc : 74.000
#> ==================== Epoch: 71/101 ====================
#> Training............
#> Validation Loss: 31.9997
#> Acc : 85.000
#> ==================== Epoch: 72/101 ====================
#> Training............
#> Validation Loss: 31.9319
#> Acc : 78.000
#> ==================== Epoch: 73/101 ====================
#> Training............
#> Validation Loss: 31.9095
#> Acc : 78.000
#> ==================== Epoch: 74/101 ====================
#> Training............
#> Validation Loss: 31.9159
#> Acc : 67.000
#> ==================== Epoch: 75/101 ====================
#> Training............
#> Validation Loss: 31.9633
#> Acc : 85.000
#> ==================== Epoch: 76/101 ====================
#> Training............
#> Validation Loss: 32.0942
#> Acc : 74.000
#> ==================== Epoch: 77/101 ====================
#> Training............
#> Validation Loss: 32.1012
#> Acc : 74.000
#> ==================== Epoch: 78/101 ====================
#> Training............
#> Validation Loss: 32.0188
#> Acc : 78.000
#> ==================== Epoch: 79/101 ====================
#> Training............
#> Validation Loss: 32.1427
#> Acc : 78.000
#> ==================== Epoch: 80/101 ====================
#> Training............
#> Validation Loss: 31.9381
#> Acc : 70.000
#> ==================== Epoch: 81/101 ====================
#> Training............
#> Validation Loss: 32.1456
#> Acc : 74.000
#> ==================== Epoch: 82/101 ====================
#> Training............
#> Validation Loss: 32.0446
#> Acc : 74.000
#> ==================== Epoch: 83/101 ====================
#> Training............
#> Validation Loss: 31.9292
#> Acc : 89.000
#> ==================== Epoch: 84/101 ====================
#> Training............
#> Validation Loss: 31.9324
#> Acc : 81.000
#> ==================== Epoch: 85/101 ====================
#> Training............
#> Validation Loss: 31.9263
#> Acc : 67.000
#> ==================== Epoch: 86/101 ====================
#> Training............
#> Validation Loss: 32.0984
#> Acc : 59.000
#> ==================== Epoch: 87/101 ====================
#> Training............
#> Validation Loss: 32.0853
#> Acc : 74.000
#> ==================== Epoch: 88/101 ====================
#> Training............
#> Validation Loss: 32.1416
#> Acc : 67.000
#> ==================== Epoch: 89/101 ====================
#> Training............
#> Validation Loss: 31.9688
#> Acc : 70.000
#> ==================== Epoch: 90/101 ====================
#> Training............
#> Validation Loss: 32.0012
#> Acc : 85.000
#> ==================== Epoch: 91/101 ====================
#> Training............
#> Validation Loss: 31.9488
#> Acc : 70.000
#> ==================== Epoch: 92/101 ====================
#> Training............
#> Validation Loss: 31.9313
#> Acc : 67.000
#> ==================== Epoch: 93/101 ====================
#> Training............
#> Validation Loss: 31.9252
#> Acc : 74.000
#> ==================== Epoch: 94/101 ====================
#> Training............
#> Validation Loss: 31.9020
#> Acc : 70.000
#> ==================== Epoch: 95/101 ====================
#> Training............
#> Validation Loss: 31.9364
#> Acc : 81.000
#> ==================== Epoch: 96/101 ====================
#> Training............
#> Validation Loss: 31.9293
#> Acc : 85.000
#> ==================== Epoch: 97/101 ====================
#> Training............
#> Validation Loss: 31.9496
#> Acc : 70.000
#> ==================== Epoch: 98/101 ====================
#> Training............
#> Validation Loss: 31.9407
#> Acc : 78.000
#> ==================== Epoch: 99/101 ====================
#> Training............
#> Validation Loss: 31.9400
#> Acc : 81.000
#> ==================== Epoch: 100/101 ====================
#> Training............
#> Validation Loss: 31.9351
#> Acc : 56.000
#> ==================== Epoch: 101/101 ====================
#> Training............
#> Validation Loss: 31.9966
#> Acc : 67.000
```

```python
@torch.no_grad() 
def inference(model, states_list, test_dataloader, device): 
  """
  Do inference for different model folds 
  """
  with HiddenPrints(): 
    model.eval() 
  all_preds = [] 
  for state in states_list: 
    print(f"State: {state}") 
    state_dict = torch.load(state) 
    model.load_state_dict(state_dict) 
    model = model.to(device) 
    
    # Clean 
    del state_dict 
    gc.collect() 
    
    preds = [] 
    prog = tqdm.tqdm(test_dataloader, total = len(test_dataloader), disable=True) 
    for x in prog: 
      x = x.to(device, dtype = torch.float32) 
      outputs = model(x) 
      preds.append(outputs.squeeze(-1).cpu().detach().numpy()) 
      
    all_preds.append(np.concatenate(preds)) 
    
    # Clean 
    gc.collect() 
    torch.cuda.empty_cache() 
    
  return all_preds 
```



```python
model_dir = "./model" 
states_list = [os.path.join(model_dir, x) for x in os.listdir(model_dir) if x.endswith(".pth")]

test_dataset = SpaceshipTitanicData(features = test_data.drop(['Transported'], axis = 1), 
                                    target = None, 
                                    is_test = True
                                    )

test_loader = DataLoader(test_dataset, batch_size = 32, shuffle = False) 

print("Predictions for all folds") 
#> Predictions for all folds
predictions = inference(model, states_list, test_loader, Config.device) 
#> State: ./model/fold_1_model.pth
#> State: ./model/fold_4_model.pth
#> State: ./model/fold_3_model.pth
#> State: ./model/fold_2_model.pth
#> State: ./model/fold_0_model.pth
pred = pd.DataFrame(predictions).T.mean(axis = 1).tolist() 
pred = [True if p >= 0.5 else False for p in pred]

submission = pd.DataFrame({
  "PassengerId" : test['PassengerId'], 
  "Transported" : pred
})

submission.to_csv(os.path.join(os.getcwd(), "model", "submission.csv"), index = False)
```













