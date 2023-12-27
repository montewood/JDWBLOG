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
  nb_epochs = 5 
  train_bs = 32 
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
      
      filepath = f"fold_{fold}_model.pth" 
      torch.save(model.state_dict(), filepath)
```

```
#> FOLD 0
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/6 ====================
#> Training............
#> Validation Loss: 32.8174
#> Acc : 64.000
#> ==================== Epoch: 2/6 ====================
#> Training............
#> Validation Loss: 32.7516
#> Acc : 71.000
#> ==================== Epoch: 3/6 ====================
#> Training............
#> Validation Loss: 32.7519
#> Acc : 64.000
#> ==================== Epoch: 4/6 ====================
#> Training............
#> Validation Loss: 32.6855
#> Acc : 68.000
#> ==================== Epoch: 5/6 ====================
#> Training............
#> Validation Loss: 32.6683
#> Acc : 79.000
#> ==================== Epoch: 6/6 ====================
#> Training............
#> Validation Loss: 32.6295
#> Acc : 68.000
#> FOLD 1
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/6 ====================
#> Training............
#> Validation Loss: 32.2687
#> Acc : 85.000
#> ==================== Epoch: 2/6 ====================
#> Training............
#> Validation Loss: 32.2114
#> Acc : 63.000
#> ==================== Epoch: 3/6 ====================
#> Training............
#> Validation Loss: 32.1925
#> Acc : 85.000
#> ==================== Epoch: 4/6 ====================
#> Training............
#> Validation Loss: 32.0480
#> Acc : 67.000
#> ==================== Epoch: 5/6 ====================
#> Training............
#> Validation Loss: 32.1194
#> Acc : 74.000
#> ==================== Epoch: 6/6 ====================
#> Training............
#> Validation Loss: 32.0231
#> Acc : 81.000
#> FOLD 2
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/6 ====================
#> Training............
#> Validation Loss: 32.3008
#> Acc : 70.000
#> ==================== Epoch: 2/6 ====================
#> Training............
#> Validation Loss: 32.2260
#> Acc : 78.000
#> ==================== Epoch: 3/6 ====================
#> Training............
#> Validation Loss: 32.2042
#> Acc : 89.000
#> ==================== Epoch: 4/6 ====================
#> Training............
#> Validation Loss: 32.1834
#> Acc : 70.000
#> ==================== Epoch: 5/6 ====================
#> Training............
#> Validation Loss: 32.2365
#> Acc : 81.000
#> ==================== Epoch: 6/6 ====================
#> Training............
#> Validation Loss: 32.1433
#> Acc : 78.000
#> FOLD 3
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/6 ====================
#> Training............
#> Validation Loss: 32.0743
#> Acc : 63.000
#> ==================== Epoch: 2/6 ====================
#> Training............
#> Validation Loss: 31.9906
#> Acc : 67.000
#> ==================== Epoch: 3/6 ====================
#> Training............
#> Validation Loss: 31.9196
#> Acc : 74.000
#> ==================== Epoch: 4/6 ====================
#> Training............
#> Validation Loss: 31.9834
#> Acc : 59.000
#> ==================== Epoch: 5/6 ====================
#> Training............
#> Validation Loss: 31.9415
#> Acc : 67.000
#> ==================== Epoch: 6/6 ====================
#> Training............
#> Validation Loss: 31.9479
#> Acc : 67.000
#> FOLD 4
#> --------------------
#> [INFO]: Starting training! 
#> 
#> ==================== Epoch: 1/6 ====================
#> Training............
#> Validation Loss: 42.0512
#> Acc : 44.000
#> ==================== Epoch: 2/6 ====================
#> Training............
#> Validation Loss: 42.0439
#> Acc : 52.000
#> ==================== Epoch: 3/6 ====================
#> Training............
#> Validation Loss: 41.9111
#> Acc : 48.000
#> ==================== Epoch: 4/6 ====================
#> Training............
#> Validation Loss: 41.9185
#> Acc : 56.000
#> ==================== Epoch: 5/6 ====================
#> Training............
#> Validation Loss: 41.7244
#> Acc : 52.000
#> ==================== Epoch: 6/6 ====================
#> Training............
#> Validation Loss: 33.6761
#> Acc : 70.000
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













