---
date: '2023-11-27'
title: 🚀0.80032|Spaceship Titanic with a FCNN in PyTorch
author: JDW
type: book
weight: 10
output: md_document
---



<center>

 **Original Notebook** : <https://www.kaggle.com/code/marcokurepa/0-80032-spaceship-titanic-with-a-fcnn-in-pytorch> 
 
</center>

# Data Dictionary


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np 
from scipy import stats 

sns.set_style("whitegrid") 
plt.style.use("fivethirtyeight")
```


```python
import os 
train = pd.read_csv(os.path.join(os.getcwd(), "data", "train.csv"))
test  = pd.read_csv(os.path.join(os.getcwd(), "data", "test.csv"))
```



```python
train.head(); test.head()
```

```
#>   PassengerId HomePlanet CryoSleep  ... VRDeck               Name  Transported
#> 0     0001_01     Europa     False  ...    0.0    Maham Ofracculy        False
#> 1     0002_01      Earth     False  ...   44.0       Juanna Vines         True
#> 2     0003_01     Europa     False  ...   49.0      Altark Susent        False
#> 3     0003_02     Europa     False  ...  193.0       Solam Susent        False
#> 4     0004_01      Earth     False  ...    2.0  Willy Santantines         True
#> 
#> [5 rows x 14 columns]
#>   PassengerId HomePlanet CryoSleep  ...     Spa VRDeck              Name
#> 0     0013_01      Earth      True  ...     0.0    0.0   Nelly Carsoning
#> 1     0018_01      Earth     False  ...  2823.0    0.0    Lerome Peckers
#> 2     0019_01     Europa      True  ...     0.0    0.0   Sabih Unhearfus
#> 3     0021_01     Europa     False  ...   181.0  585.0  Meratz Caltilter
#> 4     0023_01      Earth     False  ...     0.0    0.0   Brence Harperez
#> 
#> [5 rows x 13 columns]
```


# Exploratory Data Analysis (EDA) 


```python
train.info() 
```

```
#> <class 'pandas.core.frame.DataFrame'>
#> RangeIndex: 8693 entries, 0 to 8692
#> Data columns (total 14 columns):
#>  #   Column        Non-Null Count  Dtype  
#> ---  ------        --------------  -----  
#>  0   PassengerId   8693 non-null   object 
#>  1   HomePlanet    8492 non-null   object 
#>  2   CryoSleep     8476 non-null   object 
#>  3   Cabin         8494 non-null   object 
#>  4   Destination   8511 non-null   object 
#>  5   Age           8514 non-null   float64
#>  6   VIP           8490 non-null   object 
#>  7   RoomService   8512 non-null   float64
#>  8   FoodCourt     8510 non-null   float64
#>  9   ShoppingMall  8485 non-null   float64
#>  10  Spa           8510 non-null   float64
#>  11  VRDeck        8505 non-null   float64
#>  12  Name          8493 non-null   object 
#>  13  Transported   8693 non-null   bool   
#> dtypes: bool(1), float64(6), object(7)
#> memory usage: 891.5+ KB
```
## Looking for NAs 
If a feature has >= 30% nulls, we disregard it. 


```python
pd.Series(data  = [train[col].isna().sum() / train[col].size * 100 for col in train.columns],
          index = [col for col in train.columns], 
          name  = "Percentage Missing").sort_values(ascending = False)  
```

```
#> CryoSleep       2.496261
#> ShoppingMall    2.392730
#> VIP             2.335212
#> HomePlanet      2.312205
#> Name            2.300702
#> Cabin           2.289198
#> VRDeck          2.162660
#> FoodCourt       2.105142
#> Spa             2.105142
#> Destination     2.093639
#> RoomService     2.082135
#> Age             2.059128
#> PassengerId     0.000000
#> Transported     0.000000
#> Name: Percentage Missing, dtype: float64
```

With `rangeIndex : 8693`, it would seem that all our features are filled to a satisfactory level. 

## Checking Duplicates 

```python
duplicates = pd.DataFrame(train.loc[train.duplicated(subset=["Name"])].query("not Name.isnull()"))
print("Duplicate samples:\n"); duplicates[:5]
```

```
#> Duplicate samples:
#> 
#>      PassengerId HomePlanet CryoSleep  ...  VRDeck                Name  Transported
#> 956      1018_01      Earth      True  ...     0.0  Elaney Webstephrey         True
#> 2700     2892_03      Earth      True  ...     0.0     Sharie Gallenry         True
#> 2852     3081_01     Europa     False  ...     1.0      Gollux Reedall         True
#> 2930     3176_01     Europa     False  ...  1464.0  Ankalik Nateansive         True
#> 3291     3535_02       Mars      True  ...     0.0         Grake Porki         True
#> 
#> [5 rows x 14 columns]
```

```python
print(f"Total amount of duplicate names: {len(duplicates.index)}")
```

```
#> Total amount of duplicate names: 20
```


```python
train.query('Name == "Elaney Webstephrey"')
```

```
#>     PassengerId HomePlanet CryoSleep  ... VRDeck                Name  Transported
#> 156     0179_01      Earth     False  ...   11.0  Elaney Webstephrey         True
#> 956     1018_01      Earth      True  ...    0.0  Elaney Webstephrey         True
#> 
#> [2 rows x 14 columns]
```

```python
train.query('Name == "Sharie Gallenry"')
```

```
#>      PassengerId HomePlanet CryoSleep  ... VRDeck             Name  Transported
#> 1812     1935_01      Earth      True  ...    0.0  Sharie Gallenry        False
#> 2700     2892_03      Earth      True  ...    0.0  Sharie Gallenry         True
#> 
#> [2 rows x 14 columns]
```

```python
train.query('Name == "Gollux Reedall"')
```

```
#>      PassengerId HomePlanet CryoSleep  ... VRDeck            Name  Transported
#> 827      0881_01     Europa     False  ...   26.0  Gollux Reedall        False
#> 2852     3081_01     Europa     False  ...    1.0  Gollux Reedall         True
#> 
#> [2 rows x 14 columns]
```

As the names are not a feature which our model will consider whilst learning, and the data for these duplicates differs, we'll leave them in our model. Odds are they just duplicated by chance while the model was being generated. 


## Splitting the `Cabin` Feature 

```python
train.Cabin
```

```
#> 0          B/0/P
#> 1          F/0/S
#> 2          A/0/S
#> 3          A/0/S
#> 4          F/1/S
#>           ...   
#> 8688      A/98/P
#> 8689    G/1499/S
#> 8690    G/1500/S
#> 8691     E/608/S
#> 8692     E/608/S
#> Name: Cabin, Length: 8693, dtype: object
```

```python

train[["CabinDeck", "CabinNum", "CabinSide"]] = train.Cabin.apply(
  lambda cabin : pd.Series(
    cabin.split("/") if not pd.isnull(cabin) else [float("NaN"), float("NaN"), float("NaN")]
  )
)

test[["CabinDeck", "CabinNum", "CabinSide"]] = test.Cabin.apply(
  lambda cabin : pd.Series(
    cabin.split("/") if not pd.isnull(cabin) else [float("NaN"), float("NaN"), float("NaN")]
  )
)
```

## Univariate Analysis 
Let's separate numerical and categorical features so we can visualize them separately. 

```python
# Seperating categorical feature names and numerical feature names 
categorical_features = [col for col in train.columns if train[col].dtype == 'object']

# We can remove the following three features from the list since we won't be dealing with those 
categorical_features.remove("Name") 
categorical_features.remove("Cabin") 
categorical_features.remove("CabinNum") 

numerical_features = [
  col for col in train.columns if train[col].dtype == 'int64' or train[col].dtype == "float64"
]

print(f"categorical_features : {categorical_features}")
```

```
#> categorical_features : ['PassengerId', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide']
```

```python
print(f"numerical_features : {numerical_features}")
```

```
#> numerical_features : ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
```


```python
# Seperating the data 
train_cat = train[categorical_features] 
train_num = train[numerical_features]
```


## Numerical Data 
We'll be plotting the numerical data with Kernel Density Estimation (KDE). 


```python
train_num.columns.to_list() 
```

```
#> ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
```



```python
for i in train_num.columns: 
  fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (12, 4)) 
  
  train_num[i].plot(kind = "kde", title = f"{i} KDE All Data", ax = ax1, ) 
  if i == "Age": 
    ax1.set_xlabel("Age (Years)") 
  else: 
    ax2.set_xlabel("Money Spent (Space Currency)") 
    
    
  
  Q1 = train_num[i].quantile(0.25) 
  Q3 = train_num[i].quantile(0.75) 
  IQR = Q3-Q1 
  
  train_num.query(f'(@Q1 - 1.5 * @IQR) <= {i} <= (@Q3 + 1.5 * @IQR)')[i] \
    .plot(kind = 'kde', color = 'm', title = f"{i} KDE IQR", ax = ax2) 
  
  if i == "Age": 
    ax2.set_xlabel("Age (Years)") 
  else: 
    ax2.set_xlabel("Money Spent (Space Currency)") 
  
  plt.tight_layout() 
  plt.show() 
  
  # Get X-axis ranges 
  range_ax1 = ax1.get_xlim() 
  range_ax2 = ax2.get_xlim() 
  
  # Print X-axis ranges 
  print(f"X-axis range for {i} Distribution All Data : {range_ax1}") 
  print(f"X-axis range for {i} Distribution IQR : {range_ax2}") 
```

<img src="/courses/rolling-in-the-kaggle/Spaceship_Titanic/Spaceship-Titanic-with-a-FCNN-in-PyTorch_files/figure-html/unnamed-chunk-13-1.png" width="1152" /><img src="/courses/rolling-in-the-kaggle/Spaceship_Titanic/Spaceship-Titanic-with-a-FCNN-in-PyTorch_files/figure-html/unnamed-chunk-13-2.png" width="1152" /><img src="/courses/rolling-in-the-kaggle/Spaceship_Titanic/Spaceship-Titanic-with-a-FCNN-in-PyTorch_files/figure-html/unnamed-chunk-13-3.png" width="1152" /><img src="/courses/rolling-in-the-kaggle/Spaceship_Titanic/Spaceship-Titanic-with-a-FCNN-in-PyTorch_files/figure-html/unnamed-chunk-13-4.png" width="1152" /><img src="/courses/rolling-in-the-kaggle/Spaceship_Titanic/Spaceship-Titanic-with-a-FCNN-in-PyTorch_files/figure-html/unnamed-chunk-13-5.png" width="1152" /><img src="/courses/rolling-in-the-kaggle/Spaceship_Titanic/Spaceship-Titanic-with-a-FCNN-in-PyTorch_files/figure-html/unnamed-chunk-13-6.png" width="1152" />

- **Agre Distribution**: As the distrubution is already somewhat normalized, there was no major change between the "all data" distribution and the IQR distrubution. We'll leave this this data as is without doing any transformations. 

- **Rest of the Distrubutions**: This distrubution is spiked highly at 0, presenting what looks like an incredibly rapid exponential decay. No transformation is needed when using decision or boost tree based models like XGBoost or CatBoost as they do not prefer Gaussian or standard probability distributions. 


# Feature Engineering 
First we'll clear the dataset to get it ready for training the model with the following: 
- Imputation 
- Categorical Encoding 

First, let's split the training data into features and target variables 

```python
X = train.drop(['Transported'], axis = 1).copy()
y = train.Transported.copy()
```

We'll also pop off `PassengerID` as we'll need it for making our submission. 


```python
test_IDs = test["PassengerId"] 

X = X.drop("PassengerId", axis = 1) 
test = test.drop("PassengerId", axis = 1)  

categorical_features.remove("PassengerId") 
```

## Imputation 
Imputation is a technique for handling missing data. If greater tahn 80% of our samples, it would be wise to drop it entirely. However, in our case we were only missing 2 to 2.5% of values. 

### Imputing Numerical Data 

```python
for i in numerical_features: 
  X[i] = X[i].fillna(X[i].median()) 
  
  test[i] = test[i].fillna(X[i].median())
```

Now we check to make sure there aren't still any missing values. 


```python
X[numerical_features].isnull().sum()
```

```
#> Age             0
#> RoomService     0
#> FoodCourt       0
#> ShoppingMall    0
#> Spa             0
#> VRDeck          0
#> dtype: int64
```

```python
print()
```

```python
test[numerical_features].isnull().sum()
```

```
#> Age             0
#> RoomService     0
#> FoodCourt       0
#> ShoppingMall    0
#> Spa             0
#> VRDeck          0
#> dtype: int64
```


### Imputing Categorical Data 

```python
for i in categorical_features: 
  X[i]    = X[i].fillna(X[i].mode().iloc[0]) 
  test[i] = test[i].fillna(X[i].mode().iloc[0])
```


```python
X[categorical_features].isnull().sum()
```

```
#> HomePlanet     0
#> CryoSleep      0
#> Destination    0
#> VIP            0
#> CabinDeck      0
#> CabinSide      0
#> dtype: int64
```

```python
print()
```

```python
test[categorical_features].isnull().sum()
```

```
#> HomePlanet     0
#> CryoSleep      0
#> Destination    0
#> VIP            0
#> CabinDeck      0
#> CabinSide      0
#> dtype: int64
```

## Categorical Encoding 
Categorical encoding is the process of converting categorical data to numerical values for training models. 

First off, let's check the heads of our training dataset and our test dataset. 


```python
X.head(5)
```

```
#>   HomePlanet  CryoSleep  Cabin  ... CabinDeck  CabinNum  CabinSide
#> 0     Europa      False  B/0/P  ...         B         0          P
#> 1      Earth      False  F/0/S  ...         F         0          S
#> 2     Europa      False  A/0/S  ...         A         0          S
#> 3     Europa      False  A/0/S  ...         A         0          S
#> 4      Earth      False  F/1/S  ...         F         1          S
#> 
#> [5 rows x 15 columns]
```

```python
test.head(5) 
```

```
#>   HomePlanet  CryoSleep  Cabin  ... CabinDeck  CabinNum  CabinSide
#> 0      Earth       True  G/3/S  ...         G         3          S
#> 1      Earth      False  F/4/S  ...         F         4          S
#> 2     Europa       True  C/0/S  ...         C         0          S
#> 3     Europa      False  C/1/S  ...         C         1          S
#> 4      Earth      False  F/5/S  ...         F         5          S
#> 
#> [5 rows x 15 columns]
```

Before categorical encoding, let's remove the `Name` and `CabinNum` columns from the dataframes, since they have way too many unique values or provide way to little information for the model. 



```python
print(numerical_features + categorical_features)
```

```
#> ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck', 'HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'CabinDeck', 'CabinSide']
```

```python
X = X[numerical_features + categorical_features]
test = test[numerical_features + categorical_features]
```

Now, let's have a look at the different categorical features and how many unique values they each have (if using one-hot encoding, they should preferably have less than 15 unique values): 


```python
for i in categorical_features: 
  print(f"Feature name: {i}. \nNo. of unique values: {X[i].nunique()} ({X[i].unique()})\n\n")
```

```
#> Feature name: HomePlanet. 
#> No. of unique values: 3 (['Europa' 'Earth' 'Mars'])
#> 
#> 
#> Feature name: CryoSleep. 
#> No. of unique values: 2 ([False  True])
#> 
#> 
#> Feature name: Destination. 
#> No. of unique values: 3 (['TRAPPIST-1e' 'PSO J318.5-22' '55 Cancri e'])
#> 
#> 
#> Feature name: VIP. 
#> No. of unique values: 2 ([False  True])
#> 
#> 
#> Feature name: CabinDeck. 
#> No. of unique values: 8 (['B' 'F' 'A' 'G' 'E' 'D' 'C' 'T'])
#> 
#> 
#> Feature name: CabinSide. 
#> No. of unique values: 2 (['P' 'S'])
```
All the features are well whitin the threshold, so we can use one-hot encoding for all of them. However, we'll use ordinal encoding for `CabinDeck` as it will group related values. 

Before we start encoding let's take a quick look at the datataypes again: 


```python
X.dtypes
```

```
#> Age             float64
#> RoomService     float64
#> FoodCourt       float64
#> ShoppingMall    float64
#> Spa             float64
#> VRDeck          float64
#> HomePlanet       object
#> CryoSleep          bool
#> Destination      object
#> VIP                bool
#> CabinDeck        object
#> CabinSide        object
#> dtype: object
```

`CryoSleep` and `VIP` are of type boll, not object. We can turn those columns into 1s and 0s as so: 


```python
X[["CryoSleep", "VIP"]] = X[["CryoSleep", "VIP"]].astype(int)
test[["CryoSleep", "VIP"]] = test[["CryoSleep", "VIP"]].astype(int)
```



```python
X.head(5)
```

```
#>     Age  RoomService  FoodCourt  ...  VIP  CabinDeck  CabinSide
#> 0  39.0          0.0        0.0  ...    0          B          P
#> 1  24.0        109.0        9.0  ...    0          F          S
#> 2  58.0         43.0     3576.0  ...    1          A          S
#> 3  33.0          0.0     1283.0  ...    0          A          S
#> 4  16.0        303.0       70.0  ...    0          F          S
#> 
#> [5 rows x 12 columns]
```

```python
test.head(5) 
```

```
#>     Age  RoomService  FoodCourt  ...  VIP  CabinDeck  CabinSide
#> 0  27.0          0.0        0.0  ...    0          G          S
#> 1  19.0          0.0        9.0  ...    0          F          S
#> 2  31.0          0.0        0.0  ...    0          C          S
#> 3  38.0          0.0     6652.0  ...    0          C          S
#> 4  20.0         10.0        0.0  ...    0          F          S
#> 
#> [5 rows x 12 columns]
```


```python
X.dtypes
```

```
#> Age             float64
#> RoomService     float64
#> FoodCourt       float64
#> ShoppingMall    float64
#> Spa             float64
#> VRDeck          float64
#> HomePlanet       object
#> CryoSleep         int64
#> Destination      object
#> VIP               int64
#> CabinDeck        object
#> CabinSide        object
#> dtype: object
```

Now to separate `CabinDeck` from the other categorical features: 

```python
ordinal_column_names = ["CabinDeck"] 
one_hot_column_names = ["HomePlanet", "Destination", "CabinSide"]
```

## Ordianl Encoding 
We'll just use a mapping dictionary so we can ordinal encode using just Pandas. 


```python
deck_mapping = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "T": 8}
X[ordinal_column_names]    = X[ordinal_column_names].replace(deck_mapping)
test[ordinal_column_names] = test[ordinal_column_names].replace(deck_mapping)

X.head(5); test.head(5) 
```

```
#>     Age  RoomService  FoodCourt  ...  VIP  CabinDeck  CabinSide
#> 0  39.0          0.0        0.0  ...    0          2          P
#> 1  24.0        109.0        9.0  ...    0          6          S
#> 2  58.0         43.0     3576.0  ...    1          1          S
#> 3  33.0          0.0     1283.0  ...    0          1          S
#> 4  16.0        303.0       70.0  ...    0          6          S
#> 
#> [5 rows x 12 columns]
#>     Age  RoomService  FoodCourt  ...  VIP  CabinDeck  CabinSide
#> 0  27.0          0.0        0.0  ...    0          7          S
#> 1  19.0          0.0        9.0  ...    0          6          S
#> 2  31.0          0.0        0.0  ...    0          3          S
#> 3  38.0          0.0     6652.0  ...    0          3          S
#> 4  20.0         10.0        0.0  ...    0          6          S
#> 
#> [5 rows x 12 columns]
```

As T is so small, I'm not really worried about it hampering the model. 

### Categorical Encoding 

```python
X = pd.get_dummies(X).copy()
test = pd.get_dummies(test).copy()
```


```python
X.dtypes
```

```
#> Age                          float64
#> RoomService                  float64
#> FoodCourt                    float64
#> ShoppingMall                 float64
#> Spa                          float64
#> VRDeck                       float64
#> CryoSleep                      int64
#> VIP                            int64
#> CabinDeck                      int64
#> HomePlanet_Earth                bool
#> HomePlanet_Europa               bool
#> HomePlanet_Mars                 bool
#> Destination_55 Cancri e         bool
#> Destination_PSO J318.5-22       bool
#> Destination_TRAPPIST-1e         bool
#> CabinSide_P                     bool
#> CabinSide_S                     bool
#> dtype: object
```

```python
print()
```

```python
test.dtypes
```

```
#> Age                          float64
#> RoomService                  float64
#> FoodCourt                    float64
#> ShoppingMall                 float64
#> Spa                          float64
#> VRDeck                       float64
#> CryoSleep                      int64
#> VIP                            int64
#> CabinDeck                      int64
#> HomePlanet_Earth                bool
#> HomePlanet_Europa               bool
#> HomePlanet_Mars                 bool
#> Destination_55 Cancri e         bool
#> Destination_PSO J318.5-22       bool
#> Destination_TRAPPIST-1e         bool
#> CabinSide_P                     bool
#> CabinSide_S                     bool
#> dtype: object
```

```python
# bool column convert to unit8 
bool_columns = X.select_dtypes(include = 'bool').columns
X[bool_columns] = X[bool_columns].astype('uint8') 
test[bool_columns] = test[bool_columns].astype('uint8') 

X.dtypes
```

```
#> Age                          float64
#> RoomService                  float64
#> FoodCourt                    float64
#> ShoppingMall                 float64
#> Spa                          float64
#> VRDeck                       float64
#> CryoSleep                      int64
#> VIP                            int64
#> CabinDeck                      int64
#> HomePlanet_Earth               uint8
#> HomePlanet_Europa              uint8
#> HomePlanet_Mars                uint8
#> Destination_55 Cancri e        uint8
#> Destination_PSO J318.5-22      uint8
#> Destination_TRAPPIST-1e        uint8
#> CabinSide_P                    uint8
#> CabinSide_S                    uint8
#> dtype: object
```

```python
print()
```

```python
test.dtypes
```

```
#> Age                          float64
#> RoomService                  float64
#> FoodCourt                    float64
#> ShoppingMall                 float64
#> Spa                          float64
#> VRDeck                       float64
#> CryoSleep                      int64
#> VIP                            int64
#> CabinDeck                      int64
#> HomePlanet_Earth               uint8
#> HomePlanet_Europa              uint8
#> HomePlanet_Mars                uint8
#> Destination_55 Cancri e        uint8
#> Destination_PSO J318.5-22      uint8
#> Destination_TRAPPIST-1e        uint8
#> CabinSide_P                    uint8
#> CabinSide_S                    uint8
#> dtype: object
```


# Model Training 

## Imports 

```python
import torch 
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader, TensorDataset 
from sklearn.metrics import roc_auc_score, confusion_matrix, roc_curve 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
```

## Preparing Data for Training 
Now we need to split the data into the training set, validation set, and test set. 
Furthermore we need to convert the data which is currently stored in dataframes, into Pytorch Tensors. 


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert the data to Numpy arrays and then to Pytorch tensors 
X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
X_valid = torch.tensor(np.array(X_valid), dtype=torch.float32)
X_test = torch.tensor(np.array(X_test), dtype=torch.float32)

y_train = torch.tensor(np.array(y_train), dtype = torch.float32) 
y_valid = torch.tensor(np.array(y_valid), dtype = torch.float32) 
y_test  = torch.tensor(np.array(y_test), dtype = torch.float32) 
```


We also need to create a DataLoader for each dataset. 

```python
train_dataset = TensorDataset(X_train, y_train) 
valid_dataset = TensorDataset(X_valid, y_valid) 
test_dataset = TensorDataset(X_test, y_test) 
train_loader = DataLoader(train_dataset, batch_size = 64, shuffle = True) 
valid_loader = DataLoader(valid_dataset, batch_size = 64) 
test_loader = DataLoader(test_dataset, batch_size = 64)
```


## Define the FCNN 

Here we define the FCNN. Note that we're also using th dropout regularization technique to minimize overfitting. 

```python
class FCNN(nn.Module):
  def __init__(self, input_size): 
    super(FCNN, self).__init__() 
    self.fc1 = nn.Linear(input_size, 128) 
    self.fc2 = nn.Linear(128, 64)
    self.fc3 = nn.Linear(64, 32) 
    self.fc4 = nn.Linear(32, 1) 
    self.dropout = nn.Dropout(0.5) 
    self.sigmoid = nn.Sigmoid() 
    
  def forward(self, x): 
    x = torch.relu(self.fc1(x)) 
    x = self.dropout(x) 
    x = torch.relu(self.fc2(x)) 
    x = self.dropout(x) 
    x = torch.relu(self.fc3(x)) 
    x = self.dropout(x) 
    x = self.fc4(x) 
    x = self.sigmoid(x) 
    return(x)
```


## Initialize the Model 
Here we do a few things: 
- Set the Device : Here we set the device to the CUDA enabled GPU (if present) or the CPU. 
- Initialize the Model 
- Define the Loss Function: Here we use Binary Crossentropy(BCE) to compare the value predicted by the model and the target value. 
- Define the Optimizer and Scheduler: Here we set the attributes of the model. We use Root mean Square Propagation (RMSprop) to set the minibatches, and we add a weight decay for L2 regularization to minimize overfitting. 


```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

input_size = X_train.shape[1] 
model = FCNN(input_size).to(device) 

criterion = nn.BCELoss() 
optimizer = optim.RMSprop(model.parameters(), lr = 0.001, weight_decay = 1e-5) # Added weight_decay for L2 regularization 

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1) 
```

## Training & Validation Loop 
We implement early stops to once agine, minimie the chances of overfitting. 

```python
num_epochs = 100
best_loss = float('inf') 
train_loss_values = [] 
valid_loss_values = [] 

for epoch in range(num_epochs): 
  model.train() 
  train_loss = 0.0 
  for batch_x, batch_y in train_loader: 
    batch_x = batch_x.to(device) 
    batch_y = batch_y.to(device) 
    
    optimizer.zero_grad() 
    outputs = model(batch_x) 
    loss = criterion(outputs.squeeze(), batch_y) 
    loss.backward() 
    optimizer.step() 
    
    train_loss += loss.item() * batch_x.size(0) 
    
  scheduler.step() # Decay learning rate 
  
  # Validation 
  model.eval() 
  valid_loss = 0.0 
  with torch.no_grad(): 
    for batch_x, batch_y in valid_loader: 
      batch_x = batch_x.to(device) 
      batch_y = batch_y.to(device) 
      
      outputs = model(batch_x) 
      loss = criterion(outputs.squeeze(), batch_y) 
      
      valid_loss += loss.item() * batch_x.size(0) 
      
  train_loss /= len(train_loader.dataset) 
  valid_loss /= len(valid_loader.dataset) 
  
  train_loss_values.append(train_loss) 
  valid_loss_values.append(valid_loss) 
  
  print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}") 
  
  # Save the model if validation loss improves 
  if valid_loss < best_loss: 
    best_loss = valid_loss 
    torch.save(model.state_dict(), os.path.join(os.getcwd(), "model", "fcnn_model.pt"))
    
  else:
    print("No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.")
```

```
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 1/100, Train Loss: 5.4054, Valid Loss: 0.5135
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 2/100, Train Loss: 1.3786, Valid Loss: 0.5932
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 3/100, Train Loss: 0.7172, Valid Loss: 0.6003
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 4/100, Train Loss: 0.6188, Valid Loss: 0.5659
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 5/100, Train Loss: 0.6165, Valid Loss: 0.5404
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 6/100, Train Loss: 0.5798, Valid Loss: 0.5351
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 7/100, Train Loss: 0.5659, Valid Loss: 0.5017
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 8/100, Train Loss: 0.5823, Valid Loss: 0.4927
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 9/100, Train Loss: 0.5620, Valid Loss: 0.4810
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 10/100, Train Loss: 0.5318, Valid Loss: 0.4694
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 11/100, Train Loss: 0.5406, Valid Loss: 0.4685
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 12/100, Train Loss: 0.5076, Valid Loss: 0.4495
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 13/100, Train Loss: 0.5133, Valid Loss: 0.4459
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 14/100, Train Loss: 0.5051, Valid Loss: 0.4596
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 15/100, Train Loss: 0.5047, Valid Loss: 0.4547
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 16/100, Train Loss: 0.5070, Valid Loss: 0.4480
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 17/100, Train Loss: 0.5005, Valid Loss: 0.4441
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 18/100, Train Loss: 0.4926, Valid Loss: 0.4462
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 19/100, Train Loss: 0.5040, Valid Loss: 0.4507
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 20/100, Train Loss: 0.4832, Valid Loss: 0.4430
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 21/100, Train Loss: 0.4780, Valid Loss: 0.4379
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 22/100, Train Loss: 0.5010, Valid Loss: 0.4574
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 23/100, Train Loss: 0.4925, Valid Loss: 0.4411
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 24/100, Train Loss: 0.4823, Valid Loss: 0.4444
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 25/100, Train Loss: 0.4729, Valid Loss: 0.4399
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 26/100, Train Loss: 0.4786, Valid Loss: 0.4413
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 27/100, Train Loss: 0.4755, Valid Loss: 0.4402
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 28/100, Train Loss: 0.4640, Valid Loss: 0.4549
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 29/100, Train Loss: 0.4676, Valid Loss: 0.4393
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 30/100, Train Loss: 0.4782, Valid Loss: 0.4372
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 31/100, Train Loss: 0.4668, Valid Loss: 0.4409
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 32/100, Train Loss: 0.4766, Valid Loss: 0.4474
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 33/100, Train Loss: 0.4667, Valid Loss: 0.4434
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 34/100, Train Loss: 0.4635, Valid Loss: 0.4460
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 35/100, Train Loss: 0.4619, Valid Loss: 0.4439
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 36/100, Train Loss: 0.4687, Valid Loss: 0.4499
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 37/100, Train Loss: 0.4682, Valid Loss: 0.4447
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 38/100, Train Loss: 0.4776, Valid Loss: 0.4407
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 39/100, Train Loss: 0.4614, Valid Loss: 0.4407
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 40/100, Train Loss: 0.4552, Valid Loss: 0.4394
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 41/100, Train Loss: 0.4589, Valid Loss: 0.4428
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 42/100, Train Loss: 0.4662, Valid Loss: 0.4409
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 43/100, Train Loss: 0.4645, Valid Loss: 0.4446
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 44/100, Train Loss: 0.4563, Valid Loss: 0.4377
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 45/100, Train Loss: 0.4598, Valid Loss: 0.4412
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 46/100, Train Loss: 0.4566, Valid Loss: 0.4376
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 47/100, Train Loss: 0.4696, Valid Loss: 0.4396
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 48/100, Train Loss: 0.4630, Valid Loss: 0.4455
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 49/100, Train Loss: 0.4554, Valid Loss: 0.4485
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 50/100, Train Loss: 0.4566, Valid Loss: 0.4418
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 51/100, Train Loss: 0.4584, Valid Loss: 0.4401
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 52/100, Train Loss: 0.4601, Valid Loss: 0.4420
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 53/100, Train Loss: 0.4635, Valid Loss: 0.4373
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 54/100, Train Loss: 0.4572, Valid Loss: 0.4360
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 55/100, Train Loss: 0.4510, Valid Loss: 0.4331
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 56/100, Train Loss: 0.4581, Valid Loss: 0.4382
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 57/100, Train Loss: 0.4607, Valid Loss: 0.4362
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 58/100, Train Loss: 0.4532, Valid Loss: 0.4415
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 59/100, Train Loss: 0.4585, Valid Loss: 0.4390
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 60/100, Train Loss: 0.4549, Valid Loss: 0.4392
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 61/100, Train Loss: 0.4553, Valid Loss: 0.4390
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 62/100, Train Loss: 0.4573, Valid Loss: 0.4385
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 63/100, Train Loss: 0.4585, Valid Loss: 0.4388
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 64/100, Train Loss: 0.4532, Valid Loss: 0.4379
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 65/100, Train Loss: 0.4627, Valid Loss: 0.4380
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 66/100, Train Loss: 0.4511, Valid Loss: 0.4376
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 67/100, Train Loss: 0.4567, Valid Loss: 0.4374
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 68/100, Train Loss: 0.4576, Valid Loss: 0.4373
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 69/100, Train Loss: 0.4722, Valid Loss: 0.4368
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 70/100, Train Loss: 0.4534, Valid Loss: 0.4367
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 71/100, Train Loss: 0.4522, Valid Loss: 0.4360
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 72/100, Train Loss: 0.4558, Valid Loss: 0.4357
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 73/100, Train Loss: 0.4590, Valid Loss: 0.4363
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 74/100, Train Loss: 0.4539, Valid Loss: 0.4360
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 75/100, Train Loss: 0.4572, Valid Loss: 0.4362
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 76/100, Train Loss: 0.4549, Valid Loss: 0.4361
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 77/100, Train Loss: 0.4536, Valid Loss: 0.4362
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 78/100, Train Loss: 0.4603, Valid Loss: 0.4363
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 79/100, Train Loss: 0.4514, Valid Loss: 0.4364
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 80/100, Train Loss: 0.4593, Valid Loss: 0.4366
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 81/100, Train Loss: 0.4577, Valid Loss: 0.4366
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 82/100, Train Loss: 0.4543, Valid Loss: 0.4362
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 83/100, Train Loss: 0.4574, Valid Loss: 0.4365
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 84/100, Train Loss: 0.4553, Valid Loss: 0.4364
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 85/100, Train Loss: 0.4610, Valid Loss: 0.4367
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 86/100, Train Loss: 0.4551, Valid Loss: 0.4367
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 87/100, Train Loss: 0.4544, Valid Loss: 0.4371
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 88/100, Train Loss: 0.4610, Valid Loss: 0.4372
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 89/100, Train Loss: 0.4566, Valid Loss: 0.4373
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 90/100, Train Loss: 0.4549, Valid Loss: 0.4368
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 91/100, Train Loss: 0.4535, Valid Loss: 0.4368
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 92/100, Train Loss: 0.4696, Valid Loss: 0.4369
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 93/100, Train Loss: 0.4527, Valid Loss: 0.4369
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 94/100, Train Loss: 0.4558, Valid Loss: 0.4369
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 95/100, Train Loss: 0.4515, Valid Loss: 0.4368
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 96/100, Train Loss: 0.4535, Valid Loss: 0.4368
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 97/100, Train Loss: 0.4522, Valid Loss: 0.4367
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 98/100, Train Loss: 0.4590, Valid Loss: 0.4367
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 99/100, Train Loss: 0.4568, Valid Loss: 0.4367
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
#> Epoch 100/100, Train Loss: 0.4557, Valid Loss: 0.4366
#> No improvement in validation loss for this epoch. Model parameters from last epoch with best validation loss were saved.
```


## Evalute the Model on Test set 
First, let's load the best model. 

```python
model.load_state_dict(torch.load(os.path.join(os.getcwd(), "model", "fcnn_model.pt")))
```

```
#> <All keys matched successfully>
```

Now we run the evaluation loop in the test set. 



```python
model.eval() 
```

```
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
```

```python
test_loss = 0.0 
test_predictions = [] 
with torch.no_grad(): 
  for batch_x, batch_y in test_loader: 
    batch_x = batch_x.to(device) 
    batch_y = batch_y.to(device) 
    
    outputs = model(batch_x) 
    loss = criterion(outputs.squeeze(), batch_y) 
    
    test_loss += loss.item() * batch_x.size(0) 
    test_predictions.extend(outputs.cpu().numpy()) 
    
test_loss /= len(test_loader.dataset) 
test_predictions = np.array(test_predictions) 

# Calculate AUC-ROC for the test set 
test_auc_roc = roc_auc_score(y_test, test_predictions) 

print(f"Test Loss : {test_loss:.4f}") 
```

```
#> Test Loss : 0.4451
```

```python
print(f"Test AUC-ROC: {test_auc_roc:.4f}") 
```

```
#> Test AUC-ROC: 0.8710
```


```python
print(y_test) 
```

```
#> tensor([1., 0., 0.,  ..., 0., 1., 0.])
```


```python
print(test_predictions) 
```

```
#> [[0.2430638 ]
#>  [0.5769563 ]
#>  [0.60590726]
#>  ...
#>  [0.41472107]
#>  [0.7937017 ]
#>  [0.03511113]]
```

# Visualization 

Plot Training & Validation Loss 

```python
plt.figure(figsize = (8, 6)) 
plt.plot(train_loss_values, label = "Training Loss") 
plt.plot(valid_loss_values, label = "Validation Loss") 
plt.xlabel("Epoch") 
plt.ylabel("Loss") 
plt.title("Training and Validation Loss over Epochs") 
plt.legend() 
plt.grid(True) 
plt.show() 
```

<img src="/courses/rolling-in-the-kaggle/Spaceship_Titanic/Spaceship-Titanic-with-a-FCNN-in-PyTorch_files/figure-html/unnamed-chunk-43-13.png" width="768" />

## ROC Curve 


```python
fpr, tpr, _ = roc_curve(y_test, test_predictions) 
plt.figure(figsize = (8, 6)) 
```

```
#> <Figure size 800x600 with 0 Axes>
```

```python
plt.plot(fpr, tpr, label = f"AUC-ROC = {test_auc_roc:.4f}") 
```

```
#> [<matplotlib.lines.Line2D object at 0x7fd020384040>]
```

```python
plt.plot([0, 1], [0, 1], linestyle = "--", color = "r") 
```

```
#> [<matplotlib.lines.Line2D object at 0x7fd0203aef20>]
```

```python
plt.xlabel("False Positive Rate") 
```

```
#> Text(0.5, 0, 'False Positive Rate')
```

```python
plt.ylabel("True Positive Rate") 
```

```
#> Text(0, 0.5, 'True Positive Rate')
```

```python
plt.title("ROC Curve") 
```

```
#> Text(0.5, 1.0, 'ROC Curve')
```

```python
plt.legend() 
```

```
#> <matplotlib.legend.Legend object at 0x7fd02059fbb0>
```

```python
plt.grid(True) 
plt.show() 
```

<img src="/courses/rolling-in-the-kaggle/Spaceship_Titanic/Spaceship-Titanic-with-a-FCNN-in-PyTorch_files/figure-html/unnamed-chunk-44-15.png" width="768" />

## Confusion Matrix 


```python
plt.figure() 
```

```
#> <Figure size 700x500 with 0 Axes>
```

```python
cm = confusion_matrix(y_test.numpy(), test_predictions.round()) 
sns.heatmap(cm, annot = True, fmt = "d") 
```

```
#> <Axes: >
```

```python
plt.title("Confusion Matrix") 
```

```
#> Text(0.5, 1.0, 'Confusion Matrix')
```

```python
plt.xlabel("Predicted Label") 
```

```
#> Text(0.5, 5.583333333333313, 'Predicted Label')
```

```python
plt.ylabel("True Label") 
```

```
#> Text(26.58333333333333, 0.5, 'True Label')
```

```python
plt.show() 
```

<img src="/courses/rolling-in-the-kaggle/Spaceship_Titanic/Spaceship-Titanic-with-a-FCNN-in-PyTorch_files/figure-html/unnamed-chunk-45-17.png" width="672" />

## Classification Accuracy 

```python
binary_predictions = [1 if p>= 0.5 else 0 for p in test_predictions]
test_accuracy = accuracy_score(y_test.numpy(), binary_predictions) 

print(f"Test Accuracy : {test_accuracy:.4f}") 
```

```
#> Test Accuracy : 0.7849
```

# Submission 


```python
train  = pd.read_csv(os.path.join(os.getcwd(), "data", "train.csv"))
submit = pd.read_csv(os.path.join(os.getcwd(), "data", "sample_submission.csv"))
test_IDs = submit['PassengerId']

test_competition = torch.tensor(np.array(test), dtype = torch.float32).to(device) 

model.eval() 
```

```
#> FCNN(
#>   (fc1): Linear(in_features=17, out_features=128, bias=True)
#>   (fc2): Linear(in_features=128, out_features=64, bias=True)
#>   (fc3): Linear(in_features=64, out_features=32, bias=True)
#>   (fc4): Linear(in_features=32, out_features=1, bias=True)
#>   (dropout): Dropout(p=0.5, inplace=False)
#>   (sigmoid): Sigmoid()
#> )
```

```python
test_competition_predictions = [] 
with torch.no_grad(): 
  outputs = model(test_competition) 
  test_competition_predictions.extend(outputs.cpu().numpy()) 
  
test_competition_predictions = np.array(test_competition_predictions).flatten() 

binary_competition_predictions = [True if p >= 0.5 else False for p in test_competition_predictions]

submission = pd.DataFrame({
  "PassengerId" : test_IDs, 
  "Transported" : binary_competition_predictions 
})

submission.to_csv(os.path.join(os.getcwd(), "model", "submission.csv"), index = False)
```



