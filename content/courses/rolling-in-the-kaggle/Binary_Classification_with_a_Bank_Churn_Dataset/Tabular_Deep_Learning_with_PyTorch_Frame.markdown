---
date: '2024-01-11'
title: "📈Tabular Deep Learning with PyTorch Frame"
author: JDW
type: book
weight: 10
output: 
  rmarkdown::html_document()
editor_options: 
  markdown: 
    wrap: 255
---





<center>

 **Original Notebook** : <https://www.kaggle.com/code/yiwenyuan1998/tabular-deep-learning-with-pytorch-frame> 
 
</center>


# Deep Neural Network using Sentence Transformer with PyTorch Frame 

&nbsp; Disclaimer: This tutorial would be helpful to you if you want to know about deep tabular learning. There is no feature engineering uesd in this tutorial. 

&nbsp; PyTorch Frame is a deep learning extension for PyTorch, designed for heterogeneous tabular data with different column type. 

&nbsp; Historically, tree-based models(e.g., XGBoost, Catboost) excelled at tabular learning but had notable limitations, such as integraion difficulties with downstream models, and handling complex column types, such as texts, sequences, and embeddings. 

&nbsp; Pytorch Frame offers you the abillity to integrate with different architectures like GNNs or LLMs. It is production-ready and can also be integrated with OpenAI, CoHere or VoyageAI. Check out example [here](https://github.com/pyg-team/pytorch-frame/blob/master/examples/llm_embedding.py)

&nbsp; For the Bank Churn Prediction task, we will build a deep learning model integrated with a Pretrained Language Model using PyTorch Frame. 

What this notebook will cover 

 - Loading Data with PyTorch Frame 
 - Combining Tabular Deep Learning with Sentence Transformers 
 - Hyperparameter Search with Optuna 
 - Refit on Full Dataset 
 
 
# Import Libraries


```python
from typing import List, Optional 

import os.path as osp 
import pandas as pd 

from sklearn.metrics import roc_auc_score 

import torch 
import torch.nn.functional as F 
from torch import Tensor 
from tqdm import tqdm 

# Use GPU for faster training 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
```


# Create an Interface to use HuggingFace Sentence Transformers 

&nbsp; In this notebook, we are using a sentence transformer called [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2). However, you can play araound with other sentence transformers from HuggingFace as well. 


```python
from transformers import AutoModel, AutoTokenizer 

class TextToEmbedding: 
  def __init__(self, model: str, device: torch.device): 
    self.tokenizer = AutoTokenizer.from_pretrained(model) 
    self.model     = AutoModel.from_pretrained(model).to(device) 
    self.device    = device 
    
  def __call__(self, sentences: List[str]) -> Tensor: 
    inputs = self.tokenizer(sentences, 
                            truncation = True, 
                            padding = "max_length", 
                            return_tensors = "pt") 
    for key in inputs: 
      if isinstance(inputs[key], Tensor): 
        inputs[key] = inputs[key].to(self.device) 
    out = self.model(**inputs) 
    mask = inputs["attention_mask"] 
        
    return out.last_hidden_state[:, 0, :].detach().cpu() 
```


# Load Data into PyTorch Frame Dataset class 

&nbsp; Data in PyTorch Frame are stored in TensorFrame. For each column, we need to specify its semantic type. 

&nbsp; Clearly, `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts` and `EstimatedSalary` are numerical features and `Geography`, `HasCrCard` and `IsActiveMember` are categorical features. But what about `Surname`. One can argue thet It's categorical, but it is also text and can contain important demographic information. 



```python
from torch_frame import numerical, categorical, text_embedded, embedding 
from torch_frame.data import Dataset, DataLoader 
from torch_frame.config.text_embedder import TextEmbedderConfig 

text_encoder = TextToEmbedding(model = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device = device) 

col_to_stype = {"CustomerId"      : numerical, 
                "Surname"         : text_embedded, 
                "CreditScore"     : numerical, 
                "Geography"       : categorical, 
                "Age"             : numerical, 
                "Tenure"          : numerical, 
                "Balance"         : numerical, 
                "NumOfProducts"   : numerical, 
                "HasCrCard"       : categorical, 
                "IsActiveMember"  : categorical, 
                "EstimatedSalary" : numerical} 
test_dataset = Dataset(df = pd.read_csv("./data/test.csv"), 
                       col_to_stype = col_to_stype, 
                       col_to_text_embedder_cfg = TextEmbedderConfig(text_embedder = text_encoder, batch_size = 32))
col_to_stype = col_to_stype.copy() 
col_to_stype['Exited'] = categorical 
dataset = Dataset(df = pd.read_csv("./data/train.csv"), 
                  col_to_stype = col_to_stype, 
                  target_col = "Exited", 
                  col_to_text_embedder_cfg = TextEmbedderConfig(text_embedder = text_encoder, batch_size = 32))
```

&nbsp; Now we need to materialize the datasets. Materialization means calculating column stats and generate embeddings for the text column. 


```python
dataset.materialize(path = "./data/data_3.pt")
#> Dataset()
test_dataset.materialize(path = "./data/test_data_3.pt")
#> Dataset()
```


# Declaring Model in PyTorch Frame 

&nbsp; In this notebook, we will be using [FT Transformer](https://arxiv.org/pdf/2106.11959.pdf). We have a variety of other models offered to use directly including but not limited to TabNet, ResNet and Trompt. 

&nbsp; In Pytorch Frame, you need to declare the encoding method for different semantic types for different models. 

&nbsp; Note that the parameters of the model are generated from [Hyperparameter Search using Optuna](https://www.kaggle.com/code/yiwenyuan1998/tabular-deep-learning-with-pytorch-frame#hyperparameter_search_using_optuna). We did a 1:1 split on the full dataset for training and validation. After we found the best set of parameters, we refit on the full dataset. 


```python
from torch.optim.lr_scheduler import ExponentialLR 
from torch_frame import NAStrategy 

from torch_frame.nn import (
  EmbeddingEncoder, 
  FTTransformer, 
  LinearEmbeddingEncoder, 
  LinearBucketEncoder 
)

def create_model(): 
  stype_encoder_dict = {
    categorical: EmbeddingEncoder(na_strategy = NAStrategy.MOST_FREQUENT), 
    numerical: LinearBucketEncoder(na_strategy = NAStrategy.MEAN), 
    embedding: LinearEmbeddingEncoder() 
  }
  
  model = FTTransformer(channels = 128, 
                        num_layers = 8, 
                        out_channels = 1, 
                        col_stats = dataset.col_stats, 
                        col_names_dict = dataset.tensor_frame.col_names_dict, 
                        stype_encoder_dict = stype_encoder_dict).to(device)

  return model 
```


# Training 

&nbsp; Now let's train the model. 

&nbsp; We first declare the `train` and `test` function. 


```python
from torch.nn import Module 
from torch.nn import BCEWithLogitsLoss 
from torchmetrics import AUROC 

loss_func = BCEWithLogitsLoss() 

def train(model: Module, loader: DataLoader, optimizer: torch.optim.Optimizer, epoch: int) -> float: 
  model.train() 
  loss_accum = total_count = 0 
  for tf in tqdm(loader, desc = f'Epoch: {epoch}', disable=True): 
    tf = tf.to(device) 
    pred = model.forward(tf) 
    loss = loss_func(pred, tf.y.view(-1, 1).to(torch.float32)) 
    optimizer.zero_grad() 
    loss.backward() 
    loss_accum += float(loss) * len(tf.y) 
    total_count += len(tf.y) 
    optimizer.step() 
  return loss_accum / total_count 

@torch.no_grad() 
def test(model: Module, loader: DataLoader) -> float: 
  model.eval() 
  all_preds = [] 
  all_labels = [] 
  metric_computer = AUROC(task = 'binary').to(device) 
  for tf in loader: 
    tf = tf.to(device) 
    pred = model(tf) 
    metric_computer.update(pred, tf.y) 
    
  return metric_computer.compute().item() 
```

&nbsp; Now Let's train the model. From experince, the model converges within only a few epochs. 


```python
import time 
# Let's train the model 
metric = 'Auc' 
epochs = 20 
models = [] 
best_models = [] 

for i in range(2): 
  print(f"Running Trial {i}") 
  model = create_model() 
  best_val_metric = 0 
  optimizer = torch.optim.Adam(model.parameters(), lr = 0.0004541334318052064) 
  lr_scheduler = ExponentialLR(optimizer, gamma = 0.8094320330407814) 
  
  if i == 0: 
    val_dataset, train_dataset = dataset[:0.1], dataset[0.1:] 
  else: 
    train_dataset, val_dataset = dataset[:0.9], dataset[0.9:] 
  
  train_loader = DataLoader(train_dataset.tensor_frame, batch_size = 256, shuffle = True, drop_last = True) 
  val_loader = DataLoader(val_dataset.tensor_frame, batch_size = 256) 
  
  for epoch in range(1, epochs + 1): 
    train_loss = train(model, train_loader, optimizer, epoch) 
    train_metric = test(model, train_loader) 
    val_metric = test(model, val_loader) 
    if val_metric > best_val_metric: 
      best_model = model.state_dict() 
      best_val_metric = val_metric 
    lr_scheduler.step() 
    print(f"Train Loss: {train_loss:.4f}, train {metric}: {train_metric:.4f}, "
          f"Val {metric}: {val_metric:.4f}")
  models.append(model) 
  best_models.append(best_model) 
#> Running Trial 0
#> Train Loss: 0.3783, train Auc: 0.8814, Val Auc: 0.8789
#> Train Loss: 0.3366, train Auc: 0.8858, Val Auc: 0.8829
#> Train Loss: 0.3309, train Auc: 0.8871, Val Auc: 0.8842
#> Train Loss: 0.3279, train Auc: 0.8879, Val Auc: 0.8839
#> Train Loss: 0.3255, train Auc: 0.8892, Val Auc: 0.8853
#> Train Loss: 0.3247, train Auc: 0.8898, Val Auc: 0.8857
#> Train Loss: 0.3237, train Auc: 0.8903, Val Auc: 0.8858
#> Train Loss: 0.3224, train Auc: 0.8911, Val Auc: 0.8864
#> Train Loss: 0.3214, train Auc: 0.8914, Val Auc: 0.8865
#> Train Loss: 0.3203, train Auc: 0.8916, Val Auc: 0.8865
#> Train Loss: 0.3201, train Auc: 0.8919, Val Auc: 0.8867
#> Train Loss: 0.3192, train Auc: 0.8923, Val Auc: 0.8867
#> Train Loss: 0.3189, train Auc: 0.8924, Val Auc: 0.8868
#> Train Loss: 0.3183, train Auc: 0.8928, Val Auc: 0.8869
#> Train Loss: 0.3180, train Auc: 0.8929, Val Auc: 0.8868
#> Train Loss: 0.3176, train Auc: 0.8930, Val Auc: 0.8868
#> Train Loss: 0.3174, train Auc: 0.8930, Val Auc: 0.8869
#> Train Loss: 0.3175, train Auc: 0.8931, Val Auc: 0.8868
#> Train Loss: 0.3175, train Auc: 0.8932, Val Auc: 0.8869
#> Train Loss: 0.3172, train Auc: 0.8933, Val Auc: 0.8870
#> Running Trial 1
#> Train Loss: 0.3732, train Auc: 0.8824, Val Auc: 0.8795
#> Train Loss: 0.3362, train Auc: 0.8856, Val Auc: 0.8824
#> Train Loss: 0.3303, train Auc: 0.8870, Val Auc: 0.8834
#> Train Loss: 0.3278, train Auc: 0.8873, Val Auc: 0.8844
#> Train Loss: 0.3257, train Auc: 0.8891, Val Auc: 0.8854
#> Train Loss: 0.3242, train Auc: 0.8884, Val Auc: 0.8852
#> Train Loss: 0.3229, train Auc: 0.8905, Val Auc: 0.8867
#> Train Loss: 0.3217, train Auc: 0.8909, Val Auc: 0.8871
#> Train Loss: 0.3208, train Auc: 0.8915, Val Auc: 0.8873
#> Train Loss: 0.3204, train Auc: 0.8915, Val Auc: 0.8871
#> Train Loss: 0.3195, train Auc: 0.8919, Val Auc: 0.8872
#> Train Loss: 0.3193, train Auc: 0.8921, Val Auc: 0.8873
#> Train Loss: 0.3188, train Auc: 0.8923, Val Auc: 0.8873
#> Train Loss: 0.3180, train Auc: 0.8923, Val Auc: 0.8873
#> Train Loss: 0.3180, train Auc: 0.8926, Val Auc: 0.8871
#> Train Loss: 0.3176, train Auc: 0.8928, Val Auc: 0.8874
#> Train Loss: 0.3178, train Auc: 0.8928, Val Auc: 0.8875
#> Train Loss: 0.3174, train Auc: 0.8930, Val Auc: 0.8874
#> Train Loss: 0.3174, train Auc: 0.8930, Val Auc: 0.8874
#> Train Loss: 0.3173, train Auc: 0.8930, Val Auc: 0.8874
```


# Predict with the Trained Model 

```python
@torch.no_grad() 
def predict(model: Module, loader: DataLoader) -> float: 
  model.eval() 
  all_preds = [] 
  for tf in loader: 
    tf = tf.to(device) 
    pred = model(tf) 
    
    all_preds.append(pred) 
  all_preds = torch.cat(all_preds).cpu() 
  return all_preds 

models[0].load_state_dict(best_models[0]) 
#> <All keys matched successfully>
models[1].load_state_dict(best_models[1]) 
#> <All keys matched successfully>

test_loader = DataLoader(test_dataset.tensor_frame, batch_size = 256) 
submission = pd.read_csv("./data/sample_submission.csv") 

submission['Exited'] = (predict(models[0], test_loader).numpy() + predict(models[1], test_loader).numpy()) / 2

submission.to_csv("./data/submission.csv", index = False)
```


# Hyperparameter Search using Optuna 

&nbsp; In this section, we use optuna to tune our model. 


```python
import optuna 

from torch_frame import NAStrategy 
from torch_frame.nn import(
  EmbeddingEncoder, 
  FTTransformer, 
  LinearEmbeddingEncoder, 
  LinearEncoder, 
  LinearBucketEncoder, 
  ExcelFormerEncoder
)

epochs = 20 
continuous = ['base_lr', 'gamma_rate'] 

encoder_search_space = {
  'numerical_encoder': ['LinearEncoder', 'LinearBucketEncoder', 'ExcelFormerEncoder'], 
  'numerical_na_strategy' : ['mean', 'zeros']
}

model_search_space = {
  'channels' : [128, 256], 
  'num_layers' : [2, 4, 8], 
}

train_search_space = {
  'batch_size' : [512, 256, 128], 
  'base_lr' : [1e-4, 1e-3], 
  'gamma_rate': [0.7, 1.] 
}

TRAIN_CONFIG_KEYS = ["batch_size", "gamma_rate", "base_lr"] 

# Split the train data into training and validation set 
train_dataset, val_dataset = dataset[:0.9], dataset[0.9:] 

def objective(trial: optuna.trial.Trial) -> float: 
  encoder_cfg = {} 
  for name, search_list in encoder_search_space.items(): 
    encoder_cfg[name] = trial.suggest_categorical(name, search_list) 
  model_cfg = {} 
  for name, search_list in model_search_space.items(): 
    if name not in continuous: 
      model_cfg[name] = trial.suggest_categorical(name, search_list) 
    else: 
      model_cfg[name] = trial.suggest_float(name, saerch_list[0], search_slit[1]) 
  train_cfg = {} 
  for name, search_list in train_search_space.items(): 
    if name not in continuous: 
      train_cfg[name] = trial.suggest_categorical(name, search_list) 
    else: 
      train_cfg[name] = trial.suggest_float(name, search_list[0], search_list[1]) 
      
  best_val_metric = train_and_eval_with_cfg(encoder_cfg = encoder_cfg, 
                                            model_cfg = model_cfg, 
                                            train_cfg = train_cfg, 
                                            trial = trial) 
  return best_val_metric 

def train_and_eval_with_cfg(encoder_cfg, model_cfg, train_cfg, trial: Optional[optuna.trial.Trial] = None): 
  if encoder_cfg['numerical_encoder'] == 'LinearEncoder': 
    numerical_encoder_cls = LinearEncoder 
  elif encoder_cfg['numerical_encoder'] == 'LinearBucketEncoder': 
    numerical_encoder_cls = LinearBucketEncoder 
  else: 
    numerical_encoder_cls = ExcelFormerEncoder 
  stype_encoder_dict = {
    categorical: EmbeddingEncoder(na_strategy = NAStrategy.MOST_FREQUENT), 
    numerical: numerical_encoder_cls(na_strategy = NAStrategy(encoder_cfg['numerical_na_strategy'])), 
    embedding: LinearEmbeddingEncoder() 
  }
  
  model = FTTransformer(
    **model_cfg, 
    out_channels = 1, 
    col_stats = train_dataset.col_stats, 
    col_names_dict = train_dataset.tensor_frame.col_names_dict, 
    stype_encoder_dict = stype_encoder_dict
  ).to(device) 
  model.reset_parameters() 
  # Use train_cfg to set up training procedure 
  optimizer = torch.optim.Adam(model.parameters(), lr = train_cfg['base_lr']) 
  lr_scheduler = ExponentialLR(optimizer, gamma = train_cfg['gamma_rate']) 
  train_loader = DataLoader(train_dataset.tensor_frame, batch_size = train_cfg['batch_size'], shuffle = True, drop_last = True) 
  val_loader = DataLoader(val_dataset.tensor_frame, batch_size = train_cfg['batch_size']) 
  
  best_val_metric = 0 
  
  for epoch in range(1, epochs + 1): 
    train_loss = train(model, train_loader, optimizer, epoch) 
    val_metric = test(model, val_loader) 
    
    if val_metric > best_val_metric: 
      best_val_metric = val_metric 
      
    lr_scheduler.step() 
    print(f"Train Loss: {train_loss:.4f}, Val: {val_metric:.4f}") 
    
    if trial is not None: 
      trial.report(val_metric, epoch) 
      if trial.should_prune(): 
        raise optuna.TrialPruned() 
      
  print(f'Best val: {best_val_metric:.4f}') 
  return best_val_metric
```


# Uncomment the following section to run the full hyperparameter search. 


```python
# Hyper00parameter optimization with Optuna 
print("Hyper-parameter search via Optuna") 
#> Hyper-parameter search via Optuna
study = optuna.create_study(pruner = optuna.pruners.MedianPruner(), 
                            direction = "maximize") 
#> [I 2024-01-12 00:55:10,411] A new study created in memory with name: no-name-316f326a-9245-4d6f-86d9-ef3398f8e205

study.optimize(objective, n_trials = 5) 
#> Train Loss: 0.3637, Val: 0.8790
#> Train Loss: 0.3349, Val: 0.8832
#> Train Loss: 0.3301, Val: 0.8845
#> Train Loss: 0.3283, Val: 0.8852
#> Train Loss: 0.3268, Val: 0.8843
#> Train Loss: 0.3264, Val: 0.8865
#> Train Loss: 0.3250, Val: 0.8859
#> Train Loss: 0.3248, Val: 0.8848
#> Train Loss: 0.3236, Val: 0.8858
#> Train Loss: 0.3229, Val: 0.8871
#> Train Loss: 0.3224, Val: 0.8860
#> Train Loss: 0.3222, Val: 0.8862
#> Train Loss: 0.3216, Val: 0.8857
#> Train Loss: 0.3210, Val: 0.8865
#> Train Loss: 0.3209, Val: 0.8872
#> Train Loss: 0.3204, Val: 0.8848
#> Train Loss: 0.3198, Val: 0.8875
#> Train Loss: 0.3197, Val: 0.8872
#> Train Loss: 0.3191, Val: 0.8864
#> Train Loss: 0.3188, Val: 0.8874
#> Best val: 0.8875
#> Train Loss: 0.3942, Val: 0.8530
#> Train Loss: 0.3559, Val: 0.8797
#> Train Loss: 0.3318, Val: 0.8835
#> Train Loss: 0.3273, Val: 0.8846
#> Train Loss: 0.3262, Val: 0.8844
#> Train Loss: 0.3245, Val: 0.8857
#> Train Loss: 0.3232, Val: 0.8840
#> Train Loss: 0.3221, Val: 0.8852
#> Train Loss: 0.3212, Val: 0.8849
#> Train Loss: 0.3202, Val: 0.8851
#> Train Loss: 0.3188, Val: 0.8862
#> Train Loss: 0.3180, Val: 0.8859
#> Train Loss: 0.3168, Val: 0.8855
#> Train Loss: 0.3157, Val: 0.8861
#> Train Loss: 0.3148, Val: 0.8857
#> Train Loss: 0.3137, Val: 0.8846
#> Train Loss: 0.3129, Val: 0.8857
#> Train Loss: 0.3122, Val: 0.8848
#> Train Loss: 0.3111, Val: 0.8846
#> Train Loss: 0.3110, Val: 0.8842
#> Best val: 0.8862
#> Train Loss: 0.4464, Val: 0.8219
#> Train Loss: 0.3599, Val: 0.8734
#> Train Loss: 0.3401, Val: 0.8792
#> Train Loss: 0.3356, Val: 0.8750
#> Train Loss: 0.3330, Val: 0.8813
#> Train Loss: 0.3302, Val: 0.8828
#> Train Loss: 0.3284, Val: 0.8829
#> Train Loss: 0.3265, Val: 0.8843
#> Train Loss: 0.3251, Val: 0.8851
#> Train Loss: 0.3246, Val: 0.8853
#> Train Loss: 0.3234, Val: 0.8834
#> Train Loss: 0.3232, Val: 0.8859
#> Train Loss: 0.3224, Val: 0.8859
#> Train Loss: 0.3219, Val: 0.8863
#> Train Loss: 0.3214, Val: 0.8865
#> Train Loss: 0.3212, Val: 0.8869
#> Train Loss: 0.3205, Val: 0.8869
#> Train Loss: 0.3203, Val: 0.8867
#> Train Loss: 0.3198, Val: 0.8867
#> Train Loss: 0.3193, Val: 0.8867
#> Best val: 0.8869
#> Train Loss: 0.3709, Val: 0.8773
#> Train Loss: 0.3373, Val: 0.8823
#> Train Loss: 0.3334, Val: 0.8848
#> Train Loss: 0.3308, Val: 0.8849
#> Train Loss: 0.3281, Val: 0.8834
#> Train Loss: 0.3266, Val: 0.8858
#> Train Loss: 0.3255, Val: 0.8856
#> Train Loss: 0.3239, Val: 0.8857
#> Train Loss: 0.3229, Val: 0.8867
#> Train Loss: 0.3223, Val: 0.8862
#> Train Loss: 0.3210, Val: 0.8867
#> Train Loss: 0.3205, Val: 0.8868
#> Train Loss: 0.3198, Val: 0.8872
#> Train Loss: 0.3195, Val: 0.8869
#> Train Loss: 0.3187, Val: 0.8865
#> Train Loss: 0.3185, Val: 0.8871
#> Train Loss: 0.3179, Val: 0.8869
#> Train Loss: 0.3178, Val: 0.8871
#> Train Loss: 0.3174, Val: 0.8867
#> Train Loss: 0.3172, Val: 0.8868
#> Best val: 0.8872
#> Train Loss: 0.4067, Val: 0.8509
#> Train Loss: 0.3690, Val: 0.8643
#> Train Loss: 0.3479, Val: 0.8778
#> Train Loss: 0.3344, Val: 0.8827
#> Train Loss: 0.3296, Val: 0.8841
#> Train Loss: 0.3273, Val: 0.8856
#> Train Loss: 0.3261, Val: 0.8843
#> Train Loss: 0.3254, Val: 0.8862
#> Train Loss: 0.3238, Val: 0.8855
#> Train Loss: 0.3234, Val: 0.8860
#> Train Loss: 0.3229, Val: 0.8856
#> Train Loss: 0.3220, Val: 0.8864
#> Train Loss: 0.3217, Val: 0.8866
#> Train Loss: 0.3206, Val: 0.8859
#> Train Loss: 0.3202, Val: 0.8860
#> Train Loss: 0.3195, Val: 0.8866
#> Train Loss: 0.3192, Val: 0.8859
#> Train Loss: 0.3189, Val: 0.8868
#> Train Loss: 0.3185, Val: 0.8868
#> Train Loss: 0.3175, Val: 0.8867
#> Best val: 0.8868
#> 
#> [I 2024-01-12 00:58:01,525] Trial 0 finished with value: 0.8874854445457458 and parameters: {'numerical_encoder': 'LinearBucketEncoder', 'numerical_na_strategy': 'zeros', 'channels': 128, 'num_layers': 2, 'batch_size': 128, 'base_lr': 0.00036363988058360654, 'gamma_rate': 0.9912428316979148}. Best is trial 0 with value: 0.8874854445457458.
#> [I 2024-01-12 01:00:18,649] Trial 1 finished with value: 0.8862290382385254 and parameters: {'numerical_encoder': 'LinearEncoder', 'numerical_na_strategy': 'zeros', 'channels': 256, 'num_layers': 4, 'batch_size': 256, 'base_lr': 0.0002993354108566624, 'gamma_rate': 0.9253795068417283}. Best is trial 0 with value: 0.8874854445457458.
#> [I 2024-01-12 01:03:49,173] Trial 2 finished with value: 0.8869168758392334 and parameters: {'numerical_encoder': 'ExcelFormerEncoder', 'numerical_na_strategy': 'mean', 'channels': 256, 'num_layers': 8, 'batch_size': 512, 'base_lr': 0.0006583642617087316, 'gamma_rate': 0.8693857152100075}. Best is trial 0 with value: 0.8874854445457458.
#> [I 2024-01-12 01:09:39,145] Trial 3 finished with value: 0.8871639370918274 and parameters: {'numerical_encoder': 'LinearBucketEncoder', 'numerical_na_strategy': 'mean', 'channels': 128, 'num_layers': 8, 'batch_size': 128, 'base_lr': 0.000600676150393692, 'gamma_rate': 0.8506214745568763}. Best is trial 0 with value: 0.8874854445457458.
#> [I 2024-01-12 01:10:57,721] Trial 4 finished with value: 0.8868314027786255 and parameters: {'numerical_encoder': 'LinearEncoder', 'numerical_na_strategy': 'zeros', 'channels': 128, 'num_layers': 2, 'batch_size': 256, 'base_lr': 0.00026059399114586443, 'gamma_rate': 0.9627774437045109}. Best is trial 0 with value: 0.8874854445457458.

print("Hyper-parameter search done. Found the best config.") 
#> Hyper-parameter search done. Found the best config.
params = study.best_params 
best_train_cfg = {} 
for train_cfg_key in TRAIN_CONFIG_KEYS: 
  best_train_cfg[train_cfg_key] = params.pop(train_cfg_key) 
best_model_cfg = params

print(f"Best train config: {best_train_cfg}, Best model config: {best_model_cfg}")
#> Best train config: {'batch_size': 128, 'gamma_rate': 0.9912428316979148, 'base_lr': 0.00036363988058360654}, Best model config: {'numerical_encoder': 'LinearBucketEncoder', 'numerical_na_strategy': 'zeros', 'channels': 128, 'num_layers': 2}
```
































































