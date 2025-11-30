#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 24 09:32:41 2025

@author: JohannesSalge

This is a sketch of the layer 6b being a mediator of attention. 
In training this mechanism might be useful as a feature importance ranking me-
chanism.

A Define models of subsystems
B Define connectivity between subsystems 
C Visualize connectivity between subsystem

#
++++++++++++++++++++++++++++++++

""" 
#%% === Intended Structure ===

            # ____________________________
            # |   |   |   |   |   |   |   |   Change Weights
            # |    Attention Weights      |<------------------
            # |___|___|___|___|___|___|___|                   |
            #   |       |       |       |                     |
            #       ( external inputs )                       |
            #   |       |       |       |              _________________
            #  ___     ___     ___     ___            |                 |
            # |   |   |   |   |   |   |   |           |   HoT Module    |
            # |   |   |   |   |   |   |   |           |_________________|
            # |___|   |___|   |___|   |___|                   |   |
            #   |       |       |       |                     |   |
            # ____________________________                    |   |
            # |                           |                   |   |
            # |    Integration Layer      |                   |   |
            # |___________________________|                   |   |
            #      |   |   |   |   |                          |   |
            #      O1  O2  O3  O4  O5 ------------------------    |
            #                                                     |
            #      T1  T2  T3  T4  T5 -----------------------------

#%% === Libraries === 

import pandas as pd
import torch
import torch.nn as nn
#from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score # Performance Statistics
# from sklearn.preprocessing import LabelEncoder 


#%% === load example dataset === 

df = pd.read_csv("example_dataset.csv")


X_np = df.iloc[:, :-1].values   # iputs  
y_np = df.iloc[:, -1].values    # labels

X = torch.tensor(X_np, dtype=torch.float32)
y = torch.tensor(y_np, dtype=torch.long)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
loader_hot = DataLoader(dataset, batch_size=1, shuffle=True)



#%% === define columns (small ANNs) === 
class cortical_column(nn.Module):
    def __init__(self, input_dim = 5, hidden_dim = 5, num_layers = 5, output_dim = 5):
        super().__init__()
        
        layers = []
        # Layer 1 input -> hidden 
        layers.append(nn.Linear(hidden_dim, hidden_dim)) 
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1): 
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())     
        
        # last layer
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        # define attention weights 
        
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

#%% === define attention units for each column (weights) === 
# these weights need to be independent of backprogration during pretraining
# they must be easily changable 
# they must increase the weighting of the output of the respective column during attention
# where do I best define them?
# self.register_buffer("column_weights", torch.ones(4)) 
# model.column_weights = torch.tensor([0.1, 0.0, -0.05, 0.02])      

#%% === define integration layer ===
class integration_layer(nn.Module): 
    def __init__(self, input_dim=20, hidden_dim=10, output_dim=5):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.layers(x)
    
        # to keep it simple the integration layer does not end on a softmax
    
#%% === combine cortical columns and integration layer to a module "classifier" ===
class Classifier(nn.Module): 
    def __init__(self): 
        super().__init__()
        
        self.columns = nn.ModuleList([
            cortical_column() for _ in range(4)
        ])
        
        self.register_buffer("column_weights", torch.ones(4)) # column weights are 1 in pre-training
        
        self.integrator = integration_layer(input_dim = 4 * 5)
        
    def forward(self, x): 
        col_outputs = []
        
        for i in range(4): 
            col_input = x[:, i, :]
            out = self.columns[i](col_input)
            out = out * self.column_weights[i]  # column weights
            col_outputs.append(out)
            
        merged = torch.cat(col_outputs, dim = 1)
        
        return self.integrator(merged)


#%% === define HoT unit === 
# (gets inputs from the classifier-output-layer and tries to minimize un-
# certainty)
# Input: Integration-Layer-Output + Correct Category
#

class HoT_attention(nn.Module): 
    def __init__(self, input_dim = 10, hidden_dim = 20, output_dim = 4):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):                                   # !!!!! The there is nearly no contrast in the learned weights
        return torch.softmax(self.layers(x), dim=-1)
        #return self.layers(x)



#%% === Train classifier unit === 
# model = cortical_column() 
# model_integr = integration_layer()
model_classifier = Classifier()
model_HoT = HoT_attention()


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_classifier.parameters(), lr=0.001)

for epoch in range(150):   # 
    total_loss = 0

    for batch_x, batch_y in loader:
        # reshape flat 20 inputs → 4 columns × 5 features
        batch_x = batch_x.reshape(-1, 4, 5)

        logits = model_classifier(batch_x)
        loss = criterion(logits, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.3f}")

    
#%% ===Evaluation after training of the classifier module === 

    
model_classifier.eval()  # set model to evaluation mode

all_labels = []
all_preds = []

with torch.no_grad():
    for batch_x, batch_y in loader:
        batch_x = batch_x.reshape(-1, 4, 5)

        logits = model_classifier(batch_x)
        preds = torch.argmax(logits, dim=1)

        all_labels.extend(batch_y.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
cm = confusion_matrix(all_labels, all_preds)
print("Confusion Matrix:")
print(cm)

acc = accuracy_score(all_labels, all_preds)
print(f"Accuracy: {acc:.4f}")

sens = recall_score(all_labels, all_preds, average=None)
print("Sensitivity (per class):", sens)


#%% === Train HoT Module === 
# Step 1 Predict using classifier module 
# use classifier output as input for HoT module 
# reward fundction: maximum certainty on correct target
#  


def train_hot_step(classifier, hot, optimizer_hot, x, target):
    
#   classifier  : model_classifier
#    hot        : model_hot
#    x          : single input stimulus, shape (1, 4, 5)
#    target     : one-hot target vector (1, 5)
   


# Step 1: classifier prediction
   
    classifier.eval()    # classifier is frozen here
    with torch.no_grad():
        logits = classifier(x)
        probs = torch.softmax(logits, dim=-1)      # classifier output (1,5)


# Step 2 & 3: build HoT input

    hot_input = torch.cat([probs, target], dim=-1) # shape (1,10)


# Step 4: uncertainty signal
    correct_class_conf = (probs * target).sum()    #


# Reward 
    reward = 1.0 - correct_class_conf.detach()


# Step 5: train HoT to increase certainty

    hot.train()
    optimizer_hot.zero_grad()

    predicted_weights = hot(hot_input)   # (1,4)

    # LOSS: train HoT to output *higher target_column_weights* when reward is large
    # we want: predicted_weights *should correlate with reward*
    loss = -reward * predicted_weights.mean()

    loss.backward()
    optimizer_hot.step()


# Step 6: Update classifier's column weights
    with torch.no_grad():
        classifier.column_weights[:] = predicted_weights[0]

    return float(loss.item()), float(correct_class_conf.item())


#%% === Train HoT unit === 

hot = HoT_attention()
optimizer_hot = torch.optim.Adam(hot.parameters(), lr=0.001)

for i, (x, y) in enumerate(loader_hot):
    
    # 1-hot encoding for the target vector
    target = torch.zeros(1,5)
    target[0, y.item()] = 1.0

    # ensure input shape is correct
    x = x.reshape(-1, 4, 5)

    loss, confidence = train_hot_step(
        classifier=model_classifier,
        hot=hot,
        optimizer_hot=optimizer_hot,
        x=x,
        target=target
    )

    print(f"Stimulus {i}: Loss={loss:.3f}, Correct-class confidence={confidence:.3f}")
    



#%% === Test performance for Classifier + HoT === 
# here I have the problem that the HoT unit received the target on training
# rethink the HoT-architechture! Is there some kind of bootstrapping for this?

# The Problem: HoT unit has 10 inputs (5 classifier-guess outputs, 5 correct categories)
#   1) Target vector could be replaced by the classifier output; 
#   2) Target vector of constants; 
#   3) Target vector could iteratively be changed and the best option could be chosen

# === Iteratively change input (target) vector

target_matrix = torch.tensor(np.diag(np.ones(5)), dtype = torch.float32)

def autonomous_attention(classifier, hot, x, target_matrix): 
    #classifier = model_classifier
    
    confidence_auton = np.empty(len(target_matrix[1, :]))
    neg_entropy = np.empty(len(target_matrix[1, :]))
    margin = np.empty(len(target_matrix[1, :]))
    
    
    # Save intial classifier weights and outputs
    orig_weights = classifier.column_weights.clone()
    logits_initial = classifier(x)
    probs_initial = torch.softmax(logits_initial, dim=-1)
    
    #initial_guess = classifier(x)
    
    for ii in range(len(target_matrix[1, :])):              # this mental Operation could be put in own function,
        target = target_matrix[ii, :]
        target = target.unsqueeze(0)
        
        
        #logits_guess = classifier(x)
        #probs_guess = torch.softmax(logits_guess, dim=-1)
        
        
        hot_input = torch.cat([probs_initial, target], dim=-1)
        predicted_weights = hot(hot_input)
        
        
        with torch.no_grad():
            classifier.column_weights[:] = predicted_weights[0] 
            
        logits_after = classifier(x)
        probs_after = torch.softmax(logits_after, dim=-1)
        
        
        # Decision Criteria 
        confidence_auton[ii] = (probs_after * target).sum()   # 1) probability of the chosen category
        
        entropy = - (probs_after * probs_after.log()).sum()
        neg_entropy[ii] = -entropy                      # 2) negative entropy
        
        probs_after = probs_after.view(-1)
        sorted_probs, _ = probs_after.sort(descending=True)
        margin[ii] = sorted_probs[0] - sorted_probs[1]  # 3) probability contrast 
        
    # Choose Percept
    idx_percept = np.array(confidence_auton).argmax()
    
    classifier.column_weights[:] = orig_weights # this is not needed 
    
    target = target_matrix[idx_percept, :].unsqueeze(0)
    
    hot_input = torch.cat([probs_initial, target], dim=-1) # 
   
    predicted_weights = hot(hot_input)
    
    classifier.column_weights[:] = predicted_weights[0] 
    #print("weights:", classifier.column_weights) 
    with open("column_weights.log", "a") as f:
        f.write(f"Weight: {classifier.column_weights.tolist()}\n")
    
    logits = classifier(x)
    percept = torch.argmax(logits, dim=1)
    
    return [percept, logits]



# Test perfomrance 
y = torch.tensor(y_np, dtype=torch.long) #!!!! where does "y" get lost?
correct_baseline = 0
correct_auton = 0
probs_after = 0

model_baseline = model_classifier
model_baseline.column_weights = torch.ones(4)

for ii in range(len(X[:,1])): 
    
#for ii in range(1,500):
    x_in = X[ii, :].reshape(-1, 4, 5)

    percept, logits_hot = autonomous_attention(
        classifier = model_classifier,
        hot = hot,
        x = x_in, 
        target_matrix = target_matrix
    )
    #print("hot-logits:", logits_hot)
    correct_cat = y[ii]
    correct_auton += (percept == correct_cat).sum().item()
    
    
    model_baseline.column_weights = torch.ones(4)
    logits = model_baseline(x_in)
    #print("ori_logits:", logits)
    preds = torch.argmax(logits, dim=1)
    correct_baseline += (preds == correct_cat).sum().item()
    
print("Correct Baseline:", correct_baseline)
print("Correct Autonomous Attention:", correct_auton)



    

#%% === Thoughts === 

# if this structure had a way to store representations of the input stimuli and
# and their corresponding categories as well as of the noise structure it is 
# commonly presented with, the HoT module could be trained in offline-mode
# (without external stimuli). 1) Take real stimuli in 2) save representation
# 3) use representation to further train attention-allocation.

# Attention might only be needed for difficult cases, that is when unceertainty
# is high. Therefor certainty (e.g. probability contrast) could be checked 
# after classification by the classifier. Only when certainty is below threshold
# the HoT unit could be activatet to increase certainty. 

# In autonomous mode the HoT unit only tries to boost certainty once - however 
# in a continous perception loop it could test if the boost was strong eneugh 
# for a stable percept to arise or if uncertainty is still above threshold. If 
# so it could initiate gathering of further evidence. New evidence could be 
# modeled by feeding new inputs to the classifier. 
    

