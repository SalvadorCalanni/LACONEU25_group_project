import thoi
from thoi.measures.gaussian_copula import multi_order_measures, nplets_measures
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import os
torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))


model_dirs = os.listdir("../networks_24/")

out = np.load('mods_24.npy', allow_pickle=True).item()
out.keys()

# join and organize the data
alldata = {}
for model in model_dirs:
    modelh = {}
    for task in out[model].keys():
        #print(f'{model}_{task}')
        hlist = []
        for i in range(4):
            hmat = out[model][task]['h'][i]
            hjoinmatrix = np.swapaxes(hmat[:,:,:],0,1).reshape(-1,hmat.shape[-1]).T
            hlist.append(hjoinmatrix)
        allh = np.concatenate(hlist, axis=1)
        #print(task, allh.shape)
        modelh[task] = allh
    alldata[model] = modelh
shorterlength = min([alldata[model][task].shape[1] for model in model_dirs for task in alldata[model].keys()])


all_meanhoi = []

for model in model_dirs:
    for task in alldata[model].keys():
        print(model, task, alldata[model][task].shape)
        htensor = torch.tensor(alldata[model][task].T, dtype=torch.float32, device='cpu')
        hoidata = multi_order_measures(alldata[model][task][:,:shorterlength].T, min_order=1, max_order=6)
        meanhoi = hoidata.groupby('order').mean()[["o", "s", "tc", "dtc"]]
        # Add model and task columns
        meanhoi['model'] = model
        meanhoi['task'] = task
        all_meanhoi.append(meanhoi)

# Concatenate all meanhoi dataframes
concatenated_meanhoi = pd.concat(all_meanhoi)
#save the concat dataframe
concatenated_meanhoi = concatenated_meanhoi.reset_index()
concatenated_meanhoi.to_csv('concatenated_meanhoi_trimmed.csv')