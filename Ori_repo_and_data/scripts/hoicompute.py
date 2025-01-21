import thoi
from thoi.measures.gaussian_copula import multi_order_measures, nplets_measures
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import os

usedDevice = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device(usedDevice)
print('Using device:', usedDevice)


def cut_to_shortest(data_dir, data_filename):
    # Load the data
    hidden = np.load(data_dir+data_filename, allow_pickle=True).item()
    hidden.keys()

    # join and organize the data
    alldata = {}
    for model in model_dirs:
        modelh = {}
        for task in hidden[model].keys():
            #print(f'{model}_{task}')
            hlist = []
            for i in range(4):
                hmat = hidden[model][task]['h'][i]
                hjoinmatrix = np.swapaxes(hmat[:,:,:],0,1).reshape(-1,hmat.shape[-1]).T
                hlist.append(hjoinmatrix)
            allh = np.concatenate(hlist, axis=1)
            #print(task, allh.shape)
            modelh[task] = allh
        alldata[model] = modelh
    shorterlength = min([alldata[model][task].shape[1] for model in model_dirs for task in alldata[model].keys()])
    #cut the data to the shortest length
    for model in model_dirs:
        for task in alldata[model].keys():
            alldata[model][task] = alldata[model][task][:,:shorterlength]
    #save the data with _cropped suffix
    np.save(data_dir+data_filename[:-4]+'_cropped.npy', alldata)




def compute_hois(alldata, model_dirs, hoi_dirs, max_order = 6, mean = True, raw = False):
    counter = 0
    all_meanhoi = []
    for model in model_dirs:
        for task in alldata[model].keys():
            if "SLURM_ARRAY_TASK_ID" in os.environ and counter != int(os.environ["SLURM_ARRAY_TASK_ID"]) and mean!=True:
                counter+=1
                continue
            print(model, task, alldata[model][task].shape)
            htensor = torch.tensor(alldata[model][task].T, dtype=torch.float32, device=usedDevice)
            hoidata = multi_order_measures(alldata[model][task].T, min_order=3, max_order=max_order)
            
            if raw:
                hoidata['model'] = model
                hoidata['task'] = task
                #save the raw data in efficient way as csv zip
                hoidata.to_csv(f'{hoi_dirs}{model}_{task}_{max_order}.csv.zip', index=False, compression='zip')
                
            if mean:
            # Compute the mean hoi
                meanhoi = hoidata.groupby('order').mean()[["o", "s", "tc", "dtc"]]
                meanhoi['model'] = model
                meanhoi['task'] = task
                # Save the hoidata
                all_meanhoi.append(meanhoi)
            counter+=1
            
    if mean:
        # Concatenate all meanhoi dataframes
        concatenated_meanhoi = pd.concat(all_meanhoi)
        #save the concat dataframe
        concatenated_meanhoi = concatenated_meanhoi.reset_index()
        concatenated_meanhoi.to_csv('concatenated_meanhoi_trimmed.csv')


if __name__ == '__main__':
    model_dirs = os.listdir("../networks_24/")
    hoi_dirs = "../hois_24/"
    data_dir = "../"
    data_filename = "eval_24.npy"

    max_order = 6
    
    if not os.path.exists(data_dir+data_filename[:-4]+'_cropped.npy'):
        cut_to_shortest()
    else:
        alldata = np.load(data_dir+data_filename[:-4]+'_cropped.npy', allow_pickle=True).item()
    
    compute_hois(alldata, model_dirs, hoi_dirs, max_order, False, True)