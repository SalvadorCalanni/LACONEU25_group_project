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
    #if meanhoi file exists, mean is false
    if os.path.exists('concatenated_meanhoi_24_trimmed.csv'):
        mean = False
    for model in model_dirs:
        for task in alldata[model].keys():
            if "SLURM_ARRAY_TASK_ID" in os.environ and counter != int(os.environ["SLURM_ARRAY_TASK_ID"]):
                counter+=1
                continue
            print(model, task, alldata[model][task].shape)
            #if file exists loads it
            if os.path.exists(f'{hoi_dirs}{model}_{task}_{max_order}.csv.zip'):
                print(f'{hoi_dirs}{model}_{task}_{max_order}.csv.zip exists')
                hoidata = pd.read_csv(f'{hoi_dirs}{model}_{task}_{max_order}.csv.zip', compression='zip')
            else:
                htensor = torch.tensor(alldata[model][task].T, dtype=torch.float32, device=usedDevice)
                hoidata = multi_order_measures(alldata[model][task].T, min_order=3, max_order=max_order)
                # Save the mean hoi
                if mean:
                    meanhoi = hoidata.groupby('order').mean()[["o", "s", "tc", "dtc"]]
                    meanhoi['model'] = model
                    meanhoi['task'] = task
                    meanhoi.to_csv(f'{hoi_dirs}{model}_{task}_{max_order}_mean.csv')
                if raw:
                    hoidata['model'] = model
                    hoidata['task'] = task
                    hoidata.to_csv(f'{hoi_dirs}{model}_{task}_{max_order}.csv.zip', index=False, compression='zip')
                
            counter+=1
            
    #if mean:
        # Concatenate all meanhoi dataframes
        #concatenated_meanhoi = pd.concat(all_meanhoi)
        #save the concat dataframe
        #concatenated_meanhoi = concatenated_meanhoi.reset_index()
        #concatenated_meanhoi.to_csv('concatenated_meanhoi_trimmed.csv')

def mean_hois(hoi_dirs, max_order):
    if os.path.exists(f'concatenated_meanhoi_24_trimmed_{max_order}.csv'):
        return
    all_meanhoi = []
    for hoifile in os.listdir(hoi_dirs): # ex: laconeu_contextdelaydm1_dmcgo_24_dmcgo_8.csv.zip
        print(hoifile)
        hoidata = pd.read_csv(f'{hoi_dirs}{hoifile}', compression='zip')
        meanhoi = hoidata.groupby(['order','task','model']).mean()[["o", "s", "tc", "dtc"]]
        all_meanhoi.append(meanhoi)
    # Concatenate all meanhoi dataframes
    concatenated_meanhoi = pd.concat(all_meanhoi)
    concatenated_meanhoi = concatenated_meanhoi.reset_index()
    concatenated_meanhoi.to_csv(f'concatenated_meanhoi_24_trimmed_{max_order}.csv')

def neuron_hois(hoi_dirs):
    for model in os.listdir(hoi_dirs):
        if model.endswith('.csv'):
            continue
        #if file exists ommit 
        if os.path.exists(f'{hoi_dirs}{model.split(".")[0]}_pernode.csv'):
            print("file exists")
            print(f'{hoi_dirs}{model.split(".")[0]}_pernode.csv')
            continue
        
        hoidata = pd.read_csv(f'{hoi_dirs}{model}', compression='zip')
        means_list = []
        metrics = ["tc","dtc","o","s"]
        for i in range(24):
            var_col = f"var_{i}"
            filtered = hoidata[hoidata[var_col] == True]
            grouped_means = filtered.groupby("order")[metrics].mean().reset_index()
            grouped_means["node"] = i
            means_list.append(grouped_means)
        pernodedf = pd.concat(means_list, ignore_index=True)
        newname = f'{model.split(".")[0]}_pernode.csv'
        pernodedf.to_csv(f'{hoi_dirs}{newname}', index=False)
    print("done")
    

def joinnodemeans(hoi_dir):
    #load mean hoi files and concatenate the dataframes
    all_meanhoi = []
    for hoifile in os.listdir(hoi_dir):
        #if it ends in csv load it
        if hoifile.endswith('_pernode.csv'):
            hoidata = pd.read_csv(f'{hoi_dir}{hoifile}')
            all_meanhoi.append(hoidata)
    concatenated_meanhoi = pd.concat(all_meanhoi)
    #concatenated_meanhoi = concatenated_meanhoi.reset_index()
    concatenated_meanhoi.to_csv('concatenated_meanhoi_all_node_24.csv', index=False)
    
def joinmeans(hoi_dir):
    #load mean hoi files and concatenate the dataframes
    all_meanhoi = []
    for hoifile in os.listdir(hoi_dir):
        #if it ends in csv load it
        if hoifile.endswith('.csv'):
            hoidata = pd.read_csv(f'{hoi_dir}{hoifile}')
            all_meanhoi.append(hoidata)
    concatenated_meanhoi = pd.concat(all_meanhoi)
    #concatenated_meanhoi = concatenated_meanhoi.reset_index()
    concatenated_meanhoi.to_csv('concatenated_meanhoi_all_24.csv', index=False)


if __name__ == '__main__':
    model_dirs = os.listdir("../networks_24/")
    hoi_dirs = "../hois_24/"
    data_dir = "../"
    data_filename = "mods_24_all.npy"

    max_order = 8
    
    #if not os.path.exists(data_dir+data_filename[:-4]+'_cropped.npy'):
    #    cut_to_shortest(data_dir, data_filename)
    #else:
    #    alldata = np.load(data_dir+data_filename[:-4]+'_cropped.npy', allow_pickle=True).item()
    
    #compute_hois(alldata, model_dirs, hoi_dirs, max_order, True, True)
    #mean_hois(hoi_dirs, max_order)
    #joinmeans(hoi_dirs)
    #neuron_hois(hoi_dirs)
    #joinnodemeans(hoi_dirs)
