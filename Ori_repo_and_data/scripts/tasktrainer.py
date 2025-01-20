#alltask trainer
import train
from analysis import performance

import matplotlib.pyplot as plt
import tools
import numpy as np
from task import generate_trials, rule_name, get_dist

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU
os.environ['TF_NUM_INTEROP_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'

import tensorflow as tf

# Configure TF to use single thread
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)


def trainpair(task1,task2,hnodes=32,netfolder="networks/"):
    netname = "_".join(["laconeu",task1,task2,str(hnodes)])
    train.train(model_dir=netfolder+netname, 
            hp={'learning_rate': 0.001, 
                'n_rnn': hnodes,#512, 16384,8192,1024
                # 'w_rec_init': 'randgauss',#'randortho'
                # 'b_rec_init': 'uniform',
                'rule_strength': 1.0,
                'no_rule': False,
                'target_perf':0.8,
                'activation': 'softplus',
                'alpha':0.2},
            ruleset='all',
            rule_trains = [task1,task2],
            trainables='all')#,trainables='bias')


tasks = ['fdgo', 'reactgo', 'delaygo', 'fdanti', 'reactanti', 'delayanti',
            'delaydm1', 'delaydm2', 'contextdelaydm1', 'contextdelaydm2', 'multidelaydm',
            'dmsgo', 'dmsnogo', 'dmcgo', 'dmcnogo']
excluded = ['dm1', 'dm2', 'contextdm1', 'contextdm2','multidm']

# pair task iteration
counter = 0 
for i in range(len(tasks)):
    for j in range(i+1,len(tasks)):
        if "SLURM_ARRAY_TASK_ID" in os.environ and counter != int(os.environ["SLURM_ARRAY_TASK_ID"]):
            counter+=1
            continue
        print("SLURM_ARRAY_TASK_ID" in os.environ)
        print(counter)
        print(tasks[i],tasks[j])
        counter+=1
        trainpair(tasks[i],tasks[j],hnodes=24,netfolder="../networks_24/")
