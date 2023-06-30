#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, torch, os, random
import numpy as np

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    #torch.set_deterministic(True)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def read(path):
    data=[]
    with open(path, "r") as f:
        for line in f.readlines():
            temp = line.split()
            if temp!=[]:
                data.append(temp)
    return data