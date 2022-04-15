from __future__ import print_function
from __future__ import division
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

def get_distance(vector_a, vector_b):
    """
    计算两个向量之间的余弦相似度
    :param vector_a: 向量 a 
    :param vector_b: 向量 b
    :return: sim
    """
    #print(vector_a.shape,vector_b.shape)
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
#     sim = 0.5 + 0.5 * cos
    return cos

def get_best(x,pattern_vector_dict,hold=0.9):
#     res = []
    score = 0
    best_path = ""
    for k in pattern_vector_dict:
        tmp_score = get_distance(x,pattern_vector_dict[k])
        if tmp_score > score:
            score = tmp_score
            best_path = k
    if score > hold:
        print(best_path,score)
    return best_path,score
    
    
transfrom_tensor =  transforms.Compose([  
             transforms.Resize(250),                    
         transforms.CenterCrop(224),  
         transforms.ToTensor(),                    
         transforms.Normalize(                      
         mean=[0.485, 0.456, 0.406],               
         std=[0.229, 0.224, 0.225]                 
         )])
# ])

import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch

import legacy


device = torch.device('cuda')
with dnnlib.util.open_url("model/ffhq.pkl") as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

import numpy as np
np_res = np.zeros((100,18,512))
for i in range(1):
    z = torch.from_numpy(np.random.randn(1, G.z_dim)).to('cuda')
    ws = G.mapping(z,0)
    img = G(z, 2, truncation_psi=0.5, noise_mode='random')
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#     PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').resize((200,200)).save("result/"+str(i)+".jpg")
    np_res[i] = ws[0].cpu().numpy()
    