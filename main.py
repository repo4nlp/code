#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import collections as col
import json, argparse, torch, os, random, string
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


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
    
    
def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)
        
        
def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def read(path):
    data=[]
    with open(path, "r") as f:
        for line in f.readlines():
            temp = line.strip().split('\t')
            if temp!=[]:
                data.append(temp)
    return data


class LLM(nn.Module):
    def __init__(self, args):
        super(LLM, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    
    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        logits = outputs.logits[0]
        mask_idx = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0]
        idx = list(inputs['input_ids'].cpu().numpy()[0]).index(mask_idx)
        logits_mask = logits[idx]
        #logits_mask = F.softmax(logits_mask, dim=0)
        return logits_mask


def data_prepare(path, name_list):
    return [read_json(path+'/'+ e +'.json') for e in name_list]


def text2prompt(data):
    data_x, data_label = [], []
    for s in data:
        sent = s[0][:400]
        label = s[1]
        prompt = ["The sentiment is <mask> ."]
        if " ".join(sent)[-1] not in string.ascii_letters:
            temp = sent + prompt
        else:
            temp = sent + ['.'] + prompt
        data_x.append(" ".join(temp))
        data_label.append(label)
    return data_x, data_label


class LLM(nn.Module):
    def __init__(self, args):
        super(LLM, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        self.model = AutoModelForMaskedLM.from_pretrained(args.model_name_or_path)
    
    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", truncation=True, max_length=450).to(device)
        outputs = self.model(**inputs)
        logits = outputs.logits[0]
        mask_idx = self.tokenizer.convert_tokens_to_ids([self.tokenizer.mask_token])[0]
        idx = list(inputs['input_ids'].cpu().numpy()[0]).index(mask_idx)
        logits_mask = logits[idx]
        #logits_mask = F.softmax(logits_mask, dim=0)
        return logits_mask


def eval_using_cali(d, llm):
    llm.eval()
    for name in d.keys():
        print(f"start evaluating {name} dataset")
        data_x, data_label = d[name]
        print("%d instances are found!"%(len(data_x)))
        if set(data_label)==2:
            prob_var = ['positive', 'negative']
            print(name, "2 classes", prob_var)
        else:
            prob_var = ['positive', 'neutral', 'negative']
            print(name, "3 classes", prob_var)
        
        with torch.no_grad():
            p_cf_list=[]
            for i, x in enumerate(data_x):
                if i%100==0: print(i)
                logit_vocab =llm(x)
                p_cf = [np.exp(logit_vocab[label2id[p_v]].detach().numpy()) for p_v in prob_var]
                      
                p_cf_sum = np.sum(p_cf)
                p_cf = p_cf/p_cf_sum
                p_cf_list.append(p_cf)
         
        pred=[]
        for e in p_cf_list:
            idx = np.argmax(e)
            pred.append(prob_var[idx])
            
        corr=0
        for p, t in zip(pred, data_label):
            if p==t:
                corr+=1
        print(name, corr/len(data_label))


def args_init():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type = str, default = 'roberta-base')
    parser.add_argument('--batch_size', type = int, default = 1)
    parser.add_argument('--eval_batch_size', type = int, default = 50)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--nb_epoch', type = int, default = 10)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--pos', type = str, default = 'positive')
    parser.add_argument('--neu', type = str, default = 'neutral')
    parser.add_argument('--neg', type = str, default = 'negative')
    parser.add_argument('--nb_class', type = int, default = 2)
    return parser.parse_args()

#%%
if __name__ == '__main__':
    args = args_init()
    set_seed(args.seed)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')

    llm = LLM(args).to(device)
    llm.eval()
    label2id = {'positive':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.pos),
                'negative':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.neg),
                'neutral':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.neu)}

    
    name_list1 = ['phrasebank', 'yelp', 'debate', 'airline', 'sst2', 'amazon', 'yelp', 'imdb']
    path1 = './sentiment_7_dataset'
    data_list1 = data_prepare(path1, name_list1)
    d1={}
    for k, e in zip(name_list1, data_list1):
        data_x, data_label = text2prompt(e)
        d1[k] = (data_x, data_label)
    eval_using_cali(d1, llm)
    
    #%%
    d2={}
    for name in name_list2:
        files= os.listdir(path2+'/'+name)
        temp = []
        label = []
        for file in files:
            data = read_json(path2+'/'+name+'/'+file)
            for e in data:
                temp.append([e['doc'].split(), e['label'][0]])
                label.append(e['label'][0])
        
        label_d = dict(col.Counter(label))
        s = sum(label_d.values())
        for k in label_d.keys():
            
            label_d[k] = label_d[k]/float(s)
        try:
            print(name, len(temp), "pos:neu:neg = %.0f:%.0f:%.0f"%(100*label_d['positive'],100*label_d['neutral'],100*label_d['negative']))
        except:
            print(name, len(temp), "pos:neg = %.0f:%.0f"%(100*label_d['positive'], 100*label_d['negative']))
        print('\n')
        d2[name]=temp
    
    #%%
    for k in d2.keys():
        write_json('~/res/'+k+'.json', d2[k])