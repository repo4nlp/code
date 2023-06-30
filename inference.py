#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, argparse, torch, os, random, string, models
import numpy as np
from sklearn.metrics import f1_score
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
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


def read_json(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def data_prepare_other_test(path, name_list):
    return [read_json(path+'/'+ e +'.json') for e in name_list]


def text2prompt(data, tokenizer, label2id, prompt_text):
    data_x, data_label = [], []
    prompt = [prompt_text+" %s ."%tokenizer.mask_token]
    prompt_len = len(tokenizer(prompt[0], return_tensors="pt").input_ids[0][1:-1])
    for s in data:
        sent = " ".join(s[0])
        sent_ids = tokenizer(sent, return_tensors="pt",  truncation=True, max_length=500).input_ids[0][1:-1]
        upper_bound = 510 - prompt_len
        sent = tokenizer.convert_ids_to_tokens(sent_ids[:upper_bound])
        sent = [tokenizer.convert_tokens_to_string(sent)]
        label = s[1]
        if " ".join(sent)[-1] not in string.ascii_letters and " ".join(sent)[-1] not in string.digits:
            temp = sent + prompt
        else:
            temp = sent + ['.'] + prompt
        data_x.append(" ".join(temp))
        data_label.append(label)
    data_label = [label2id[l] for l in data_label]
    return data_x, data_label


def evaluate(args, llm, model, eval_x, eval_label, label2id):
    model.eval()
    llm.eval()
    with torch.no_grad():
        nb_iter = len(eval_x)//args.eval_batch_size if len(eval_x)>args.eval_batch_size else 1
        N=0
        corr=0
        pred_list=[]
        true_list=[]
        for iter_i in range(nb_iter+1): 
            a = iter_i*args.eval_batch_size
            b = (iter_i+1)*args.eval_batch_size if (iter_i+1)*args.eval_batch_size<=len(eval_x) else len(eval_x)
            if a==b:
                continue
            x = eval_x[a:b] if len(eval_x)>args.eval_batch_size else eval_x
            x = [llm(x_i) for x_i in x]
            x = FloatTensor(np.array([x_i.detach().cpu().numpy() for x_i in x]))
            try:           
                logits = model(x)
            except:
                print("logits = model(x)", x)
            y = eval_label[a:b] if len(eval_x)>args.eval_batch_size else eval_label
            d = {}
            for pred, true in zip(logits, y):
                for key in label2id.keys():
                    d[label2id[key]] = pred[label2id[key]].detach().cpu().numpy()
                p = sorted(d.items(), key=lambda x:x[1], reverse=True)[0][0]
                pred_list.append(p)
                true_list.append(true)
                N+=1
                if p==true:
                    corr+=1
    acc=corr/float(N)
    return acc, pred_list, true_list


def args_init():
    parser=argparse.ArgumentParser()
    parser.add_argument('--test_dir', type = str, default = '/')
    parser.add_argument('-m', '--model_name_or_path', type = str, default = 'bert-base-cased')
    parser.add_argument('-d', '--debias_model', type = str, default = 'best_bert-base-cased_epoch100_bz10_lr1e-05_The_sentiment_is_positive_neutral_negative.pkl')
    parser.add_argument('-s', '--senti_data', type = str, default ='sentiment_dataset')
    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--eval_batch_size', type = int, default = 50)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--nb_epoch', type = int, default = 10)
    parser.add_argument('--max_len', type = int, default = 64) #apply it only to training set to avoid OOM
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--pos', type = str, default = 'great')
    parser.add_argument('--neu', type = str, default = 'okay')
    parser.add_argument('--neg', type = str, default = 'terrible')
    parser.add_argument('--prompt_text', type = str, default = 'It was')
    return parser.parse_args()


#%%
if __name__ == '__main__':
    args = args_init()
    set_seed(args.seed)
    #initialization
    print("load llm...")
    llm = models.LLM(args).to(device)
    llm.eval()
    print("load best debais model...") 
    model = torch.load(f"./model_save/{args.debias_model}")
    tokenizer = llm.tokenizer
    if args.model_name_or_path.split('-')[0]=='roberta':
        label2id_2 = {'positive':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.pos),
                       'negative':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.neg)}
        label2id_3 = {'positive':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.pos),
                       'negative':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.neg),
                       'neutral':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.neu)}
        
    elif args.model_name_or_path.split('-')[0]=='bert':
        label2id_2 = {'positive':llm.tokenizer.convert_tokens_to_ids(args.pos),
                      'negative':llm.tokenizer.convert_tokens_to_ids(args.neg)}
        label2id_3 = {'positive':llm.tokenizer.convert_tokens_to_ids(args.pos),
                      'negative':llm.tokenizer.convert_tokens_to_ids(args.neg),
                      'neutral':llm.tokenizer.convert_tokens_to_ids(args.neu)}
        
    else:
        print("not support other models")
    label2id = {2:label2id_2, 3:label2id_3}
    prompt_text = model.prompt_text

    path = args.senti_data 
    name_list = ['sst2']#, 'imdb', 'yelp', 'amazon', 'debate', 'airline', 'phrasebank']
    print("load sentiment datasets...")
    data_list = data_prepare_other_test(path, name_list)
    res={}
    for k, e in zip(name_list, data_list):
        data_x, data_label = text2prompt(e, tokenizer, label2id[3], prompt_text)
        #data_label = [label2id[l] for l in data_label]
        res[k] = (data_x, data_label)

    for k in name_list:
        data_x, data_label = res[k]
        acc, pred_list, true_list = evaluate(args, llm, model, data_x, data_label, label2id[len(set(data_label))])
        f1 = f1_score(true_list, pred_list, average='macro')  
        print(k, f"instances:{len(pred_list)}  acc/Macro-f1: {acc}/{f1}")