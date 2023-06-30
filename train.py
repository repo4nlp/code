#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, torch, string, utils, models, os
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import numpy as np
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
 
    
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


def data_prepare_dialogue(args, val_ratio=0.2):
    def random_num(n, m):
        val_size = int(n*m)
        internal = n//val_size
        val_list = []
        for i in range(0, n, internal):
            val_list.append(i)
        return val_list
    def prepare(data, label='positive'):
        data1=[]
        for s in data:
            data1.append([s, label])
        return data1
    data_pos = utils.read(args.pos_dir)
    data_neg = utils.read(args.neg_dir)
    data_neu = utils.read(args.neu_dir)
    data_pos = prepare(data_pos, 'positive')
    data_neg = prepare(data_neg, 'negative')
    data_neu = prepare(data_neu, 'neutral')
    data_all = data_pos + data_neg + data_neu
    val_list = random_num(len(data_all), m=val_ratio) #0.8 for training and 0.2 for validation
    data_train, data_val = [], []
    for i in range(len(data_all)):
        if i in val_list:
            data_val.append(data_all[i])
        else:
            data_train.append(data_all[i])
    return data_train, data_val


def evaluate(args, llm, model, eval_x, eval_label):
    model.eval()
    with torch.no_grad():
        nb_iter = len(eval_x)//args.eval_batch_size if len(eval_x)>args.eval_batch_size else 1
        N=0
        corr=0
        pred_list=[]
        true_list=[]
        for iter_i in range(nb_iter):
            a = iter_i*args.eval_batch_size
            b = (iter_i+1)*args.eval_batch_size if (iter_i+1)*args.eval_batch_size<=len(eval_x) else len(eval_x)
            x = eval_x[a:b] if len(eval_x)>args.eval_batch_size else eval_x
            x = [llm(x_i) for x_i in x]
            x = FloatTensor(np.array([x_i.detach().cpu().numpy() for x_i in x]))
            logits = model(x)
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
    parser.add_argument('--model_name_or_path', type = str, default = 'bert-base-cased')
    parser.add_argument('--model_dir', type = str, default = 'model_save')
    parser.add_argument('--batch_size', type = int, default = 10)
    parser.add_argument('--eval_batch_size', type = int, default = 50)
    parser.add_argument('--lr', type = float, default = 1e-5)
    parser.add_argument('--nb_epoch', type = int, default = 100)
    parser.add_argument('--pos_dir', type = str, default = './positive.txt')
    parser.add_argument('--neg_dir', type = str, default = './negative.txt')
    parser.add_argument('--neu_dir', type = str, default = './neutral.txt')
    parser.add_argument('--max_len', type = int, default = 64) #apply it only to training set to avoid OOM
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--pos', type = str, default = 'positive')
    parser.add_argument('--neu', type = str, default = 'neutral')
    parser.add_argument('--neg', type = str, default = 'negative')
    parser.add_argument('--prompt_text', type = str, default = 'The sentiment is')
    return parser.parse_args()

#%%
if __name__ == '__main__':
    args = args_init()
    utils.set_seed(args.seed)
    #initialization
    llm = models.LLM(args).to(device)
    llm.eval()#frozen the parameters of LM
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        llm = nn.DataParallel(llm)
    llm.eval()#frozen the parameters of LM
    
    if args.model_name_or_path.split('-')[0]=='roberta':
        label2id = {'positive':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.pos),
                    'negative':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.neg),
                    'neutral':llm.tokenizer.convert_tokens_to_ids(chr(288)+args.neu)}
    elif args.model_name_or_path.split('-')[0]=='bert':
        label2id = {'positive':llm.tokenizer.convert_tokens_to_ids(args.pos),
                    'negative':llm.tokenizer.convert_tokens_to_ids(args.neg),
                    'neutral':llm.tokenizer.convert_tokens_to_ids(args.neu)}
    print(f'label2id: {label2id}')
    
    data_train, data_dev = data_prepare_dialogue(args)
    train_x, train_label = text2prompt(data_train, llm.tokenizer, label2id, args.prompt_text)
    dev_x, dev_label = text2prompt(data_dev, llm.tokenizer, label2id, args.prompt_text)
    print("train: %d  val: %d"%(len(train_x), len(dev_x)))
    
    model = models.DebiasModel(llm.tokenizer.vocab_size, label2id, args.prompt_text, diagonal=True)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    criterion = nn.CrossEntropyLoss()
    
    writer = SummaryWriter(comment='debias_tensorboard')
    data_all = list(zip(train_x, train_label))
    nb_iter = len(train_x)//args.batch_size
    model.train()
    globle_step=0
    dev_acc_best=0
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    for epoch_i in range(args.nb_epoch):
        np.random.shuffle(data_all)
        train_x, train_label = zip(*data_all)
        for iter_i in range(nb_iter):
            optimizer.zero_grad()
            x = train_x[iter_i*args.batch_size:(iter_i+1)*args.batch_size]
            x = [llm(x_i) for x_i in x]
            x = FloatTensor(np.array([x_i.detach().cpu().numpy() for x_i in x]))
            y = train_label[iter_i*args.batch_size:(iter_i+1)*args.batch_size]
            y = LongTensor(y)
            #print(list(llm.named_parameters())[100][1][:10])
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss = loss.item()
            dev_acc, _, _ = evaluate(args, llm, model, dev_x, dev_label)
            if dev_acc > dev_acc_best:
                dev_acc_best = dev_acc
                best_model = model
                torch.save(best_model, f"./model_save/best_{args.model_name_or_path}_epoch{args.nb_epoch}_bz{args.batch_size}_lr{args.lr}_{'_'.join(args.prompt_text.split())}_{args.pos}_{args.neu}_{args.neg}.pkl")
            #print(f"epoch:{epoch_i}  iter:{iter_i}  train_loss:{loss}  val_acc:{dev_acc}")
            writer.add_scalar('train loss', float(loss), globle_step)
            globle_step+=1
        print(f"epoch:{epoch_i}  iter:{iter_i}  train_loss:{loss}  val_acc:{dev_acc}")   