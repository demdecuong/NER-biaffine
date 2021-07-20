import os
import json
import pytz
import torch
import re
import datetime
import fasttext
import pandas as pd
import time

from os import path
from config import Config
from transformers import AutoTokenizer

from dataset import MyDataSet
from trainer import Trainer
from model import Model
from utils import load_model
from config import get_config
from inference import Inference

import numpy as np                                                               
import matplotlib.pyplot as plt

def padding_punct(s):
    s = re.sub('([.,!?()])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s.strip()


def verify_testset(path):
    testset = pd.read_csv(path, encoding='utf-8')
    testset['name_1'].fillna("nan")
    testset['name_2'].fillna("nan")
    code = testset['code'].tolist()
    trg = testset['input'].tolist()
    name1 = testset['name_1'].tolist()
    name2 = testset['name_2'].tolist()

    label_tokens = []
    for n1, n2 in zip(name1, name2):
        tmp = []
        if type(n1) != float:
            tmp.append(" ".join(n1.split()))

        if type(n2) != float:
            tmp.append(" ".join(n2.split()))

        if tmp == []:
            tmp.append('nan')
        label_tokens.append(tmp)

    trg = [padding_punct(" ".join(sent.split())) for sent in trg]

    label = []
    invalid_cnt = 0
    for i in range(len(trg)):
        if label_tokens[i][0] != 'nan':
            for tokens in label_tokens[i]:
                if str(tokens.split(' ')[-1]).isnumeric():
                    if len(tokens.split(' ')) > 1:
                        tokens = ' '.join(tokens.split(' ')[:-1])
                    else:
                        tokens = tokens.split(' ')[:-1]
                if tokens not in str(trg[i]):
                    print(i, trg[i])
                    invalid_cnt += 1
        else:
            continue
    print('Total invalid sample = ', invalid_cnt)
    return code, trg, label_tokens


def main(args, path, out_path):
    extractor = Inference(args)
    code, src, trg = verify_testset(path)

    correct = 0
    total = 0
    failure_samples = []
    for cod, sentence, target in zip(code, src, trg):
        output = extractor.inference(sentence)['entities']
        tmp_correct = 0
        for tokens in target:
            if str(tokens.split(' ')[-1]).isnumeric():
                if len(tokens.split(' ')) > 1:
                    tokens = ' '.join(tokens.split(' ')[:-1])
                else:
                    tokens = tokens.split(' ')[:-1]
        if output == []:
            if target[0] == 'nan':
                tmp_correct += 1
        elif target[0] == 'nan':
            if output == []:
                tmp_correct += 1
        else:
            for out in output:
                pername = out['value']
                if pername in target:
                    tmp_correct += 1
        total += len(target)
        correct += tmp_correct
        if tmp_correct != len(target):
            failure_samples.append(
                {
                    'code' : cod,
                    'input': sentence,
                    'label': target,
                    'pred' : output
                })
    print('# Correct labels = ', correct)
    print('# Failed labels = ', total-correct)
    print("Accuracy : ", correct/total, correct)

    print(f'Saving failure samples to {out_path}')
    with open(out_path,'w',encoding='utf-8') as f:
        for obj in failure_samples:
            f.write(str(json.dumps(obj,ensure_ascii=False)) + '\n')

def stats_pred_output(human_test_path,path,png_out_path):
    testset = pd.read_csv(human_test_path, encoding='utf-8')
    total_code = testset['code'].tolist()

    code = [] 
    label = []
    sentence = []
    pred_confidence = []
    sentence_len = []

    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            sentence.append(sample['input'])
            label.append(sample['label'])
            # pred_confidence.append(sample['code'])
            code.append(sample['code'])

    # Number of samples
    code_visual = {}
    for c in code:
        if c not in code_visual:
            code_visual[c] = code.count(c)
    code_visual = {k: v for k, v in sorted(code_visual.items(), key=lambda item: item[1])}
    plt.figure(figsize=(12,12))
    plt.title('Number of Failure code')
    plt.xlabel('testset code')
    plt.ylabel('number of samples')
    plt.bar(*zip(*code_visual.items()))
    plt.savefig(png_out_path)

    # Percetange of samples per code
    code_visual_per = {}
    for c in code:
        if c not in code_visual_per:
            code_visual_per[c] = code.count(c) / total_code.count(c)

    code_visual_per = {k: v for k, v in sorted(code_visual_per.items(), key=lambda item: item[1])}

    plt.figure(figsize=(12,12))
    plt.title('Number of Percentage of failure code per code')
    plt.xlabel('testset code')
    plt.ylabel('number of percentage')
    plt.axhline(y=0.5, color='r', linestyle='-')
    plt.bar(*zip(*code_visual_per.items()))
    plt.savefig('asset/failure_code_percentage.png')

    # number of sample in Most failure in percetange
    code_visual_num = {}
    for k,v in code_visual_per.items():
        code_visual_num[k] = code_visual[k]
    plt.figure(figsize=(12,12))
    plt.title('Number of sample of most failure code')
    plt.xlabel('testset code')
    plt.ylabel('number of sample')
    plt.bar(*zip(*code_visual_num.items()))
    plt.savefig('asset/failure_code_number.png')

if __name__ == "__main__":
    args = get_config()
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using Device : {args.device}')

    if torch.cuda.is_available():
        print(f"GPU device : {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

    human_test_path = "data/person_name/human_testset.csv"
    out_path = 'investigate/failure_invs.json'
    png_out_path='asset/failure_code_density.png'
    
    print('--------------------- INVESTIGATING ---------------------')
    main(args, human_test_path, out_path)
    print('--------------------- STATISTIC PREDICTION OUTPUT ---------------------')
    stats_pred_output(human_test_path, out_path,png_out_path)

