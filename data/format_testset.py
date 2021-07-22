'''
    Convert csv testset format into trainable format
'''
import os
import re
import random
import pandas as pd
import json
from transformers import AutoTokenizer
from vncorenlp import VnCoreNLP

# tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')
# tokenizer = VnCoreNLP("../vncorenlp/VnCoreNLP-1.1.1.jar", annotators="wseg", max_heap_size='-Xmx500m')

def padding_punct(s):
    s = re.sub('([.,!?()])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s.strip()

def save_to_json(data,label, out_path):
    print(f'Saving to {out_path} ... ')
    with open(out_path,'w',encoding='utf-8') as f:
        for src,trg in zip(data,label):
            obj = {
                'sentence' : src,
                'label' : trg 
            }
            f.write(str(json.dumps(obj,ensure_ascii=True)) + '\n')
    
# {"sentence": "Chúng_tôi hỏi nguyen ho thuy trang : “ Anh có nghĩ việc mình làm là phạm_pháp ? ” .", "label": [[2, 5, "PERSON"]]}


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

def format_csv(path="person_name/human_testset.csv",out_path="person_name/human_test.json"):
    testset = pd.read_csv(path, encoding='utf-8',index_col=False)
    testset['name_1'].fillna("nan")
    testset['name_2'].fillna("nan")
    trg = testset['input'].tolist()
    name1 = testset['name_1'].tolist()
    name2 = testset['name_2'].tolist()

    label_tokens = []
    for n1, n2 in zip(name1, name2):
        tmp = []
        if type(n1) != float:
            tmp.append(" ".join(n1.split()))

        if type(n2) != float:
            # tmp.append(tokenizer.tokenize(n2))
            tmp.append(" ".join(n2.split()))

        if tmp == []:
            tmp.append('nan')
            print('....')
        label_tokens.append(tmp)
        print(tmp)
    trg = [padding_punct(" ".join(sent.split())) for sent in trg] 
    label = []
    for i in range(len(trg)):
        tmp = []
        for tokens in label_tokens[i]:
            get_first = False
            if str(tokens.split(' ')[-1]).isnumeric():
                if len(tokens.split(' ')) > 1:
                    tokens = ' '.join(tokens.split(' ')[:-1])
                else:
                    tokens = tokens.split(' ')[:-1]
                get_first = True
            indx = find_sub_list(tokens.strip().split(' '), trg[i].split(' '))
            
            if indx == []:
                break
            else:
                if get_first:
                    start_idx, end_idx = indx[0][0], indx[0][1]
                else:
                    start_idx, end_idx = indx[-1][0], indx[-1][1]

                tmp.append([start_idx,end_idx,"PERSON"])
        label.append(tmp)

    save_to_json(trg,label,out_path)

def read_txt(path):
    data = []
    with open(path,'r') as f:
        for line in f:
            data.append(line.replace('\n',''))
    return data


def from_sequence_labeling_to_json(src,label):
    json_label = []
    for i in range(len(src)):
        start_idx = -1
        end_idx = -1
        tmp = []
        label_seq = label[i].split(' ')
        for idx, token in enumerate(src[i].split(' ')):
            if label_seq[idx] == 'B-PERSON':
                if start_idx != -1:
                    tmp.append([start_idx,end_idx,'PERSON'])
                start_idx = idx
                end_idx = idx
            elif label_seq[idx] == 'I-PERSON':
                end_idx += 1
            else:
                if start_idx != -1:
                    tmp.append([start_idx,end_idx,'PERSON'])
                    start_idx = -1
                    end_idx = -1
        json_label.append(tmp)
    
    return src, json_label

def format_aug_data(root_dir,out_path):
    list_dir = os.listdir(root_dir)
    for folder in list_dir:
        src = read_txt('/'.join([root_dir,folder,'data.in']))
        label = read_txt('/'.join([root_dir,folder,'label.out']))
        c = list(zip(src, label))

        random.shuffle(c)

        src, label = zip(*c)

        src = src[:5000]
        label = label[:5000]
        
        src,label = from_sequence_labeling_to_json(src,label)

        print(len(src),len(label))

    save_to_json(src,label,out_path)

if __name__ == '__main__':
    # From human_test.csv to human_test.json
    format_csv(path="person_name/human_testset_v2.csv",out_path="person_name/human_testset_v2.json")

    # format_aug_data('aug_data','person_name/train_synth.json')
    # src =[ 
    #     'hoa phương là tên ông đặt buổi hẹn nay da vàng',
    #     'chị book lịch dùm buổi gặp tại bệnh viện trung ươn VinMec vào THÁNG 4 27, 1985 thì làm sao'
    # ]
    # label = [
    #     'B-PERSON I-PERSON O O O O O O O O O',
    #     'O O O O O O O O O O O O O B-DATE I-DATE I-DATE I-DATE O O O'
    # ]
    # a,b = from_sequence_labeling_to_json(src,label)
    # print(a,b)