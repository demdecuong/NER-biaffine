'''
    Convert csv testset format into trainable format
'''

import re
import pandas as pd
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('vinai/phobert-base')

def padding_punct(s):
    s = re.sub('([.,!?()])', r' \1 ', s)
    s = re.sub('\s{2,}', ' ', s)
    return s

def save_to_json(data,label, out_path):
    print(f'Saving to {out_path} ... ')
    with open(out_path,'w',encoding='utf-8') as f:
        for src,trg in zip(data,label):
            obj = {
                'sentence' : src,
                'label' : trg 
            }
            f.write(str(json.dumps(obj,ensure_ascii=False)) + '\n')
    
# {"sentence": "Chúng_tôi hỏi nguyen ho thuy trang : “ Anh có nghĩ việc mình làm là phạm_pháp ? ” .", "label": [[2, 5, "PERSON"]]}


def find_sub_list(sl,l):
    results=[]
    sll=len(sl)
    for ind in (i for i,e in enumerate(l) if e==sl[0]):
        if l[ind:ind+sll]==sl:
            results.append((ind,ind+sll-1))
    return results

if __name__ == '__main__':
    # missing name is nan
    testset = pd.read_csv("person_name/human_testset.csv", encoding='utf-8')
    testset['name_1'].fillna("nan")
    testset['name_2'].fillna("nan")
    trg = testset['input'].tolist()
    name1 = testset['name_1'].tolist()
    name2 = testset['name_2'].tolist()
   
    label_tokens = []
    for n1, n2 in zip(name1, name2):
        tmp = []
        if type(n1) != float:
            tmp.append(tokenizer.tokenize(n1))
        
        if type(n2) != float:
            tmp.append(tokenizer.tokenize(n2))
    
        label_tokens.append(tmp)

    trg = [' '.join(tokenizer.tokenize(padding_punct(sent))) for sent in trg]

    label = []
    for i in range(len(trg)):
        tmp = []
        for tokens in label_tokens[i]:
            get_first = False
            if str(tokens[-1]).isnumeric():
                tokens = tokens[:-1]
                get_first = True
            indx = find_sub_list(tokens, trg[i].split(' '))
            try:
                if get_first:
                    start_idx, end_idx = indx[0][0], indx[0][1]
                else:
                    start_idx, end_idx = indx[-1][0], indx[-1][1]
            except:
                print(indx)
                print(tokens)
                print(trg[i])
                print(i)
            tmp.append([start_idx,end_idx,"PERSON"])
        label.append(tmp)

    save_to_json(trg,label,"person_name/human_test.json")